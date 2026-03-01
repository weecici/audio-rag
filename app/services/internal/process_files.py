"""Internal service: load files, chunk, generate titles, embed -> Document list.

Supports multiple file types via LangChain loaders:
- .txt, .md  -> TextLoader
- .pdf       -> PyPDFLoader
- .docx/.doc -> Docx2txtLoader

Each file is processed independently.  The public ``process_single_file``
coroutine handles one file end-to-end (load -> chunk -> title -> embed)
and is designed to be fanned-out with ``asyncio.gather`` for concurrency.

``process_files`` is the batch entry-point that processes all files in
parallel.
"""

import asyncio
import hashlib
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

from app.core.config import settings
from app.core.logging import logger
from app.models import Document
from .chunk import chunk_text, generate_titles, TextChunk
from .embed import dense_embed


# ---------------------------------------------------------------------------
# File-type -> loader mapping
# ---------------------------------------------------------------------------

_LOADER_MAP: dict[str, type] = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}


def _load_text(fpath: Path) -> str:
    """Load text content from a file using the appropriate LangChain loader.

    Returns the concatenated page content, or empty string on failure.
    """
    ext = fpath.suffix.lower()
    loader_cls = _LOADER_MAP.get(ext)

    if loader_cls is None:
        # Fallback: try TextLoader for unknown extensions
        logger.warning(f"No specific loader for '{ext}', falling back to TextLoader")
        loader_cls = TextLoader

    try:
        if loader_cls is TextLoader:
            loader = loader_cls(str(fpath), encoding="utf-8")
        else:
            loader = loader_cls(str(fpath))
        lc_docs = loader.load()
        return "\n".join(d.page_content for d in lc_docs)
    except Exception as exc:
        logger.error(f"Failed to load {fpath} with {loader_cls.__name__}: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Deterministic doc ID
# ---------------------------------------------------------------------------


def _stable_doc_id(source: str, chunk_index: int) -> int:
    """Produce a deterministic 63-bit integer ID from source path + chunk index.

    Using a hash avoids collisions across different files while keeping IDs
    reproducible (idempotent upserts).
    """
    raw = f"{source}::{chunk_index}".encode()
    h = hashlib.sha256(raw).hexdigest()
    return int(h[:15], 16)  # 60-bit positive int, fits in INT64


# ---------------------------------------------------------------------------
# Single-file processing (designed for concurrent fan-out)
# ---------------------------------------------------------------------------


async def process_single_file(fpath: Path) -> list[Document]:
    """Process one text file end-to-end: load -> chunk -> title -> embed.

    Returns a list of Document objects (one per chunk). Returns an empty
    list on failure (logged, not raised).
    """
    path = Path(fpath)
    if not path.exists():
        logger.warning(f"File not found, skipping: {fpath}")
        return []

    # 1. Load -- run in executor since PDF/DOCX loaders do I/O
    loop = asyncio.get_running_loop()
    full_text = await loop.run_in_executor(None, _load_text, path)

    if not full_text.strip():
        logger.warning(f"Empty file, skipping: {fpath}")
        return []

    # 2. Chunk
    chunks = chunk_text(full_text, source=str(path))
    logger.info(f"Loaded {path.name}: {len(chunks)} chunks")

    if not chunks:
        return []

    # 3. Generate titles via Cerebras (concurrent per chunk internally)
    if settings.TITLE_GEN_ENABLED:
        logger.info(f"Generating titles for {len(chunks)} chunks from {path.name}...")
        chunks = await generate_titles(chunks)
    else:
        logger.info(f"Title generation disabled, skipping title gen...")

    # 4. Embed all chunks via Google Gemini
    logger.info(f"Embedding {len(chunks)} chunks from {path.name}...")
    titles = [c.title for c in chunks]
    texts = [c.text for c in chunks]
    vectors = await dense_embed(texts, titles)

    # 5. Assemble Document objects
    documents: list[Document] = []
    for chunk, vector in zip(chunks, vectors):
        doc = Document(
            doc_id=_stable_doc_id(chunk.source, chunk.index),
            title=chunk.title,
            metadata={
                "source": chunk.source,
                "chunk_index": chunk.index,
                "source_filename": Path(chunk.source).name,
            },
            text=chunk.text,
            dense_vector=vector,
        )
        documents.append(doc)

    logger.info(f"Processed {len(documents)} chunks from {path.name}")
    return documents


# ---------------------------------------------------------------------------
# Batch entry-point (backward-compatible)
# ---------------------------------------------------------------------------


async def process_files(file_paths: list[Path]) -> list[Document]:
    """Process multiple text files concurrently.

    Each file is processed independently via ``process_single_file``,
    all running in parallel with ``asyncio.gather``.
    """
    if not file_paths:
        return []

    # Fan-out: process all files concurrently
    results = await asyncio.gather(
        *[process_single_file(fpath) for fpath in file_paths],
        return_exceptions=True,
    )

    all_docs: list[Document] = []
    for fpath, result in zip(file_paths, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to process {fpath}: {result}")
            continue
        all_docs.extend(result)

    logger.info(
        f"Processed {len(all_docs)} document chunks from {len(file_paths)} files"
    )
    return all_docs
