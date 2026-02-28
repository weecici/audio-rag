"""Internal service: load files, chunk, generate titles, embed -> Document list."""

import hashlib
from pathlib import Path
from langchain_community.document_loaders import TextLoader

from app.core.logging import logger
from app.models import Document
from .chunk import chunk_text, generate_titles, TextChunk
from .embed import dense_encode


def _stable_doc_id(source: str, chunk_index: int) -> int:
    """Produce a deterministic 63-bit integer ID from source path + chunk index.

    Using a hash avoids collisions across different files while keeping IDs
    reproducible (idempotent upserts).
    """
    raw = f"{source}::{chunk_index}".encode()
    h = hashlib.sha256(raw).hexdigest()
    return int(h[:15], 16)  # 60-bit positive int, fits in INT64


async def process_files(file_paths: list[Path]) -> list[Document]:
    """End-to-end pipeline: load -> chunk -> title -> embed -> Document list.

    Currently handles text files only.  Audio support is deferred.
    """
    if not file_paths:
        return []

    # 1. Load text from each file
    all_chunks: list[TextChunk] = []
    for fpath in file_paths:
        path = Path(fpath)
        if not path.exists():
            logger.warning(f"File not found, skipping: {fpath}")
            continue

        try:
            loader = TextLoader(str(path), encoding="utf-8")
            lc_docs = loader.load()
            full_text = "\n".join(d.page_content for d in lc_docs)
        except Exception as exc:
            logger.error(f"Failed to load {fpath}: {exc}")
            continue

        if not full_text.strip():
            logger.warning(f"Empty file, skipping: {fpath}")
            continue

        chunks = chunk_text(full_text, source=str(path))
        all_chunks.extend(chunks)
        logger.info(f"Loaded {path.name}: {len(chunks)} chunks")

    if not all_chunks:
        return []

    # 2. Generate titles via Cerebras
    logger.info(f"Generating titles for {len(all_chunks)} chunks...")
    all_chunks = await generate_titles(all_chunks)

    # 3. Embed all chunks via Google Gemini
    logger.info(f"Embedding {len(all_chunks)} chunks...")
    titles = [c.title for c in all_chunks]
    texts = [c.text for c in all_chunks]
    vectors = await dense_encode(texts, titles)

    # 4. Assemble Document objects
    documents: list[Document] = []
    for chunk, vector in zip(all_chunks, vectors):
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

    logger.info(
        f"Processed {len(documents)} document chunks from {len(file_paths)} files"
    )
    return documents
