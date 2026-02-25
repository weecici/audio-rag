import uuid
import re
import asyncio
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from app import schema
from .chunking import chunk_text

_TIMESTAMP_LINE_RE = re.compile(
    r"^\s*\[(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s\]\s+.+$"
)


def _is_transcript_file(filepath: str) -> bool:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Check up to the first 10 lines to find the first non-empty line
            for _ in range(10):
                line = f.readline()
                if line == "":  # EOF
                    break
                s = line.strip()
                if not s:
                    continue
                return bool(_TIMESTAMP_LINE_RE.match(s))
    except Exception as e:
        raise RuntimeError(f"Error reading file {filepath}: {e}")
    return False


def _load_documents(file_paths: list[str], file_dir: str) -> list[Document]:
    docs: list[Document] = []

    if file_dir:
        base_dir = Path(file_dir)
        if base_dir.exists():
            for path in base_dir.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    loader = TextLoader(str(path), encoding="utf-8")
                    docs.extend(loader.load())
                except Exception:
                    continue

    for fp in file_paths or []:
        try:
            loader = TextLoader(fp, encoding="utf-8")
            docs.extend(loader.load())
        except Exception:
            continue

    return docs


async def process_documents(file_paths: list[str], file_dir: str) -> list[Document]:

    docs = _load_documents(file_paths=file_paths, file_dir=file_dir)

    nodes: list[Document] = []
    docs_info: list[tuple[str, str, str]] = []  # (audio_url, audio_title, filepath)
    tasks = []
    for doc in docs:
        filepath = doc.metadata.get("source", "unknown")
        filename = Path(filepath).stem
        parts = filename.split("$")

        audio_title = parts[0].strip()
        audio_url = parts[1].strip() if len(parts) == 2 else audio_title

        docs_info.append((audio_url, audio_title, filepath))

        text_type = "transcript" if _is_transcript_file(filepath) else "document"
        tasks.append(
            asyncio.create_task(
                chunk_text(raw_text=doc.page_content, text_type=text_type)
            )
        )

    all_chunks: list[list[tuple[str, str]]] = await asyncio.gather(*tasks)

    for i, chunks in enumerate(all_chunks):
        audio_url, audio_title, filepath = docs_info[i]

        try:
            if chunks == []:
                raise ValueError("No chunks returned from chunking process")
            for title, chunk in chunks:
                node_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{title}_{audio_url}"))

                metadata = schema.DocumentMetadata(
                    document_id=audio_url,
                    title=title,
                    file_name=audio_title,
                    file_path=filepath,
                )

                node = Document(
                    page_content=chunk,
                    metadata={"id": node_id, **metadata.model_dump()},
                )
                nodes.append(node)
        except Exception as e:
            print(f"Error processing chunks for {audio_title}: {e}")
            continue

    return nodes
