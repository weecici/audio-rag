import uuid
import re
import asyncio
from src import schemas
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from ._chunk import chunk_text


splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)


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


async def process_documents(file_paths: list[str], file_dir: str) -> list[TextNode]:

    reader = SimpleDirectoryReader(input_files=file_paths, input_dir=file_dir)
    docs = reader.load_data()

    nodes: list[TextNode] = []
    tasks = []
    docs_info: list[tuple[str, str, str]] = []  # (document_id, filename, filepath)
    for doc in docs:
        filepath = doc.metadata.get("file_path", "unknown")
        filename = Path(doc.metadata.get("file_name", "unknown")).stem
        parts = filename.split("$")

        audio_title = parts[0].strip()
        audio_url = parts[1].strip() if len(parts) == 2 else audio_title

        doc.doc_id = audio_url
        docs_info.append((audio_url, filename, filepath))

        text_type = "transcript" if _is_transcript_file(filepath) else "document"
        tasks.append(
            asyncio.create_task(chunk_text(raw_text=doc.text, text_type=text_type))
        )

    all_chunks: list[list[tuple[str, str]]] = await asyncio.gather(*tasks)

    for i, chunks in enumerate(all_chunks):
        document_id, filename, filepath = docs_info[i]
        for title, chunk in chunks:
            node_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{title}_{document_id}"))

            metadata = schemas.DocumentMetadata(
                document_id=document_id,
                title=title,
                file_name=filename,
                file_path=filepath,
            )

            node = TextNode(id_=node_id, text=chunk, metadata=metadata.model_dump())
            nodes.append(node)

    return nodes
