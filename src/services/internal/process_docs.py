import uuid
from src.core import config
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)


def process_documents(file_paths: list[str], file_dir: str) -> list[TextNode]:

    reader = SimpleDirectoryReader(input_files=file_paths, input_dir=file_dir)
    docs = reader.load_data()

    nodes = []
    for doc in docs:
        doc.doc_id = doc.metadata.get("file_name", "unknown").split(".")[0]
        chunks = splitter.split_text(doc.text)
        for idx, chunk in enumerate(chunks):
            node_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc.doc_id}_{idx}"))

            node = TextNode(
                id_=node_id,
                text=chunk,
                metadata={
                    "document_id": doc.doc_id,
                    "title": doc.metadata.get("title", "none"),
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "file_path": doc.metadata.get("file_path", "unknown"),
                },
            )
            nodes.append(node)

    return nodes
