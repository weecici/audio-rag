import os
import sys
import argparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient


def list_collections(client: QdrantClient) -> None:
    collections = client.get_collections().collections
    print(f"Existing Qdrant collections: {collections}")


def display_collection_info(client: QdrantClient, collection_name: str) -> None:
    try:
        info = client.get_collection(collection_name=collection_name)
        print(f"Information for collection '{collection_name}':")
        print(f"  - Points count: {info.points_count}")
        print(f"  - Vectors count: {info.vectors_count}")
        print(f"  - Indexed vectors count: {info.indexed_vectors_count}")
        print(f"  - Segments count: {info.segments_count}")

        if isinstance(info.config.params.vectors, dict):
            for vector_name, vector_params in info.config.params.vectors.items():
                print(f"  - Vector params ('{vector_name}'):")
                print(f"    + Size: {vector_params.size}")
                print(f"    + Distance: {vector_params.distance}")

        if isinstance(info.config.params.sparse_vectors, dict):
            for sparse_name, sparse_params in info.config.params.sparse_vectors.items():
                print(f"  - Sparse vector params ('{sparse_name}'):")
                print(f"    + Type: {type(sparse_params)}")

    except Exception:
        print(f"Collection '{collection_name}' does not exist.")


def delete_collection(client: QdrantClient, collection_name: str) -> None:
    client.delete_collection(collection_name=collection_name)
    print(f"Deleted collection: {collection_name}")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Utils for testing/managing Qdrant collections."
    )

    arg_specs = [
        {
            "flags": ["--list"],
            "kwargs": {
                "action": "store_true",
                "help": "List all existing Qdrant collections.",
            },
        },
        {
            "flags": ["--check"],
            "kwargs": {
                "type": str,
                "metavar": "COLLECTION_NAME",
                "help": "Check a specific collection and show its info.",
            },
        },
        {
            "flags": ["--delete"],
            "kwargs": {
                "type": str,
                "metavar": "COLLECTION_NAME",
                "help": "Delete a specific collection.",
            },
        },
    ]

    group = parser.add_mutually_exclusive_group(required=True)
    for spec in arg_specs:
        group.add_argument(*spec["flags"], **spec["kwargs"])  # single place usage

    args = parser.parse_args()

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    try:
        if args.list:
            list_collections(client)
        elif args.check is not None:
            display_collection_info(client, args.check)
        elif args.delete is not None:
            delete_collection(client, args.delete)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)
