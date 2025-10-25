import os
import sys
import argparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient


def list_collections(client: QdrantClient) -> None:
    collections = client.get_collections().collections
    print(f"Existing Qdrant collections: {collections}")


def check_collection_exists(client: QdrantClient, collection_name: str) -> bool:
    collections = client.get_collections().collections
    return any(col.name == collection_name for col in collections)


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
                "help": "Check if a specific collection exists.",
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
            exists = check_collection_exists(client, args.check)
            print(
                f"Collection '{args.check}' {'exists' if exists else 'does not exist'}."
            )
        elif args.delete is not None:
            delete_collection(client, args.delete)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)
