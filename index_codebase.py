#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "source_code"
QDRANT_URL = "http://127.0.0.1:6333"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

SUPPORTED_EXTENSIONS = {
    ".swift",
    ".m",
    ".mm",
    ".h",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".kt",
    ".kts",
    ".go",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".php",
    ".rb",
    ".sh",
    ".zsh",
    ".sql",
    ".graphql",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
}

IGNORED_DIRECTORIES = {
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "node_modules",
    "Pods",
    "DerivedData",
    ".build",
    "build",
    "dist",
    "coverage",
    "__pycache__",
}

MAX_FILE_BYTES = 2_000_000
CHUNK_LINES = 80
CHUNK_OVERLAP = 15
BATCH_SIZE = 64


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    path: str
    language: str
    start_line: int
    end_line: int
    content: str
    file_hash: str


def language_for(path: Path) -> str:
    mapping = {
        ".swift": "swift",
        ".m": "objective-c",
        ".mm": "objective-c++",
        ".h": "header",
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cc": "c++",
        ".cpp": "c++",
        ".cs": "c-sharp",
        ".php": "php",
        ".rb": "ruby",
        ".sh": "shell",
        ".zsh": "shell",
        ".sql": "sql",
        ".graphql": "graphql",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
    }

    return mapping.get(path.suffix.lower(), "text")


def should_index(path: Path) -> bool:
    if not path.is_file():
        return False

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False

    if any(part in IGNORED_DIRECTORIES for part in path.parts):
        return False

    try:
        return path.stat().st_size <= MAX_FILE_BYTES
    except OSError:
        return False


def iter_source_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if should_index(path):
            yield path


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_chunks(root: Path, path: Path) -> list[CodeChunk]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    relative_path = str(path.relative_to(root))
    file_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    lines = raw.splitlines()

    if not lines:
        return []

    chunks: list[CodeChunk] = []
    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)

    for start_index in range(0, len(lines), step):
        end_index = min(start_index + CHUNK_LINES, len(lines))
        body = "\n".join(lines[start_index:end_index]).strip()

        if not body:
            continue

        start_line = start_index + 1
        end_line = end_index

        embedding_text = (
            f"File: {relative_path}\n"
            f"Language: {language_for(path)}\n"
            f"Lines: {start_line}-{end_line}\n\n"
            f"{body}"
        )

        identity = (
            f"{relative_path}:"
            f"{start_line}:"
            f"{end_line}:"
            f"{file_hash}"
        )

        chunks.append(
            CodeChunk(
                chunk_id=stable_id(identity),
                path=relative_path,
                language=language_for(path),
                start_line=start_line,
                end_line=end_line,
                content=embedding_text,
                file_hash=file_hash,
            )
        )

        if end_index == len(lines):
            break

    return chunks


def batched(items: list[CodeChunk], size: int) -> Iterator[list[CodeChunk]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", help="Path to the source-code repository")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the collection",
    )
    args = parser.parse_args()

    root = Path(args.project).expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    vector_size = model.get_sentence_embedding_dimension()

    if vector_size is None:
        raise RuntimeError("Could not determine embedding dimension")

    client = QdrantClient(url=QDRANT_URL)

    collection_exists = client.collection_exists(COLLECTION_NAME)

    if args.recreate and collection_exists:
        print(f"Deleting collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
        collection_exists = False

    if not collection_exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    chunks: list[CodeChunk] = []

    for path in iter_source_files(root):
        chunks.extend(create_chunks(root, path))

    print(f"Indexing {len(chunks)} chunks")

    for number, batch in enumerate(batched(chunks, BATCH_SIZE), start=1):
        vectors = model.encode(
            [chunk.content for chunk in batch],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=vector.tolist(),
                payload={
                    "path": chunk.path,
                    "language": chunk.language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "file_hash": chunk.file_hash,
                    "content": chunk.content,
                },
            )
            for chunk, vector in zip(batch, vectors, strict=True)
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(f"Uploaded batch {number}")

    print("Index complete.")


if __name__ == "__main__":
    main()