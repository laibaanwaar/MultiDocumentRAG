from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

EMBEDDING_DIMENSION = int(
    os.getenv(
        "EMBEDDING_DIMENSION",
        "384",
    )
)

EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cpu",
)

EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "EMBEDDING_BATCH_SIZE",
        "32",
    )
)

NORMALIZE_EMBEDDINGS = (
    os.getenv(
        "NORMALIZE_EMBEDDINGS",
        "True",
    ).lower()
    == "true"
)

SHOW_EMBEDDING_PROGRESS = (
    os.getenv(
        "SHOW_EMBEDDING_PROGRESS",
        "True",
    ).lower()
    == "true"
)


def validate_embedding_settings() -> None:
    if not EMBEDDING_MODEL_NAME.strip():
        raise ValueError(
            "EMBEDDING_MODEL cannot be empty."
        )

    if EMBEDDING_DIMENSION <= 0:
        raise ValueError(
            "EMBEDDING_DIMENSION must be greater than zero."
        )

    if EMBEDDING_BATCH_SIZE <= 0:
        raise ValueError(
            "EMBEDDING_BATCH_SIZE must be greater than zero."
        )

    if not EMBEDDING_DEVICE.strip():
        raise ValueError(
            "EMBEDDING_DEVICE cannot be empty."
        )


def get_embedding_dimension() -> int:
    return EMBEDDING_DIMENSION


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    validate_embedding_settings()

    print(
        "Loading embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )
    print(
        "Embedding device: "
        f"{EMBEDDING_DEVICE}"
    )

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": EMBEDDING_DEVICE,
        },
        encode_kwargs={
            "batch_size": EMBEDDING_BATCH_SIZE,
            "normalize_embeddings": (
                NORMALIZE_EMBEDDINGS
            ),
        },
    )


def clean_embedding_text(text: str) -> str:
    return " ".join(text.split()).strip()


def embed_query(text: str) -> list[float]:
    cleaned_text = clean_embedding_text(text)

    if not cleaned_text:
        raise ValueError(
            "Query text cannot be empty."
        )

    model = get_embedding_model()
    vector = model.embed_query(cleaned_text)

    validate_vector_dimension(
        vector=vector,
        context="query",
    )

    return vector


def embed_documents(
    texts: Sequence[str],
) -> list[list[float]]:
    if not texts:
        return []

    cleaned_texts: list[str] = []

    for index, text in enumerate(
        texts,
        start=1,
    ):
        cleaned_text = clean_embedding_text(text)

        if not cleaned_text:
            raise ValueError(
                "Document text cannot be empty. "
                f"Invalid item position: {index}."
            )

        cleaned_texts.append(cleaned_text)

    model = get_embedding_model()
    vectors = model.embed_documents(
        cleaned_texts
    )

    if len(vectors) != len(cleaned_texts):
        raise RuntimeError(
            "Embedding count does not match document count. "
            f"Documents: {len(cleaned_texts)}, "
            f"vectors: {len(vectors)}."
        )

    for index, vector in enumerate(
        vectors,
        start=1,
    ):
        validate_vector_dimension(
            vector=vector,
            context=f"document {index}",
        )

    return vectors


def validate_vector_dimension(
    vector: Sequence[float],
    context: str,
) -> None:
    actual_dimension = len(vector)

    if actual_dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            "Embedding dimension mismatch for "
            f"{context}. Expected "
            f"{EMBEDDING_DIMENSION}, received "
            f"{actual_dimension}. "
            "Recheck EMBEDDING_MODEL and "
            "EMBEDDING_DIMENSION."
        )


def test_embedding() -> None:
    sample_query = (
        "What is the punishment for theft under "
        "the Pakistan Penal Code?"
    )

    sample_documents = [
        (
            "The Pakistan Penal Code contains "
            "criminal offences and punishments."
        ),
        (
            "The Constitution of Pakistan contains "
            "constitutional rights and state principles."
        ),
    ]

    query_vector = embed_query(
        sample_query
    )
    document_vectors = embed_documents(
        sample_documents
    )

    print("\nEmbedding test completed successfully.")
    print(
        f"Model: {EMBEDDING_MODEL_NAME}"
    )
    print(
        f"Configured dimension: "
        f"{get_embedding_dimension()}"
    )
    print(
        f"Query vector dimension: "
        f"{len(query_vector)}"
    )
    print(
        "Document vectors generated: "
        f"{len(document_vectors)}"
    )
    print(
        "First 10 query values: "
        f"{query_vector[:10]}"
    )


if __name__ == "__main__":
    try:
        test_embedding()

    except Exception as error:
        print(
            f"\nEmbedding error: {error}"
        )
        raise
