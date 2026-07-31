import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()


def get_embedding_dimension() -> int:
    """Return the configured embedding vector size."""

    return 384


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Create and return the local sentence-transformers embedding model."""

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


def test_embedding() -> None:
    """Generate one sample embedding for testing."""

    embedding_model = get_embedding_model()

    sample_text = (
        "The Pakistan Penal Code provides a general "
        "penal code for Pakistan."
    )

    vector = embedding_model.embed_query(sample_text)

    print("Embedding created successfully.")
    print(f"Model: {os.getenv('EMBEDDING_MODEL')}")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 10 values: {vector[:10]}")


if __name__ == "__main__":
    try:
        test_embedding()

    except Exception as error:
        print(f"\nEmbedding error: {error}")
