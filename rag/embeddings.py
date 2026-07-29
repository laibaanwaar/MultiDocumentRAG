import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()


def get_embedding_dimension() -> int:
    """Return the configured embedding vector size."""

    return int(
        os.getenv("EMBEDDING_DIMENSION", "768")
    )


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Create and return the Gemini embedding model."""

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "gemini-embedding-2",
    )

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing from the .env file."
        )

    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key,
        output_dimensionality=get_embedding_dimension(),
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