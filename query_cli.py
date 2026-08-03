from __future__ import annotations

from typing import Any

from rag.rag_chain import (
    answer_question,
    create_rag_components,
)


def display_answer(result: dict[str, Any]) -> None:
    """Print the answer and its supporting legal sources."""

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.get("answer", "No answer generated."))

    sources = result.get("sources", [])

    if not sources:
        print("\nNo supporting sources were found.")
        return

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in sources:
        provision_type = (
            source.get("provision_type")
            or "Provision"
        ).title()

        provision_number = (
            source.get("provision_number")
            or "Unsectioned"
        )

        print(
            f"- [{source.get('label', 'Source')}] "
            f"{source.get('document_name', 'Unknown document')} | "
            f"{provision_type} {provision_number} | "
            f"Pages: {source.get('page_range', 'Unknown')}"
        )


def main() -> None:
    """Run the command-line legal RAG assistant."""

    client = None

    try:
        print("=" * 70)
        print("PAKISTAN MULTI-DOCUMENT LEGAL RAG ASSISTANT")
        print("=" * 70)
        print(
            "Indexed laws: PPC, Constitution, ATA, and AMLA"
        )
        print(
            "Opening the existing Qdrant collection..."
        )

        retriever, chat_model, client = (
            create_rag_components()
        )

        print(
            "RAG assistant is ready. "
            "Type 'exit' or 'quit' to stop."
        )

        while True:
            question = input(
                "\nEnter your question: "
            ).strip()

            if question.lower() in {
                "exit",
                "quit",
            }:
                print(
                    "Closing the RAG assistant."
                )
                break

            if len(question) < 3:
                print(
                    "Please enter a valid question."
                )
                continue

            print(
                "\nRetrieving context and "
                "generating the answer..."
            )

            result = answer_question(
                question=question,
                retriever=retriever,
                chat_model=chat_model,
            )

            display_answer(result)

    except KeyboardInterrupt:
        print(
            "\nRAG assistant stopped."
        )

    except Exception as error:
        print(
            f"\nRAG error: {error}"
        )

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()