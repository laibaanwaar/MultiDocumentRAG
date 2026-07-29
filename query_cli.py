from rag.rag_chain import (
    answer_question,
    create_rag_components,
)


def display_answer(result: dict) -> None:
    """Display the generated answer and source references."""

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    sources = result.get("sources", [])

    if not sources:
        print("No supporting sources were found.")
        return

    for source in sources:
        print(
            f"- [{source['label']}] "
            f"{source['document_name']} | "
            f"Page: {source['page_number']} | "
            f"Chunk: {source['chunk_number']}"
        )


def main() -> None:
    client = None

    try:
        print("=" * 70)
        print("PAKISTAN PENAL CODE RAG ASSISTANT")
        print("=" * 70)

        print(
            "Opening the existing Qdrant collection..."
        )

        retriever, chat_model, client = (
            create_rag_components()
        )

        print("RAG assistant is ready.")
        print("Type 'exit' or 'quit' to stop.")

        while True:
            question = input(
                "\nEnter your question: "
            ).strip()

            if question.lower() in {
                "exit",
                "quit",
            }:
                print("Closing the RAG assistant.")
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
        print("\nRAG assistant stopped.")

    except Exception as error:
        print(f"\nRAG error: {error}")

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()