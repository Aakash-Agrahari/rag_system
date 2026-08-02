from retriever import retrieve
from llm import ask_gemini


def build_context(results):
    """
    Combine all retrieved chunks into one context string.
    """

    context = ""

    for index, result in enumerate(results, start=1):

        context += (
            f"[Source {index}: {result['filename']}]\n"
            f"{result['document']}\n\n"
        )

    return context


def build_prompt(question, context):
    """
    Construct the final prompt sent to Gemini.
    """

    return f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

Rules:
1. Do not make up information.
2. If the answer is not present in the context, say:
   "I couldn't find that information in the provided documents."
3. If multiple sources contain useful information, combine them naturally.
4. Be concise but complete.

======================
CONTEXT
======================

{context}

======================
QUESTION
======================

{question}

======================
ANSWER
======================
"""


def display_sources(results):
    """
    Display source files used to answer the question.
    """

    print("\nSources Used")

    unique_sources = []

    for result in results:

        filename = result["filename"]

        if filename not in unique_sources:
            unique_sources.append(filename)

    for source in unique_sources:
        print(f"- {source}")


def main():

    print("=" * 70)
    print("RAG DOCUMENT ASSISTANT")
    print("=" * 70)

    while True:

        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        try:

            print("\nSearching knowledge base.....\n")

            retrieved_chunks = retrieve(question)

            if not retrieved_chunks:

                print("No relevant documents found.")

                continue

            context = build_context(retrieved_chunks)

            prompt = build_prompt(question, context)

            answer = ask_gemini(prompt)

            print("=" * 70)

            print("ANSWER\n")

            print(answer)

            print("=" * 70)

            display_sources(retrieved_chunks)

            print("=" * 70)

        except Exception as e:

            print("\nAn error occurred:")
            print(e)


if __name__ == "__main__":
    main()