import chromadb

from config import CHROMA_PATH, COLLECTION_NAME
from embedding import generate_embedding


# Connect to ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Load the existing collection
collection = client.get_collection(
    name=COLLECTION_NAME
)


def retrieve(query: str, top_k: int = 3):
    """
    Retrieve the most relevant document chunks for a user query.
    """

    # Convert the user's question into an embedding
    query_embedding = generate_embedding(query)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        retrieved_chunks.append(
            {
                "document": document,
                "filename": metadata["filename"],
                "distance": distance
            }
        )

    return retrieved_chunks


if __name__ == "__main__":

    print("=" * 60)
    print("RAG RETRIEVAL")
    print("=" * 60)

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        results = retrieve(query)

        print("\nRetrieved Chunks")

        for index, result in enumerate(results, start=1):

            print("\n" + "=" * 60)

            print(f"Result {index}")

            print(f"Source File : {result['filename']}")

            print(f"Distance    : {result['distance']:.4f}")

            print("\nChunk:\n")

            print(result["document"])

        print("\n" + "=" * 60)