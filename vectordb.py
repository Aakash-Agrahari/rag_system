import chromadb
from config import CHROMA_PATH, COLLECTION_NAME
from embedding import embed_chunks

#this is just a simple client creation for chromadb, like python->chroma client->vector db, simply for the database connection
#also we have created a persistent client so that the data will be stored in the specified path and will be available for future use and won't disappear as we close the program
client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

#now we will be making a collection i.e table in the database to store the data, we will be using the collection name from the config file
#this collection will be used to store the chunks of the documents and their embeddings
collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

def store_embeddings():
    """
    Generate embeddings for all chunks and store them in ChromaDB.
    Uses upsert() so the script can be safely re-run.
    """

    chunks = embed_chunks()

    print(f"\nFound {len(chunks)} chunks.\n")

    for chunk in chunks:

        collection.upsert(
            ids=[chunk["id"]],
            embeddings=[chunk["embedding"]],
            documents=[chunk["chunk"]],
            metadatas=[
                {
                    "filename": chunk["filename"]
                }
            ]
        )

    print(f"Successfully stored {len(chunks)} chunks in ChromaDB.\n")


def display_collection_info():
    """
    Display information about the collection.
    """

    print("Collection Name:")
    print(collection.name)

    print("\nTotal Stored Documents:")
    print(collection.count())


def display_sample_document():
    """
    Display one stored document to verify insertion.
    """

    if collection.count() == 0:
        print("\nCollection is empty.")
        return

    result = collection.get(limit=1)

    print("\nSample Stored Document")
    print("-" * 50)

    print("ID:")
    print(result["ids"][0])

    print("\nFilename:")
    print(result["metadatas"][0]["filename"])

    print("\nDocument Preview:")
    print(result["documents"][0][:300])


if __name__ == "__main__":

    print("=" * 60)
    print("CHROMADB INGESTION")
    print("=" * 60)

    store_embeddings()

    print("\n" + "=" * 60)

    display_collection_info()

    print("\n" + "=" * 60)

    display_sample_document()

    print("\n" + "=" * 60)