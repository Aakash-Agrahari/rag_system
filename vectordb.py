import chromadb
from config import CHROMA_PATH
from config import COLLECTION_NAME

#this is just a simple client creation for chromadb, like python->chroma client->vector db, simply for the database connection
#also we have created a persistent client so that the data will be stored in the specified path and will be available for future use and won't disappear as we close the program
client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

#this is just to verify from the terminal that connection is successful
print("Connected successfully!")

#now we will be making a collection i.e table in the database to store the data, we will be using the collection name from the config file
#this collection will be used to store the chunks of the documents and their embeddings
collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

if __name__ == "__main__":

    print("Connected to ChromaDB\n")

    print("Collection:")
    print(collection.name)

    print("\nCollections:")

    collections = client.list_collections()

    for col in collections:
        print("-", col.name)

    print("\nDocuments Stored:")
    print(collection.count())


