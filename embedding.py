from google import genai
from config import GEMINI_API_KEY
from chunker import create_chunks

client = genai.Client(api_key=GEMINI_API_KEY)

#this function will generate the embedding for the given text using the gemini embedding model and then return the embedding vector as a list of floats
def generate_embedding(text: str):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


#this function will create the chunks of the documents and then generate the embedding for each chunk and return a list of dictionaries containing the chunk id, filename, chunk content, and embedding vector
#basically this function stores the metadata of the chunks like where it come from, what was the original text, and what embedding belongs to it
def embed_chunks():
    chunks = create_chunks()

    embedded_chunks = []

    for chunk in chunks:

        embedding = generate_embedding(chunk["chunk"])

        embedded_chunks.append({
            "id": chunk["id"],
            "filename": chunk["filename"],
            "chunk": chunk["chunk"],
            "embedding": embedding
        })

    return embedded_chunks




if __name__ == "__main__":

    embedded = embed_chunks()

    if not embedded:
        print("No chunks found.")
        exit()

    print(f"Generated embeddings for {len(embedded)} chunks\n")

    first = embedded[0]

    print("Chunk ID:", first["id"])
    print("Filename:", first["filename"])
    print("Chunk Preview:")
    print(first["chunk"][:200])

    print("\nEmbedding Dimension:", len(first["embedding"]))

    print("\nFirst 10 Values:")
    print(first["embedding"][:10])