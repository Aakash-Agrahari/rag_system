from loader import load_documents

CHUNK_SIZE=500 #makes chunk of 500 characters
OVERLAP = 100 #It wll work as senA to senB then senB to senC then senC to senD; so that all the concepts will be covered in the chunking process, help to avoid any missing information from the documents

#this function chunks the given document like character 0 to 500, then 400 to 900, then 800 to 1300.... there will be shared 100 characters between each chunks
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


#basically this function will save the metadata of the chunks so that we can where it came from i.e document name, which chunk is it, content of the chunk
#good for production level code as it will help to track the source of the chunk
def create_chunks():
    documents = load_documents()

    all_chunks = []

    for doc in documents:

        chunks = chunk_text(doc["content"])

        for index, chunk in enumerate(chunks):

            all_chunks.append({
                "id": f"{doc['filename']}_{index}",
                "filename": doc["filename"],
                "chunk": chunk
            })

    return all_chunks


if __name__ == "__main__":
    chunks = create_chunks()

    print(f"Created {len(chunks)} chunks\n")

    for chunk in chunks:
        print("=" * 60)
        print(chunk["id"])
        print(chunk["chunk"])
        print()