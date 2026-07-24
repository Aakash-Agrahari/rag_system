import os
from pathlib import Path
from pypdf import PdfReader

from config import DOCUMENTS_PATH

#this function read the content of a text file and return it as a string
def load_txt(file_path): 
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

#this function read the content of a pdf file and return it as a string
def load_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


#this function loads all the documents from the specified folder and returns a list of dictionaries containing the filename and content of each document
#basically this function will provide all the content of the documents in the specified folder to the RAG system for further processing i.e. chunking, embedding, and store in the vector db
def load_documents():
    documents = []

    document_folder = Path(DOCUMENTS_PATH)

    for file in document_folder.iterdir():

        if file.suffix == ".txt":
            text = load_txt(file)

        elif file.suffix == ".pdf":
            text = load_pdf(file)

        else:
            continue

        documents.append({
            "filename": file.name,
            "content": text
        })

    return documents


if __name__ == "__main__":
    docs = load_documents()

    print(f"Loaded {len(docs)} documents\n")

    for doc in docs:
        print("=" * 50)
        print(f"File: {doc['filename']}")
        print(doc["content"][:300])  # Preview first 300 characters
        print()