from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DOCUMENTS_PATH = "documents"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_documents"