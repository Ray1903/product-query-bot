import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL

class Indexer:
    """
    Utility class for creating and loading FAISS vector stores using HuggingFace embeddings.
    """
    def __init__(self, persist_path: str = "vectorstore"):
        self.embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.persist_path = persist_path

    def index_documents(self, documents: list[str]):
        """
        Creates a FAISS vector store from a list of texts and saves it locally.
        """
        vectorstore = FAISS.from_texts(documents, self.embedding)
        vectorstore.save_local(self.persist_path)
        return self.persist_path

    def load_index(self):
        """
        Loads an existing FAISS vector store from disk.
        """
        if not os.path.exists(self.persist_path):
            raise FileNotFoundError(f"No FAISS index found at {self.persist_path}")
        return FAISS.load_local(self.persist_path, self.embedding)
