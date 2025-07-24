from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL, TOP_K
import os

class RetrieverAgent:
    """
    Agent that manages document retrieval using FAISS and HuggingFace embeddings.
    Loads or creates a FAISS index and retrieves the most relevant documents for a query.
    """
    def __init__(self, vectorstore_path: str = "vectorstore"):
        # Uses updated HuggingFaceEmbeddings
        self.embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore_path = vectorstore_path
        self.vectorstore = None

        # Try to load an existing index with protection
        if os.path.exists(self.vectorstore_path):
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embedding,
                allow_dangerous_deserialization=True 
            )

    def index_documents(self, documents: list[str]):
        """
        Creates a new FAISS index from the documents and saves it locally.
        """
        self.vectorstore = FAISS.from_texts(documents, self.embedding)
        self.vectorstore.save_local(self.vectorstore_path)

    def retrieve(self, query: str):
        """
        Retrieves the most similar documents to the query.
        """
        if not self.vectorstore:
            raise ValueError("No indexed documents. Call index_documents() first.")
        return self.vectorstore.similarity_search(query, k=TOP_K)

    def __call__(self, state):
        """
        Integrates with LangGraph by retrieving relevant documents for the query and updating the state.
        """
        query = state["query"]
        docs = self.retrieve(query)
        state["docs"] = docs
        return state
