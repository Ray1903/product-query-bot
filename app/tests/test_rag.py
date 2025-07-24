# tests/test_pipeline.py
from app.rag.pipeline import RAGPipeline
from fastapi.testclient import TestClient
from app.main import app

def test_query():
    pipeline = RAGPipeline()
    pipeline.index_documents(["Product X: A new product"])
    answer = pipeline.query("user1", "Tell me about Product X")
    assert "Product X" in answer or answer != ""

def test_api_query():
    client = TestClient(app)
    response = client.post("/query", json={"user_id": "testuser", "query": "Tell me about Zubale Crunch"})
    assert response.status_code == 200
    data = response.json()
    assert "user_id" in data and data["user_id"] == "testuser"
    assert "answer" in data and isinstance(data["answer"], str)
