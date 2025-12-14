import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, delete
from src.main import app
from src.database import get_session, engine, Interaction

client = TestClient(app)

@pytest.fixture(name="session")
def session_fixture():
    with Session(engine) as session:
        yield session

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

def test_search_basic():
    payload = {"query": "nutrition", "limit": 5}
    response = client.post("/api/v1/search", json=payload)
    assert response.status_code == 200
    assert len(response.json()["results"]) > 0

def test_feedback_loop(session: Session):
    session.exec(delete(Interaction))
    session.commit()
    
    query = "vitamin"
    res1 = client.post("/api/v1/search", json={"query": query})
    target_id = res1.json()["results"][2]["id"]
    
    for _ in range(5):
        client.post("/api/v1/feedback", json={"document_id": target_id, "query": query, "score_delta": 10})
        
    res2 = client.post("/api/v1/search", json={"query": query})
    updated = next(d for d in res2.json()["results"] if d["id"] == target_id)
    assert updated["feedback_score"] == 50
