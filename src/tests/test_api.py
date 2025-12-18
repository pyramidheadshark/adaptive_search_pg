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
    assert response.json() == {"status": "ok"}

def test_search_basic():
    payload = {"query": "nutrition", "limit": 5}
    response = client.post("/api/v1/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) > 0
    assert "execution_time_ms" in data

def test_feedback_loop_scenario(session: Session):
    session.exec(delete(Interaction))
    session.commit()

    query = "vitamin"
    
    res1 = client.post("/api/v1/search", json={"query": query, "limit": 10})
    data1 = res1.json()["results"]
    
    target_doc = data1[2]
    target_id = target_doc["id"]
    initial_score = target_doc["score"]

    for _ in range(5):
        client.post("/api/v1/feedback", json={
            "document_id": target_id,
            "query": query,
            "score_delta": 10
        })

    res2 = client.post("/api/v1/search", json={"query": query, "limit": 10})
    data2 = res2.json()["results"]
    
    updated_doc = next((d for d in data2 if d["id"] == target_id), None)
    
    assert updated_doc is not None
    
    print(f"\n[Test Info] Doc ID: {target_id}")
    print(f"Initial Score: {initial_score} | New Score: {updated_doc['score']}")
    
    assert updated_doc["score"] > initial_score
    assert updated_doc["feedback_score"] == 50
