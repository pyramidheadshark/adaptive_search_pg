from fastapi import FastAPI
from src.core.config import settings

app = FastAPI(title="Adaptive Search")

@app.get("/")
def root():
    return {"msg": "Welcome to Adaptive Search API"}

@app.get("/health")
def health():
    return {"status": "ok", "db": settings.DATABASE_URL}
