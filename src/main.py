from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="Adaptive Search")
app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health():
    return {"status": "ok"}
