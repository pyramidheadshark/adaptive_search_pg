import os
from datetime import datetime
from typing import Optional, Generator
from sqlmodel import Field, SQLModel, create_engine, Session, text
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://app_user:app_password@localhost:5432/adaptive_search")

class User(SQLModel, table=True):
    __tablename__ = "users"
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    password_hash: str = Field(nullable=False) 
    role: str = Field(default="user")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Document(SQLModel, table=True):
    __tablename__ = "documents"
    id: Optional[int] = Field(default=None, primary_key=True)
    content: str
    category: Optional[str] = None
    embedding: list[float] = Field(sa_column=Column(Vector(384)))
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Interaction(SQLModel, table=True):
    __tablename__ = "interactions"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id")
    document_id: int = Field(foreign_key="documents.id")
    query_text: str
    score_delta: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    with Session(engine) as session:
        session.exec(text("CREATE EXTENSION IF NOT EXISTS vector"))
        session.commit()
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
