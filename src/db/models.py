from typing import Optional
from sqlmodel import SQLModel, Field
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from datetime import datetime

class Document(SQLModel, table=True):
    __tablename__ = "documents"
    id: Optional[int] = Field(default=None, primary_key=True)
    content: str
    embedding: list[float] = Field(sa_column=Column(Vector(384)))
    created_at: datetime = Field(default_factory=datetime.utcnow)
