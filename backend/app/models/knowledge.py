# app/models/knowledge_base.py
from datetime import datetime
from sqlalchemy import String, TEXT, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_name: Mapped[str] = mapped_column(String(200), nullable=False)
    uploader: Mapped[str] = mapped_column(String(100), nullable=True)
    document_content: Mapped[str] = mapped_column(TEXT, nullable=False)
    upload_time: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.utcnow)

    def __repr__(self):
        return f"<KnowledgeDocument(id={self.id}, name={self.document_name[:20]})>"
