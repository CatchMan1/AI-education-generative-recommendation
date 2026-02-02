# app/models/announcement.py
from datetime import datetime
from sqlalchemy import String, TEXT, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class Announcement(Base):
    __tablename__ = 'announcements'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(TEXT, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=True)
    publish_date: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.utcnow)

    def __repr__(self):
        return f"<Announcement(id={self.id}, title={self.title[:20]})>"
