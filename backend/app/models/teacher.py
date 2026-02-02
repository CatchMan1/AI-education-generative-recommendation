from datetime import datetime
from sqlalchemy import String, TIMESTAMP, Text ,Column
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class Teacher(Base):
    __tablename__ = 'teachers'

    teacher_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[str] = mapped_column(String(11), nullable=True)
    college: Mapped[str] = mapped_column(String(100), nullable=True)
    major: Mapped[str] = mapped_column(String(100), nullable=True)
    password: Mapped[str] = mapped_column(String(100), nullable=False)
    registration_date: Mapped[str] = mapped_column(
        String(30),
        default=lambda: datetime.utcnow().isoformat()
    )
    interest_tags: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="兴趣标签，Text类型，支持JSON或长文本"
    )

    def __repr__(self):
        return f"<Teacher(id={self.teacher_id}, name={self.name})>"
