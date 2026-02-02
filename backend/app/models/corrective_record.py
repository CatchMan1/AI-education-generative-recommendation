from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class CorrectiveRecord(Base):
    __tablename__ = 'corrective_records'

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    document: Mapped[str] = mapped_column(Text, nullable=False)
    mark_records: Mapped[str] = mapped_column(Text, nullable=True)

    def __repr__(self):
        return f"<CorrectiveRecord(id={self.id}, document={self.document[:20]})>"