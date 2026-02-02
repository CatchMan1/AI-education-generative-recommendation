from sqlalchemy import String, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

class InterestTag(Base):
    __tablename__ = "interest_tag"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, index=True)
    tag: Mapped[str] = mapped_column(String(255), nullable=False)
