from datetime import datetime
from sqlalchemy import Integer, String, Text, Column, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

class InteractionRecord(Base):
    __tablename__ = 'interaction_records'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, comment="唯一标识符")
    student_id: Mapped[str] = mapped_column(String(50), nullable=False, comment="用户ID（学号）")
    class_id: Mapped[int] = mapped_column(Integer, nullable=False, comment="学习资源ID")
    class_name: Mapped[str] = mapped_column(String(255), nullable=True, comment="学习资源名称")
    keywords_pos: Mapped[str] = mapped_column(Text, nullable=True, comment="积极关键词")
    keywords_neg: Mapped[str] = mapped_column(Text, nullable=True, comment="消极关键词")
    preference: Mapped[str] = mapped_column(String(255), nullable=True, comment="偏好标签")

    def __repr__(self):
        return (f"<InteractionRecord(id={self.id}, student_id={self.student_id}, "
                f"class_name={self.class_name}, preference={self.preference})>")