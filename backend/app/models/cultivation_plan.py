from datetime import datetime
from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

class CultivationPlan(Base):
    __tablename__ = 'cultivation_plan'

    id: Mapped[int] = mapped_column(Integer, primary_key=True,comment="唯一标识符")
    learning_stage: Mapped[str] = mapped_column(String(50), nullable=True, comment="学习阶段，例如本科、硕士、博士等")
    major: Mapped[str] = mapped_column(String(100), nullable=True, comment="专业名称")
    training_target: Mapped[str] = mapped_column(Text, nullable=True, comment="培养目标，可为长文本")
    major_introduction: Mapped[str] = mapped_column(Text, nullable=True, comment="专业简介，可为长文本")
    main_courses: Mapped[str] = mapped_column(Text, nullable=True, comment="主要开设课程，可为长文本")

    def __repr__(self):
        return f"<CultivationPlan(id={self.id}, major={self.major})>"