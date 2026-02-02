from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

class ClassIndex(Base):
    __tablename__ = 'class_index'

    class_id: Mapped[int] = mapped_column(Integer, primary_key=True, comment="主码")
    class_name: Mapped[str] = mapped_column(String(100), nullable=True, comment="学习资源名称")
    content: Mapped[str] = mapped_column(Text, nullable=True, comment="学习资源完整内容")
    keywords_pos: Mapped[str] = mapped_column(Text, nullable=True, comment="积极关键词")
    keywords_neg: Mapped[str] = mapped_column(Text, nullable=True, comment="消极关键词")
    url: Mapped[str] = mapped_column(String(255), nullable=True, comment="学习资源的在线链接")

    def __repr__(self):
        return f"<ClassIndex(class_id={self.class_id}, class_name={self.class_name})>"