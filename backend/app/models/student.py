from datetime import datetime
from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

# class Student(Base):
#     __tablename__ = 'students'
#
#     student_id: Mapped[str] = mapped_column(String(20), primary_key=True)
#     name: Mapped[str] = mapped_column(String(100), nullable=True)
#     phone: Mapped[str | None] = mapped_column(String(11), nullable=False)
#     college: Mapped[str | None] = mapped_column(String(100), nullable=False)
#     major: Mapped[str | None] = mapped_column(String(100), nullable=False)
#     grade: Mapped[str | None] = mapped_column(String(10), nullable=True, comment="学生年级")
#     password: Mapped[str] = mapped_column(String(100), nullable=False)
#     registration_date: Mapped[str] = mapped_column(
#         String(30),
#         default=lambda: datetime.utcnow().isoformat()
#     )
#     interest_profile: Mapped[str | None] = mapped_column(
#         Text, nullable=True, comment="兴趣标签，Text类型，支持JSON或长文本"
#     )
#     interest_long_profile: Mapped[str | None] = mapped_column(
#         Text, nullable=True, comment="更长兴趣画像，可为结构化/非结构化长文本"
#     )
#
#     def __repr__(self):
#         return f"<Student(id={self.student_id}, name={self.name})>"
from datetime import datetime
from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

class Student(Base):
    __tablename__ = 'students'

    student_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(11), nullable=True)
    college: Mapped[str | None] = mapped_column(String(100), nullable=True)
    major: Mapped[str | None] = mapped_column(String(100), nullable=True)
    grade: Mapped[str | None] = mapped_column(String(10), nullable=True, comment="学生年级")
    password: Mapped[str] = mapped_column(String(100), nullable=False)
    registration_date: Mapped[str] = mapped_column(
        String(30),
        default=lambda: datetime.utcnow().isoformat()
    )
    interest_profile: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="兴趣标签，Text类型，支持JSON或长文本"
    )
    interest_long_profile: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="更长兴趣画像，可为结构化/非结构化长文本"
    )

    def __repr__(self):
        return f"<Student(id={self.student_id}, name={self.name})>"
