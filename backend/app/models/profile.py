from datetime import datetime
from sqlalchemy import String, Column
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

class AdminProfile(Base):
    __tablename__ = 'admin_profiles'  # 数据库表名

    # 管理员ID（主键，不可修改，与用户系统关联）
    admin_id: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
        comment="管理员唯一标识ID"
    )

    # 管理员姓名（可编辑）
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="管理员姓名"
    )

    # 联系电话（可编辑，允许为空）
    phone: Mapped[str | None] = mapped_column(
        String(11),  # 手机号通常为11位
        nullable=True,
        comment="管理员联系电话"
    )

    # 密码（可修改，不可为空）
    password: Mapped[str] = mapped_column(
        String(100),  # 存储加密后的密码
        nullable=False,
        comment="加密存储的登录密码"
    )

    # 账号创建时间（不可编辑，自动记录）
    create_time: Mapped[str] = mapped_column(
        String(30),
        default=lambda: datetime.utcnow().isoformat(),  # 自动生成ISO格式时间
        comment="账号创建时间（UTC）"
    )

    # 最后更新时间（自动更新）
    last_update_time: Mapped[str] = mapped_column(
        String(30),
        default=lambda: datetime.utcnow().isoformat(),
        onupdate=lambda: datetime.utcnow().isoformat(),  # 更新时自动刷新
        comment="最后信息更新时间（UTC）"
    )

    def __repr__(self):
        # 模型的字符串表示，便于调试
        return f"<AdminProfile(admin_id={self.admin_id}, name={self.name})>"