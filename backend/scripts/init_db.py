# scripts/init_db.py
import asyncio
import sys
import os

# 将项目根目录（backend）添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 切换工作目录到 backend 目录，确保数据库在正确位置创建
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(backend_dir)
print(f"✅ 工作目录已切换到: {backend_dir}")

from app.core.config import settings
from sqlalchemy.ext.asyncio import create_async_engine
from app.models.student import Student
from app.models.teacher import Teacher
from app.models.announcement import Announcement
from app.models.knowledge import KnowledgeBase
from app.models.course1 import Course
from app.models.chat1 import Conversation
from app.models.chat1 import Message
from app.models.corrective_record import CorrectiveRecord
from app.models.profile import AdminProfile
from app.models.interest_tag import InterestTag
from app.models.cultivation_plan import CultivationPlan
from app.models.class_index import ClassIndex
from app.models.interaction_records import InteractionRecord
from app.db.base import Base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.security import hash_password
# import bcrypt
from datetime import datetime
import pandas as pd

import os


# 创建数据库引擎
print(f"📊 数据库URL: {settings.DATABASE_URL}")
print(f"📁 当前工作目录: {os.getcwd()}")
engine = create_async_engine(settings.DATABASE_URL, echo=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


# 初始化数据库结构
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# 创建默认管理员账号（如不存在）
async def create_default_admin():
    async with async_session() as session:
        result = await session.execute(
            # 检查是否已存在
            AdminProfile.__table__.select().where(AdminProfile.admin_id == "admin001")
        )
        if result.first():
            print("✅ 管理员 admin001 已存在，跳过创建")
            return

        # 例如初始化时
        admin = AdminProfile(
            admin_id="admin001",
            name="管理员",
            phone="13800000000",
            password="123456"  # 使用 hash_password
        )
        session.add(admin)
        await session.commit()
        print("✅ 管理员 admin001 创建成功")


# 插入学生数据
async def insert_students():
    async with async_session() as session:
        # 检查是否已有学生数据
        result = await session.execute(Student.__table__.select().limit(1))
        if result.first():
            print("✅ 学生数据已存在，跳过插入")
            return

        # 示例学生数据
        student_data = [
            {
                "student_id": "S001",
                "name": "张三",
                "phone": "13900000000",
                "college": "计算机学院",
                "major": "软件工程",
                "password": "password123"
            },
            {
                "student_id": "S002",
                "name": "李四",
                "phone": "13900000001",
                "college": "电子信息学院",
                "major": "通信工程",
                "password": "password456"
            }
        ]

        for data in student_data:
            # 临时使用明文密码，避免哈希函数的版本冲突问题
            # data["password"] = hash_password(data["password"])
            # 暂时直接使用明文密码，后续可以手动更新
            pass

            student = Student(**data)
            session.add(student)

        await session.commit()
        print("✅ 学生数据插入成功")


# 获取当前脚本的目录
current_dir = os.path.dirname(__file__)

async def insert_cultivation_plans_from_excel():
    async with async_session() as session:
        # 检查是否已有培养计划数据
        result = await session.execute(CultivationPlan.__table__.select().limit(1))
        if result.first():
            print("✅ 培养计划数据已存在，跳过插入")
            return
        # 读取 Excel 文件
        df = pd.read_excel(os.path.join(current_dir, 'cultivation_plan.xlsx'))

        # 将 DataFrame 转换为字典列表
        cultivation_plans_data = df.to_dict(orient='records')

        # 使用 executemany 进行批量插入
        await session.execute(
            CultivationPlan.__table__.insert(),
            cultivation_plans_data
        )

        await session.commit()
        print("✅ 培养计划数据从 Excel 文件批量插入成功")

async def insert_class_index_from_excel():
    async with async_session() as session:
        # 检查是否已有学习资源数据
        result = await session.execute(ClassIndex.__table__.select().limit(1))
        if result.first():
            print("✅ 学习资源数据已存在，跳过插入")
            return
        # 读取 Excel 文件
        df = pd.read_excel(os.path.join(current_dir, 'class_index.xlsx'))


        # 将 DataFrame 转换为字典列表
        class_index_data = df.to_dict(orient='records')

        # 使用 executemany 进行批量插入
        await session.execute(
            ClassIndex.__table__.insert(),
            class_index_data
        )

        await session.commit()
        print("✅ 学习资源数据从 Excel 文件批量插入成功")


async def insert_interaction_records_from_csv():
    async with async_session() as session:
        # 检查是否已有互动记录数据
        result = await session.execute(InteractionRecord.__table__.select().limit(1))
        if result.first():
            print("✅ 互动记录数据已存在，跳过插入")
            return
        csv_path = os.path.join(current_dir, 'interaction_records.csv')
        df = pd.read_csv(csv_path)


        # 将DataFrame转换为字典列表（键名需与数据库模型字段一致）
        interaction_data = df.to_dict(orient='records')

        # 使用executemany批量插入
        await session.execute(
            InteractionRecord.__table__.insert(),
            interaction_data
        )

        await session.commit()
        print("✅ 互动记录数据从CSV文件批量插入成功")



async def insert_student_records_from_excel():
    async with async_session() as session:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Excel 文件路径
        excel_path = os.path.join(current_dir, 'student_model.xlsx')

        # 读取 Excel 文件
        df = pd.read_excel(excel_path)

        df['registration_date'] = df['registration_date'].fillna(datetime.utcnow().isoformat())

        # 将 DataFrame 转换为字典列表（键名需与数据库模型字段一致）
        student_data = df.to_dict(orient='records')

        # 使用 executemany 批量插入
        await session.execute(
            Student.__table__.insert(),
            student_data
        )

        await session.commit()
        print("✅ 学生数据从Excel文件批量插入成功")



# 主执行函数
async def init():
    await init_db()
    await create_default_admin()
    await insert_cultivation_plans_from_excel()
    await insert_class_index_from_excel()
    await insert_interaction_records_from_csv()
    # await insert_students()
    await insert_student_records_from_excel()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(init())