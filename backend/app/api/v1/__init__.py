from fastapi import APIRouter
from .chat import router as chat_router
from .text_organization import router as text_organization_router
from .ppt_creation import router as ppt_creation_router
from .homework_grading import router as homework_grading_router
from .learning_path import router as learning_path_router
from .lesson_plan import router as lesson_plan_router

# 创建API v1路由
api_router = APIRouter()

# 注册子路由
api_router.include_router(chat_router, prefix="/chat", tags=["AI问答"])
api_router.include_router(text_organization_router, tags=["AI效率-文本梳理"])
api_router.include_router(ppt_creation_router, tags=["AI效率-PPT制作"])
api_router.include_router(homework_grading_router, tags=["AI效率-作业批改"])
api_router.include_router(learning_path_router, tags=["AI效率-学习路径规划"])
api_router.include_router(lesson_plan_router, tags=["AI效率-教案生成"])
