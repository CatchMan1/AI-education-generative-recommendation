"""
AI教育助手后端应用入口
"""
from app.main import app

# 导出app实例供uvicorn使用
__all__ = ["app"]