"""
AI教育助手后端主应用
"""
# 在现有导入基础上添加
from sqlalchemy import text  # 添加这一行
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.core.config import settings
from app.api.v1 import (
    api_router,
    chat,
    text_organization,
    ppt_creation,
    lesson_plan,
    learning_path,
    homework_grading,
    file_upload,
)
from app.schemas.common import HealthResponse
from app.utils.logger import get_logger
import os

# 创建日志记录器
logger = get_logger(__name__)

# 创建FastAPI应用实例
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(api_router, prefix="/api/v1")

# 兼容旧版API路径
app.include_router(api_router, prefix="/api")

app.include_router(learning_path.router, prefix="/api/v1/learning_path", tags=["learning_path"])
app.include_router(homework_grading.router, prefix="/api/v1/homework_grading", tags=["homework_grading"])
app.include_router(file_upload.router, prefix="/api/v1/files", tags=["files"])


@app.get("/", tags=["系统"], summary="服务状态")
async def root():
    """
    获取服务基本信息
    """
    logger.info("访问根路径")
    return {
        "message": f"{settings.APP_NAME}运行中",
        "status": "healthy",
        "version": settings.APP_VERSION,
        "frontend_url": "http://localhost:3000",
        "docs_url": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["系统"], summary="健康检查")
async def health_check():
    """
    健康检查接口
    """
    logger.info("健康检查")
    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION
    )


# 静态文件服务（生产环境）
frontend_dist_path = "../frontend/dist"
if os.path.exists(frontend_dist_path):
    app.mount("/static", StaticFiles(directory=frontend_dist_path), name="static")
    
    @app.get("/app", tags=["前端"], summary="前端应用")
    async def serve_frontend():
        """
        提供前端应用服务
        """
        return FileResponse(f"{frontend_dist_path}/index.html")


# 应用启动事件
@app.on_event("startup")
async def startup_event():
    """
    应用启动时执行
    """
    logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} 启动成功")
    logger.info(f"文档地址: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"API地址: http://{settings.HOST}:{settings.PORT}/api/v1")

# 应用关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """
    应用关闭时执行
    """
    logger.info(f"{settings.APP_NAME} 正在关闭...")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
