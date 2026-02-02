"""
通用响应模型
"""
from typing import Any, Optional, Generic, TypeVar
from pydantic import BaseModel

DataType = TypeVar('DataType')


class ResponseModel(BaseModel, Generic[DataType]):
    """通用响应模型"""
    success: bool = True
    message: str = "操作成功"
    data: Optional[DataType] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "操作成功",
                "data": None
            }
        }


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "healthy"
    service: str = "AI教育助手后端"
    version: str = "1.0.0"
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "AI教育助手后端",
                "version": "1.0.0"
            }
        }
