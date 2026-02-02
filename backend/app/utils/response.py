"""
响应工具函数
"""
from typing import Any, Optional
from fastapi import HTTPException
from app.schemas.common import ResponseModel


def create_response(
    data: Any = None,
    message: str = "操作成功",
    success: bool = True
) -> ResponseModel:
    """创建标准响应"""
    return ResponseModel(
        success=success,
        message=message,
        data=data
    )


def success_response(
    data: Any = None,
    message: str = "操作成功"
) -> ResponseModel:
    """创建成功响应"""
    return ResponseModel(
        success=True,
        message=message,
        data=data
    )


def error_response(
    message: str = "操作失败",
    data: Any = None
) -> ResponseModel:
    """创建错误响应（不抛出异常）"""
    return ResponseModel(
        success=False,
        message=message,
        data=data
    )


def create_error_response(
    message: str = "操作失败",
    status_code: int = 400,
    data: Any = None
) -> HTTPException:
    """创建错误响应（抛出异常）"""
    return HTTPException(
        status_code=status_code,
        detail={
            "success": False,
            "message": message,
            "data": data
        }
    )
