"""
作业批改API路由
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from app.schemas.homework import (
    HomeworkGradingRequest,
    HomeworkGradingResponse,
    HomeworkSubmissionRequest,
    HomeworkSubmissionResponse
)
from app.utils.response import success_response
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/homework-grading", tags=["作业批改"])


@router.post("/submit", response_model=HomeworkSubmissionResponse, summary="提交作业")
async def submit_homework(request: HomeworkSubmissionRequest):
    """提交作业进行批改"""
    try:
        # 模拟作业提交逻辑
        result = {
            "id": 1,
            "student_id": request.student_id,
            "homework_type": request.homework_type,
            "title": request.title,
            "submitted_at": "2024-01-15T10:30:00",
            "is_graded": False
        }
        return success_response(data=result, message="作业提交成功")
    except Exception as e:
        logger.error(f"作业提交失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/grade", response_model=HomeworkGradingResponse, summary="批改作业")
async def grade_homework(request: HomeworkGradingRequest):
    """AI智能批改作业"""
    try:
        # 模拟批改结果
        result = {
            "homework_id": 1,
            "total_score": "85",
            "grade": "B+",
            "rank": "良好",
            "percentile": "78%",
            "details": [
                {"category": "内容准确性", "score": 34, "total": 40, "comment": "内容丰富，观点明确"},
                {"category": "结构逻辑性", "score": 25, "total": 30, "comment": "结构清晰，逻辑合理"},
                {"category": "语言表达", "score": 16, "total": 20, "comment": "表达流畅，用词准确"},
                {"category": "创新性", "score": 8, "total": 10, "comment": "有一定创新思路"}
            ],
            "suggestions": [
                "建议在第二段增加更多具体例证",
                "结论部分可以更加简洁明了",
                "注意检查标点符号的使用"
            ],
            "graded_at": "2024-01-15T10:35:00"
        }
        return success_response(data=result, message="作业批改完成")
    except Exception as e:
        logger.error(f"作业批改失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/homework/{homework_id}", summary="获取作业详情")
async def get_homework_detail(homework_id: int):
    """获取作业详情和批改结果"""
    try:
        # 模拟作业详情
        result = {
            "id": homework_id,
            "title": "我的理想",
            "content": "我的理想是成为一名教师...",
            "homework_type": "essay",
            "submitted_at": "2024-01-15T10:30:00",
            "is_graded": True,
            "grading_result": {
                "total_score": "85",
                "grade": "B+",
                "details": [],
                "suggestions": []
            }
        }
        return success_response(data=result, message="获取作业详情成功")
    except Exception as e:
        logger.error(f"获取作业详情失败: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/list", summary="获取作业列表")
async def get_homework_list(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    homework_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None)
):
    """获取作业列表"""
    try:
        # 模拟作业列表
        homeworks = [
            {
                "id": 1,
                "title": "我的理想",
                "homework_type": "essay",
                "submitted_at": "2024-01-15T10:30:00",
                "is_graded": True,
                "score": "85"
            },
            {
                "id": 2,
                "title": "数学练习题",
                "homework_type": "math",
                "submitted_at": "2024-01-14T15:20:00",
                "is_graded": False,
                "score": None
            }
        ]
        
        result = {
            "homeworks": homeworks,
            "total": len(homeworks),
            "page": page,
            "size": size
        }
        return success_response(data=result, message="获取作业列表成功")
    except Exception as e:
        logger.error(f"获取作业列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
