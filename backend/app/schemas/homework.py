"""
作业批改相关Schema
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class HomeworkGradingRequest(BaseModel):
    """作业批改请求Schema"""
    homework_type: str = Field(
        ..., 
        pattern="^(essay|math|science|other)$",
        description="作业类型"
    )
    title: str = Field(..., min_length=1, max_length=200, description="作业标题")
    content: str = Field(..., min_length=10, max_length=50000, description="作业内容")
    grading_criteria: List[str] = Field(
        ..., 
        min_items=1,
        description="评分标准列表"
    )
    total_score: int = Field(100, ge=1, le=1000, description="总分设置")
    feedback_level: str = Field(
        "detailed",
        pattern="^(basic|detailed|comprehensive)$",
        description="反馈详细程度"
    )
    include_comments: bool = Field(True, description="是否生成改进建议")
    
    class Config:
        json_schema_extra = {
            "example": {
                "homework_type": "essay",
                "title": "我的理想",
                "content": "我的理想是成为一名教师...",
                "grading_criteria": ["content", "structure", "language"],
                "total_score": 100,
                "feedback_level": "detailed",
                "include_comments": True
            }
        }


class GradingDetailResponse(BaseModel):
    """评分详情响应Schema"""
    category: str = Field(..., description="评分类别")
    score: float = Field(..., description="得分")
    total: float = Field(..., description="总分")
    comment: Optional[str] = Field(None, description="评语")


class HomeworkGradingResponse(BaseModel):
    """作业批改响应Schema"""
    homework_id: int = Field(..., description="作业ID")
    total_score: str = Field(..., description="总分")
    grade: str = Field(..., description="等级")
    rank: str = Field(..., description="排名")
    percentile: str = Field(..., description="百分位")
    details: List[GradingDetailResponse] = Field(..., description="详细评分")
    suggestions: List[str] = Field(..., description="改进建议")
    graded_at: datetime = Field(..., description="批改时间")
    
    class Config:
        json_schema_extra = {
            "example": {
                "homework_id": 1,
                "total_score": "85",
                "grade": "B+",
                "rank": "良好",
                "percentile": "78%",
                "details": [
                    {
                        "category": "内容准确性",
                        "score": 34,
                        "total": 40,
                        "comment": "内容丰富，观点明确"
                    }
                ],
                "suggestions": [
                    "建议在第二段增加更多具体例证",
                    "结论部分可以更加简洁明了"
                ],
                "graded_at": "2024-01-15T10:30:00"
            }
        }


class HomeworkSubmissionRequest(BaseModel):
    """作业提交请求Schema"""
    student_id: int = Field(..., description="学生ID")
    homework_type: str = Field(
        ..., 
        pattern="^(essay|math|science|other)$",
        description="作业类型"
    )
    title: str = Field(..., min_length=1, max_length=200, description="作业标题")
    content: str = Field(..., min_length=10, max_length=50000, description="作业内容")
    
    class Config:
        json_schema_extra = {
            "example": {
                "student_id": 1001,
                "homework_type": "essay",
                "title": "我的理想",
                "content": "我的理想是成为一名教师..."
            }
        }


class HomeworkSubmissionResponse(BaseModel):
    """作业提交响应Schema"""
    id: int = Field(..., description="作业ID")
    student_id: int = Field(..., description="学生ID")
    homework_type: str = Field(..., description="作业类型")
    title: str = Field(..., description="作业标题")
    submitted_at: datetime = Field(..., description="提交时间")
    is_graded: bool = Field(..., description="是否已批改")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "student_id": 1001,
                "homework_type": "essay",
                "title": "我的理想",
                "submitted_at": "2024-01-15T09:00:00",
                "is_graded": False
            }
        }


class HomeworkListResponse(BaseModel):
    """作业列表响应Schema"""
    homeworks: List[HomeworkSubmissionResponse] = Field(..., description="作业列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页码")
    size: int = Field(..., description="每页数量")


class HomeworkTypeStats(BaseModel):
    """作业类型统计Schema"""
    homework_type: str = Field(..., description="作业类型")
    total_count: int = Field(..., description="总数量")
    graded_count: int = Field(..., description="已批改数量")
    average_score: Optional[float] = Field(None, description="平均分")


class HomeworkStatsResponse(BaseModel):
    """作业统计响应Schema"""
    total_submissions: int = Field(..., description="总提交数")
    graded_submissions: int = Field(..., description="已批改数")
    pending_submissions: int = Field(..., description="待批改数")
    type_stats: List[HomeworkTypeStats] = Field(..., description="类型统计")
