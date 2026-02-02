"""
作业批改相关数据模型
"""
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime


class HomeworkSubmissionBase(BaseModel):
    """作业提交基础模型"""
    student_id: int
    homework_type: str  # 'essay', 'math', 'science', 'other'
    title: str
    content: str
    file_path: Optional[str] = None


class HomeworkSubmission(HomeworkSubmissionBase):
    """作业提交模型"""
    id: int
    submitted_at: datetime
    
    # 批改结果
    is_graded: bool = False
    total_score: Optional[float] = None
    grade: Optional[str] = None
    rank: Optional[str] = None
    percentile: Optional[str] = None
    feedback: Optional[str] = None
    graded_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class GradingCriteriaBase(BaseModel):
    """评分标准基础模型"""
    category: str
    weight: float  # 权重百分比
    description: str


class GradingCriteria(GradingCriteriaBase):
    """评分标准模型"""
    id: int
    homework_type: str
    
    class Config:
        from_attributes = True


class GradingDetailBase(BaseModel):
    """评分详情基础模型"""
    category: str
    score: float
    total: float
    comment: Optional[str] = None


class GradingDetail(GradingDetailBase):
    """评分详情模型"""
    id: int
    homework_id: int
    
    class Config:
        from_attributes = True


class HomeworkGradingResultBase(BaseModel):
    """批改结果基础模型"""
    homework_id: int
    total_score: float
    grade: str
    rank: str
    percentile: str
    overall_feedback: str


class HomeworkGradingResult(HomeworkGradingResultBase):
    """批改结果模型"""
    id: int
    graded_at: datetime
    details: List[GradingDetail] = []
    suggestions: List[str] = []
    
    class Config:
        from_attributes = True


class HomeworkCreate(HomeworkSubmissionBase):
    """创建作业模型"""
    pass


class HomeworkUpdate(BaseModel):
    """更新作业模型"""
    title: Optional[str] = None
    content: Optional[str] = None
    homework_type: Optional[str] = None
