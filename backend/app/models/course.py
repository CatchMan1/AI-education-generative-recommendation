"""
课程数据模型
"""
from typing import Optional
from pydantic import BaseModel


class CourseBase(BaseModel):
    """课程基础模型"""
    name: str
    teacher: str
    description: Optional[str] = None
    category: Optional[str] = None


class Course(CourseBase):
    """课程模型"""
    id: int
    
    class Config:
        from_attributes = True


class CourseCreate(CourseBase):
    """创建课程模型"""
    pass


class CourseUpdate(BaseModel):
    """更新课程模型"""
    name: Optional[str] = None
    teacher: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
