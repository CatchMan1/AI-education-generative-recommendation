"""
学习路径规划API路由
"""
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from app.utils.response import success_response
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/learning-path", tags=["学习路径规划"])


class LearningPathRequest(BaseModel):
    """学习路径规划请求"""
    subject: str = Field(..., description="学习科目")
    grade: str = Field(..., description="年级")
    learning_goal: str = Field(..., description="学习目标")
    duration: str = Field(..., description="计划时长")
    daily_time: str = Field(..., description="每日学习时间")
    intensity: str = Field(..., description="学习强度")
    current_level: str = Field(..., description="当前水平")
    learning_style: List[str] = Field(..., description="学习方式偏好")
    content_preference: List[str] = Field(..., description="内容偏好")


class LearningPathResponse(BaseModel):
    """学习路径规划响应"""
    path_id: int = Field(..., description="路径ID")
    subject: str = Field(..., description="学习科目")
    total_days: int = Field(..., description="总学习天数")
    total_hours: int = Field(..., description="总学时")
    milestones: int = Field(..., description="里程碑数量")
    phases: List[dict] = Field(..., description="学习阶段")
    created_at: str = Field(..., description="创建时间")


@router.post("/generate", response_model=LearningPathResponse, summary="生成学习路径")
async def generate_learning_path(request: LearningPathRequest):
    """AI智能生成个性化学习路径"""
    try:
        # 模拟学习路径生成
        duration_map = {
            "1week": 7,
            "2weeks": 14,
            "1month": 30,
            "3months": 90,
            "6months": 180
        }
        
        time_map = {
            "30min": 0.5,
            "1hour": 1,
            "1.5hours": 1.5,
            "2hours": 2,
            "3hours": 3
        }
        
        total_days = duration_map.get(request.duration, 30)
        daily_hours = time_map.get(request.daily_time, 1)
        total_hours = int(total_days * daily_hours)
        
        # 生成学习阶段
        phases = [
            {
                "title": "基础概念掌握",
                "duration": f"第1-{total_days//4}天",
                "description": "学习基本概念和原理，建立知识框架",
                "topics": ["基本定义", "核心概念", "基础公式"]
            },
            {
                "title": "技能训练",
                "duration": f"第{total_days//4+1}-{total_days//2}天",
                "description": "通过练习巩固基础知识，提高解题能力",
                "topics": ["基础练习", "技巧训练", "错题分析"]
            },
            {
                "title": "综合应用",
                "duration": f"第{total_days//2+1}-{total_days*3//4}天",
                "description": "学习综合应用，解决复杂问题",
                "topics": ["综合题目", "实际应用", "案例分析"]
            },
            {
                "title": "总结提升",
                "duration": f"第{total_days*3//4+1}-{total_days}天",
                "description": "总结知识点，查漏补缺，准备考试",
                "topics": ["知识梳理", "模拟测试", "重点复习"]
            }
        ]
        
        result = {
            "path_id": 1,
            "subject": request.subject,
            "total_days": total_days,
            "total_hours": total_hours,
            "milestones": 4,
            "phases": phases,
            "created_at": "2024-01-15T10:30:00"
        }
        
        return success_response(data=result, message="学习路径生成成功")
        
    except Exception as e:
        logger.error(f"学习路径生成失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/paths", summary="获取学习路径列表")
async def get_learning_paths(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    subject: Optional[str] = Query(None)
):
    """获取用户的学习路径列表"""
    try:
        # 模拟学习路径列表
        paths = [
            {
                "id": 1,
                "subject": "数学",
                "grade": "高一",
                "learning_goal": "提高数学成绩",
                "total_days": 30,
                "total_hours": 30,
                "progress": 65,
                "status": "in_progress",
                "created_at": "2024-01-15T10:30:00"
            },
            {
                "id": 2,
                "subject": "物理",
                "grade": "高二",
                "learning_goal": "掌握力学基础",
                "total_days": 60,
                "total_hours": 90,
                "progress": 100,
                "status": "completed",
                "created_at": "2024-01-10T09:00:00"
            }
        ]
        
        if subject:
            paths = [p for p in paths if p["subject"] == subject]
        
        result = {
            "paths": paths,
            "total": len(paths),
            "page": page,
            "size": size
        }
        
        return success_response(data=result, message="获取学习路径列表成功")
        
    except Exception as e:
        logger.error(f"获取学习路径列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paths/{path_id}", summary="获取学习路径详情")
async def get_learning_path_detail(path_id: int):
    """获取学习路径详情"""
    try:
        # 模拟学习路径详情
        result = {
            "id": path_id,
            "subject": "数学",
            "grade": "高一",
            "learning_goal": "提高数学成绩，掌握函数概念",
            "duration": "1month",
            "daily_time": "1hour",
            "intensity": "moderate",
            "current_level": "intermediate",
            "total_days": 30,
            "total_hours": 30,
            "progress": 65,
            "status": "in_progress",
            "phases": [
                {
                    "title": "基础概念掌握",
                    "duration": "第1-7天",
                    "description": "学习基本概念和原理",
                    "topics": ["基本定义", "核心概念"],
                    "progress": 100,
                    "status": "completed"
                },
                {
                    "title": "技能训练",
                    "duration": "第8-14天",
                    "description": "通过练习巩固基础知识",
                    "topics": ["基础练习", "技巧训练"],
                    "progress": 80,
                    "status": "in_progress"
                }
            ],
            "created_at": "2024-01-15T10:30:00"
        }
        
        return success_response(data=result, message="获取学习路径详情成功")
        
    except Exception as e:
        logger.error(f"获取学习路径详情失败: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/paths/{path_id}/progress", summary="更新学习进度")
async def update_learning_progress(path_id: int, phase_id: int, progress: int):
    """更新学习路径进度"""
    try:
        # 模拟更新进度
        result = {
            "path_id": path_id,
            "phase_id": phase_id,
            "progress": progress,
            "updated_at": "2024-01-15T10:35:00"
        }
        
        return success_response(data=result, message="学习进度更新成功")
        
    except Exception as e:
        logger.error(f"更新学习进度失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
