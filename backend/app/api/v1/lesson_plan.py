"""
教案生成API路由
"""
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from app.utils.response import success_response
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/lesson-plan", tags=["教案生成"])


class LessonPlanRequest(BaseModel):
    """教案生成请求"""
    course_name: str = Field(..., description="课程名称")
    subject: str = Field(..., description="学科")
    grade: str = Field(..., description="年级")
    duration: str = Field(..., description="课时长度")
    objectives: str = Field(..., description="教学目标")
    key_points: str = Field(..., description="教学重点")
    difficulties: str = Field(..., description="教学难点")
    teaching_methods: List[str] = Field(..., description="教学方法")
    resources: List[str] = Field(..., description="教学资源")
    student_analysis: Optional[str] = Field(None, description="学生特点分析")


class LessonPlanResponse(BaseModel):
    """教案生成响应"""
    plan_id: int = Field(..., description="教案ID")
    course_name: str = Field(..., description="课程名称")
    subject: str = Field(..., description="学科")
    grade: str = Field(..., description="年级")
    duration: str = Field(..., description="课时长度")
    steps: List[dict] = Field(..., description="教学步骤")
    evaluations: List[str] = Field(..., description="评价方案")
    created_at: str = Field(..., description="创建时间")


@router.post("/generate", response_model=LessonPlanResponse, summary="生成教案")
async def generate_lesson_plan(request: LessonPlanRequest):
    """AI智能生成教学方案"""
    try:
        # 根据课时长度分配时间
        duration_minutes = int(request.duration)
        
        # 生成教学步骤
        steps = [
            {
                "title": "导入新课",
                "description": f"通过生活实例引入{request.course_name}概念，激发学生学习兴趣",
                "time": f"{max(5, duration_minutes // 9)}分钟",
                "activities": ["问题导入", "情境创设", "学习目标展示"]
            },
            {
                "title": "概念讲解",
                "description": f"详细讲解{request.course_name}的核心概念，通过图表和实例帮助理解",
                "time": f"{duration_minutes // 3}分钟",
                "activities": ["概念阐述", "例题演示", "互动问答"]
            },
            {
                "title": "例题演示",
                "description": "通过典型例题演示解题方法和思路",
                "time": f"{duration_minutes // 4}分钟",
                "activities": ["例题分析", "解题步骤", "方法总结"]
            },
            {
                "title": "课堂练习",
                "description": "学生独立完成练习题，教师巡视指导",
                "time": f"{duration_minutes // 4}分钟",
                "activities": ["独立练习", "小组讨论", "个别指导"]
            },
            {
                "title": "总结归纳",
                "description": "总结本节课重点内容，布置课后作业",
                "time": f"{max(5, duration_minutes // 9)}分钟",
                "activities": ["知识梳理", "重点回顾", "作业布置"]
            }
        ]
        
        # 生成评价方案
        evaluations = [
            "课堂提问评价学生对概念的理解程度",
            "练习题完成情况反映学生掌握水平",
            "课后作业检验学习效果",
            "小组讨论观察学生参与度"
        ]
        
        # 根据教学方法调整评价方案
        if "discussion" in request.teaching_methods:
            evaluations.append("小组讨论表现评价")
        if "demonstration" in request.teaching_methods:
            evaluations.append("演示操作技能评价")
        
        result = {
            "plan_id": 1,
            "course_name": request.course_name,
            "subject": request.subject,
            "grade": request.grade,
            "duration": request.duration,
            "steps": steps,
            "evaluations": evaluations,
            "created_at": "2024-01-15T10:30:00"
        }
        
        return success_response(data=result, message="教案生成成功")
        
    except Exception as e:
        logger.error(f"教案生成失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/plans", summary="获取教案列表")
async def get_lesson_plans(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    subject: Optional[str] = Query(None),
    grade: Optional[str] = Query(None)
):
    """获取教案列表"""
    try:
        # 模拟教案列表
        plans = [
            {
                "id": 1,
                "course_name": "函数的概念与性质",
                "subject": "数学",
                "grade": "高一",
                "duration": "45",
                "created_at": "2024-01-15T10:30:00",
                "updated_at": "2024-01-15T10:35:00",
                "status": "completed"
            },
            {
                "id": 2,
                "course_name": "牛顿第一定律",
                "subject": "物理",
                "grade": "高一",
                "duration": "40",
                "created_at": "2024-01-14T15:20:00",
                "updated_at": "2024-01-14T15:25:00",
                "status": "completed"
            }
        ]
        
        # 应用过滤条件
        if subject:
            plans = [p for p in plans if p["subject"] == subject]
        if grade:
            plans = [p for p in plans if p["grade"] == grade]
        
        result = {
            "plans": plans,
            "total": len(plans),
            "page": page,
            "size": size
        }
        
        return success_response(data=result, message="获取教案列表成功")
        
    except Exception as e:
        logger.error(f"获取教案列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plans/{plan_id}", summary="获取教案详情")
async def get_lesson_plan_detail(plan_id: int):
    """获取教案详情"""
    try:
        # 模拟教案详情
        result = {
            "id": plan_id,
            "course_name": "函数的概念与性质",
            "subject": "数学",
            "grade": "高一",
            "duration": "45",
            "objectives": "让学生理解函数的概念，掌握函数的表示方法",
            "key_points": "函数的定义、函数的三种表示方法",
            "difficulties": "函数概念的理解、定义域和值域的确定",
            "teaching_methods": ["lecture", "discussion", "practice"],
            "resources": ["ppt", "worksheet"],
            "student_analysis": "学生已学习集合知识，具备一定的抽象思维能力",
            "steps": [
                {
                    "title": "导入新课",
                    "description": "通过生活实例引入函数概念",
                    "time": "5分钟",
                    "activities": ["问题导入", "情境创设"]
                }
            ],
            "evaluations": [
                "课堂提问评价学生理解程度",
                "练习题检验掌握情况"
            ],
            "created_at": "2024-01-15T10:30:00"
        }
        
        return success_response(data=result, message="获取教案详情成功")
        
    except Exception as e:
        logger.error(f"获取教案详情失败: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/plans/{plan_id}", summary="更新教案")
async def update_lesson_plan(plan_id: int, request: LessonPlanRequest):
    """更新教案内容"""
    try:
        # 模拟更新教案
        result = {
            "plan_id": plan_id,
            "course_name": request.course_name,
            "updated_at": "2024-01-15T10:40:00"
        }
        
        return success_response(data=result, message="教案更新成功")
        
    except Exception as e:
        logger.error(f"教案更新失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/plans/{plan_id}", summary="删除教案")
async def delete_lesson_plan(plan_id: int):
    """删除教案"""
    try:
        # 模拟删除教案
        logger.info(f"删除教案: {plan_id}")
        
        return success_response(message="教案删除成功")
        
    except Exception as e:
        logger.error(f"删除教案失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates", summary="获取教案模板")
async def get_lesson_plan_templates():
    """获取教案模板列表"""
    try:
        templates = [
            {
                "id": 1,
                "name": "标准课堂教学模板",
                "description": "适用于常规课堂教学",
                "steps": ["导入", "新授", "练习", "总结"],
                "duration": "45分钟"
            },
            {
                "id": 2,
                "name": "实验课教学模板",
                "description": "适用于实验课教学",
                "steps": ["准备", "演示", "实验", "总结"],
                "duration": "90分钟"
            }
        ]
        
        return success_response(data=templates, message="获取教案模板成功")
        
    except Exception as e:
        logger.error(f"获取教案模板失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
