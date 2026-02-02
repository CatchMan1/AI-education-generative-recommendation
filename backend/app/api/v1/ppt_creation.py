"""
PPT制作API路由
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from app.schemas.ppt_creation import (
    PPTCreationRequest,
    PPTCreationResponse,
    PPTGenerationStatusResponse,
    PPTGenerationResultResponse,
    PPTTemplateResponse
)
from app.utils.response import success_response
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/ppt-creation", tags=["PPT制作"])


@router.post("/create", response_model=PPTCreationResponse, summary="创建PPT项目")
async def create_ppt_project(request: PPTCreationRequest):
    """创建PPT制作项目并开始生成"""
    try:
        # 计算预估时间
        slide_count = int(request.slide_count) if request.slide_count != "auto" else 15
        estimated_time = slide_count * 10  # 每页预估10秒
        
        result = {
            "project_id": 1,
            "task_id": 1,
            "title": request.title,
            "status": "generating",
            "progress": 0,
            "estimated_time": estimated_time
        }
        
        return success_response(data=result, message="PPT项目创建成功，开始生成")
        
    except Exception as e:
        logger.error(f"PPT项目创建失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status/{task_id}", response_model=PPTGenerationStatusResponse, summary="获取生成状态")
async def get_generation_status(task_id: int):
    """获取PPT生成状态"""
    try:
        # 模拟生成状态
        result = {
            "task_id": task_id,
            "project_id": 1,
            "status": "generating",
            "progress": 75,
            "current_step": "生成页面布局...",
            "error_message": None
        }
        
        return success_response(data=result, message="获取生成状态成功")
        
    except Exception as e:
        logger.error(f"获取生成状态失败: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/result/{task_id}", response_model=PPTGenerationResultResponse, summary="获取生成结果")
async def get_generation_result(task_id: int):
    """获取PPT生成结果"""
    try:
        # 模拟生成结果
        slides = [
            {
                "slide_number": 1,
                "title": "人工智能在教育中的应用",
                "content": "主标题页面",
                "layout_type": "title",
                "notes": "开场介绍"
            },
            {
                "slide_number": 2,
                "title": "目录",
                "content": "1. AI概述\n2. 教育应用\n3. 案例分析\n4. 发展趋势",
                "layout_type": "content",
                "notes": "课程大纲"
            }
        ]
        
        result = {
            "task_id": task_id,
            "project_id": 1,
            "title": "人工智能在教育中的应用",
            "status": "completed",
            "file_path": "/files/ppt/ai_education_20240115.pptx",
            "file_size": 2048000,
            "slide_count": 15,
            "slides": slides,
            "completed_at": "2024-01-15T10:35:00",
            "generation_time": 180.5
        }
        
        return success_response(data=result, message="获取生成结果成功")
        
    except Exception as e:
        logger.error(f"获取生成结果失败: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/templates", summary="获取PPT模板")
async def get_ppt_templates():
    """获取可用的PPT模板列表"""
    try:
        templates = [
            {
                "id": 1,
                "name": "商务蓝",
                "style": "business_blue",
                "description": "专业商务风格模板",
                "preview_image": "/images/templates/business_blue.png",
                "color_scheme": {
                    "primary": "#1e40af",
                    "secondary": "#3b82f6",
                    "accent": "#60a5fa"
                }
            },
            {
                "id": 2,
                "name": "学术绿",
                "style": "academic_green",
                "description": "清新学术风格模板",
                "preview_image": "/images/templates/academic_green.png",
                "color_scheme": {
                    "primary": "#059669",
                    "secondary": "#10b981",
                    "accent": "#34d399"
                }
            },
            {
                "id": 3,
                "name": "科技紫",
                "style": "tech_purple",
                "description": "现代科技风格模板",
                "preview_image": "/images/templates/tech_purple.png",
                "color_scheme": {
                    "primary": "#7c3aed",
                    "secondary": "#8b5cf6",
                    "accent": "#a78bfa"
                }
            }
        ]
        
        return success_response(data=templates, message="获取PPT模板成功")
        
    except Exception as e:
        logger.error(f"获取PPT模板失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects", summary="获取PPT项目列表")
async def get_ppt_projects(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    status: Optional[str] = Query(None)
):
    """获取用户的PPT项目列表"""
    try:
        projects = [
            {
                "id": 1,
                "title": "人工智能在教育中的应用",
                "template_style": "business_blue",
                "slide_count": 15,
                "status": "completed",
                "created_at": "2024-01-15T10:30:00",
                "completed_at": "2024-01-15T10:35:00",
                "file_size": 2048000
            },
            {
                "id": 2,
                "title": "机器学习基础概念",
                "template_style": "academic_green",
                "slide_count": 20,
                "status": "generating",
                "created_at": "2024-01-14T15:20:00",
                "completed_at": None,
                "file_size": None
            }
        ]
        
        if status:
            projects = [p for p in projects if p["status"] == status]
        
        result = {
            "projects": projects,
            "total": len(projects),
            "page": page,
            "size": size
        }
        
        return success_response(data=result, message="获取PPT项目列表成功")
        
    except Exception as e:
        logger.error(f"获取PPT项目列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}", summary="获取PPT项目详情")
async def get_ppt_project_detail(project_id: int):
    """获取PPT项目详情"""
    try:
        result = {
            "id": project_id,
            "title": "人工智能在教育中的应用",
            "outline": "1. AI概述\n2. 教育应用\n3. 案例分析\n4. 发展趋势",
            "template_style": "business_blue",
            "slide_count": "15",
            "language_style": "formal",
            "include_images": True,
            "status": "completed",
            "created_at": "2024-01-15T10:30:00",
            "file_path": "/files/ppt/ai_education_20240115.pptx",
            "file_size": 2048000
        }
        
        return success_response(data=result, message="获取PPT项目详情成功")
        
    except Exception as e:
        logger.error(f"获取PPT项目详情失败: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/projects/{project_id}", summary="删除PPT项目")
async def delete_ppt_project(project_id: int):
    """删除PPT项目"""
    try:
        logger.info(f"删除PPT项目: {project_id}")
        return success_response(message="PPT项目删除成功")
        
    except Exception as e:
        logger.error(f"删除PPT项目失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
