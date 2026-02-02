"""
PPT制作相关Schema
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class PPTCreationRequest(BaseModel):
    """PPT创建请求Schema"""
    title: str = Field(..., min_length=1, max_length=200, description="PPT标题")
    outline: str = Field(..., min_length=10, max_length=5000, description="内容大纲")
    template_style: str = Field(
        "business_blue",
        pattern="^(business_blue|academic_green|tech_purple|simple_white|creative_orange)$",
        description="模板样式"
    )
    slide_count: str = Field("auto", description="页数设置")
    language_style: str = Field(
        "formal",
        pattern="^(formal|casual|academic|business)$",
        description="语言风格"
    )
    include_images: bool = Field(True, description="是否包含图片")
    custom_instructions: Optional[str] = Field(None, max_length=1000, description="自定义指令")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "人工智能在教育中的应用",
                "outline": "1. 人工智能概述\n2. AI在教育领域的应用\n3. 具体案例分析\n4. 未来发展趋势",
                "template_style": "business_blue",
                "slide_count": "15",
                "language_style": "formal",
                "include_images": True,
                "custom_instructions": "请重点突出技术优势"
            }
        }


class PPTCreationResponse(BaseModel):
    """PPT创建响应Schema"""
    project_id: int = Field(..., description="项目ID")
    task_id: int = Field(..., description="任务ID")
    title: str = Field(..., description="PPT标题")
    status: str = Field(..., description="生成状态")
    progress: int = Field(..., description="进度百分比")
    estimated_time: Optional[int] = Field(None, description="预估完成时间（秒）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "project_id": 1,
                "task_id": 1,
                "title": "人工智能在教育中的应用",
                "status": "generating",
                "progress": 0,
                "estimated_time": 180
            }
        }


class PPTGenerationStatusResponse(BaseModel):
    """PPT生成状态响应Schema"""
    task_id: int = Field(..., description="任务ID")
    project_id: int = Field(..., description="项目ID")
    status: str = Field(..., description="生成状态")
    progress: int = Field(..., description="进度百分比")
    current_step: str = Field(..., description="当前步骤")
    error_message: Optional[str] = Field(None, description="错误信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": 1,
                "project_id": 1,
                "status": "generating",
                "progress": 65,
                "current_step": "生成页面布局...",
                "error_message": None
            }
        }


class PPTSlideResponse(BaseModel):
    """PPT幻灯片响应Schema"""
    slide_number: int = Field(..., description="幻灯片编号")
    title: str = Field(..., description="幻灯片标题")
    content: str = Field(..., description="幻灯片内容")
    layout_type: str = Field(..., description="布局类型")
    notes: Optional[str] = Field(None, description="备注")


class PPTGenerationResultResponse(BaseModel):
    """PPT生成结果响应Schema"""
    task_id: int = Field(..., description="任务ID")
    project_id: int = Field(..., description="项目ID")
    title: str = Field(..., description="PPT标题")
    status: str = Field(..., description="生成状态")
    file_path: Optional[str] = Field(None, description="文件路径")
    file_size: Optional[int] = Field(None, description="文件大小")
    slide_count: Optional[int] = Field(None, description="幻灯片数量")
    slides: Optional[List[PPTSlideResponse]] = Field(None, description="幻灯片列表")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    generation_time: Optional[float] = Field(None, description="生成耗时")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": 1,
                "project_id": 1,
                "title": "人工智能在教育中的应用",
                "status": "completed",
                "file_path": "/files/ppt/ai_education_20240115.pptx",
                "file_size": 2048000,
                "slide_count": 15,
                "completed_at": "2024-01-15T10:35:00",
                "generation_time": 180.5
            }
        }


class PPTTemplateResponse(BaseModel):
    """PPT模板响应Schema"""
    id: int = Field(..., description="模板ID")
    name: str = Field(..., description="模板名称")
    style: str = Field(..., description="模板样式")
    description: str = Field(..., description="模板描述")
    preview_image: str = Field(..., description="预览图片")
    color_scheme: Dict[str, str] = Field(..., description="颜色方案")
    
    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


class PPTProjectListResponse(BaseModel):
    """PPT项目列表响应Schema"""
    projects: List[Dict] = Field(..., description="项目列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页码")
    size: int = Field(..., description="每页数量")


class PPTHistoryResponse(BaseModel):
    """PPT历史记录响应Schema"""
    history: List[Dict] = Field(..., description="历史记录列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页码")
    size: int = Field(..., description="每页数量")


class PPTStatsResponse(BaseModel):
    """PPT统计响应Schema"""
    total_projects: int = Field(..., description="总项目数")
    completed_projects: int = Field(..., description="已完成项目数")
    failed_projects: int = Field(..., description="失败项目数")
    total_slides: int = Field(..., description="总幻灯片数")
    average_slides_per_project: float = Field(..., description="平均每项目幻灯片数")
    most_used_template: str = Field(..., description="最常用模板")
    template_usage: Dict[str, int] = Field(..., description="模板使用统计")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_projects": 50,
                "completed_projects": 45,
                "failed_projects": 5,
                "total_slides": 675,
                "average_slides_per_project": 13.5,
                "most_used_template": "business_blue",
                "template_usage": {
                    "business_blue": 20,
                    "academic_green": 15,
                    "tech_purple": 10,
                    "simple_white": 5
                }
            }
        }
