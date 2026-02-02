"""
PPT制作相关数据模型
"""
from typing import Optional, List, Dict
from pydantic import BaseModel
from datetime import datetime
from enum import Enum


class PPTStatus(str, Enum):
    """PPT生成状态枚举"""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class TemplateStyle(str, Enum):
    """模板样式枚举"""
    BUSINESS_BLUE = "business_blue"
    ACADEMIC_GREEN = "academic_green"
    TECH_PURPLE = "tech_purple"
    SIMPLE_WHITE = "simple_white"
    CREATIVE_ORANGE = "creative_orange"


class LanguageStyle(str, Enum):
    """语言风格枚举"""
    FORMAL = "formal"
    CASUAL = "casual"
    ACADEMIC = "academic"
    BUSINESS = "business"


class PPTProjectBase(BaseModel):
    """PPT项目基础模型"""
    title: str
    outline: str
    template_style: TemplateStyle
    slide_count: str  # 'auto' 或具体数字
    language_style: LanguageStyle
    include_images: bool = True
    user_id: Optional[int] = None


class PPTProject(PPTProjectBase):
    """PPT项目模型"""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class PPTGenerationTaskBase(BaseModel):
    """PPT生成任务基础模型"""
    project_id: int
    custom_instructions: Optional[str] = None


class PPTGenerationTask(PPTGenerationTaskBase):
    """PPT生成任务模型"""
    id: int
    status: PPTStatus
    progress: int = 0  # 进度百分比
    current_step: str = ""
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class PPTSlideBase(BaseModel):
    """PPT幻灯片基础模型"""
    project_id: int
    slide_number: int
    title: str
    content: str
    layout_type: str  # 'title', 'content', 'image', 'chart'
    notes: Optional[str] = None


class PPTSlide(PPTSlideBase):
    """PPT幻灯片模型"""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class PPTTemplateBase(BaseModel):
    """PPT模板基础模型"""
    name: str
    style: TemplateStyle
    description: str
    preview_image: str
    color_scheme: Dict[str, str]  # 颜色方案
    font_settings: Dict[str, str]  # 字体设置


class PPTTemplate(PPTTemplateBase):
    """PPT模板模型"""
    id: int
    is_active: bool = True
    created_at: datetime
    
    class Config:
        from_attributes = True


class PPTGenerationResultBase(BaseModel):
    """PPT生成结果基础模型"""
    task_id: int
    project_id: int
    file_path: str
    file_size: int
    slide_count: int
    generation_time: float  # 生成耗时（秒）


class PPTGenerationResult(PPTGenerationResultBase):
    """PPT生成结果模型"""
    id: int
    created_at: datetime
    download_count: int = 0
    last_downloaded: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class PPTGenerationHistory(BaseModel):
    """PPT生成历史记录"""
    id: int
    project_id: int
    task_id: int
    title: str
    template_style: str
    slide_count: int
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    file_size: Optional[int] = None
    
    class Config:
        from_attributes = True


class PPTProjectStats(BaseModel):
    """PPT项目统计"""
    total_projects: int
    completed_projects: int
    failed_projects: int
    total_slides: int
    average_slides_per_project: float
    most_used_template: str
    total_generation_time: float
