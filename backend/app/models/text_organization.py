"""
文本梳理相关数据模型
"""
from typing import Optional, List, Dict
from pydantic import BaseModel
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(str, Enum):
    """文件类型枚举"""
    TXT = "txt"
    DOC = "doc"
    DOCX = "docx"
    PDF = "pdf"


class ProcessingOption(str, Enum):
    """处理选项枚举"""
    KEYWORD_EXTRACTION = "keyword_extraction"
    CONTENT_SUMMARY = "content_summary"
    STRUCTURE_ORGANIZATION = "structure_organization"
    GRAMMAR_CHECK = "grammar_check"


class TextDocumentBase(BaseModel):
    """文本文档基础模型"""
    title: str
    file_type: FileType
    file_size: int  # 文件大小（字节）
    original_content: str
    user_id: Optional[int] = None


class TextDocument(TextDocumentBase):
    """文本文档模型"""
    id: int
    file_path: str
    uploaded_at: datetime
    
    class Config:
        from_attributes = True


class TextProcessingTaskBase(BaseModel):
    """文本处理任务基础模型"""
    document_id: int
    processing_options: List[ProcessingOption]
    custom_instructions: Optional[str] = None


class TextProcessingTask(TextProcessingTaskBase):
    """文本处理任务模型"""
    id: int
    status: ProcessingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: int = 0  # 进度百分比
    
    class Config:
        from_attributes = True


class ProcessingResultBase(BaseModel):
    """处理结果基础模型"""
    task_id: int
    result_type: str  # 'keywords', 'summary', 'structure', 'grammar'
    result_data: Dict  # JSON格式的结果数据


class ProcessingResult(ProcessingResultBase):
    """处理结果模型"""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class KeywordExtractionResult(BaseModel):
    """关键词提取结果"""
    keywords: List[str]
    keyword_scores: Dict[str, float]
    total_keywords: int


class ContentSummaryResult(BaseModel):
    """内容摘要结果"""
    summary: str
    key_points: List[str]
    summary_length: int
    compression_ratio: float


class StructureOrganizationResult(BaseModel):
    """结构化整理结果"""
    organized_content: str
    sections: List[Dict[str, str]]  # [{"title": "标题", "content": "内容"}]
    outline: List[str]


class GrammarCheckResult(BaseModel):
    """语法检查结果"""
    corrected_content: str
    errors_found: List[Dict[str, str]]  # [{"type": "错误类型", "original": "原文", "corrected": "修正"}]
    error_count: int
    accuracy_score: float


class TextProcessingHistory(BaseModel):
    """文本处理历史记录"""
    id: int
    document_id: int
    task_id: int
    document_title: str
    file_type: str
    file_size: int
    processing_options: List[str]
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
