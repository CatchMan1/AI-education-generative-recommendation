"""
文本梳理相关Schema
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class FileUploadRequest(BaseModel):
    """文件上传请求Schema"""
    title: str = Field(..., min_length=1, max_length=200, description="文档标题")
    file_type: str = Field(..., pattern="^(txt|doc|docx|pdf)$", description="文件类型")
    content: str = Field(..., min_length=1, max_length=1000000, description="文件内容")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "机器学习基础文档",
                "file_type": "txt",
                "content": "机器学习是人工智能的一个重要分支..."
            }
        }


class FileUploadResponse(BaseModel):
    """文件上传响应Schema"""
    document_id: int = Field(..., description="文档ID")
    title: str = Field(..., description="文档标题")
    file_type: str = Field(..., description="文件类型")
    file_size: int = Field(..., description="文件大小（字节）")
    uploaded_at: datetime = Field(..., description="上传时间")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": 1,
                "title": "机器学习基础文档",
                "file_type": "txt",
                "file_size": 1024,
                "uploaded_at": "2024-01-15T10:30:00"
            }
        }


class TextProcessingRequest(BaseModel):
    """文本处理请求Schema"""
    document_id: int = Field(..., description="文档ID")
    processing_options: List[str] = Field(
        ..., 
        min_items=1,
        description="处理选项列表"
    )
    custom_instructions: Optional[str] = Field(
        None, 
        max_length=1000,
        description="自定义处理指令"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": 1,
                "processing_options": ["keyword_extraction", "content_summary"],
                "custom_instructions": "请重点关注技术术语的提取"
            }
        }


class TextProcessingResponse(BaseModel):
    """文本处理响应Schema"""
    task_id: int = Field(..., description="任务ID")
    document_id: int = Field(..., description="文档ID")
    status: str = Field(..., description="处理状态")
    progress: int = Field(..., description="进度百分比")
    estimated_time: Optional[int] = Field(None, description="预估完成时间（秒）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": 1,
                "document_id": 1,
                "status": "processing",
                "progress": 25,
                "estimated_time": 120
            }
        }


class ProcessingStatusResponse(BaseModel):
    """处理状态响应Schema"""
    task_id: int = Field(..., description="任务ID")
    status: str = Field(..., description="处理状态")
    progress: int = Field(..., description="进度百分比")
    current_step: str = Field(..., description="当前处理步骤")
    error_message: Optional[str] = Field(None, description="错误信息")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": 1,
                "status": "processing",
                "progress": 75,
                "current_step": "正在生成内容摘要...",
                "error_message": None
            }
        }


class KeywordExtractionResponse(BaseModel):
    """关键词提取响应Schema"""
    keywords: List[str] = Field(..., description="提取的关键词")
    keyword_scores: Dict[str, float] = Field(..., description="关键词权重分数")
    total_keywords: int = Field(..., description="关键词总数")
    
    class Config:
        json_schema_extra = {
            "example": {
                "keywords": ["机器学习", "人工智能", "深度学习"],
                "keyword_scores": {
                    "机器学习": 0.95,
                    "人工智能": 0.87,
                    "深度学习": 0.76
                },
                "total_keywords": 3
            }
        }


class ContentSummaryResponse(BaseModel):
    """内容摘要响应Schema"""
    summary: str = Field(..., description="内容摘要")
    key_points: List[str] = Field(..., description="关键要点")
    summary_length: int = Field(..., description="摘要字数")
    compression_ratio: float = Field(..., description="压缩比例")
    
    class Config:
        json_schema_extra = {
            "example": {
                "summary": "本文档主要介绍了机器学习的基本概念和应用...",
                "key_points": [
                    "机器学习是AI的重要分支",
                    "包含监督学习和无监督学习",
                    "在多个领域有广泛应用"
                ],
                "summary_length": 150,
                "compression_ratio": 0.15
            }
        }


class ProcessingResultsResponse(BaseModel):
    """处理结果响应Schema"""
    task_id: int = Field(..., description="任务ID")
    document_id: int = Field(..., description="文档ID")
    status: str = Field(..., description="处理状态")
    results: Dict = Field(..., description="处理结果")
    completed_at: datetime = Field(..., description="完成时间")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": 1,
                "document_id": 1,
                "status": "completed",
                "results": {
                    "keyword_extraction": {
                        "keywords": ["机器学习", "人工智能"],
                        "total_keywords": 2
                    },
                    "content_summary": {
                        "summary": "本文档介绍机器学习基础概念...",
                        "key_points": ["概念介绍", "应用场景"]
                    }
                },
                "completed_at": "2024-01-15T10:35:00"
            }
        }


class DocumentListResponse(BaseModel):
    """文档列表响应Schema"""
    documents: List[Dict] = Field(..., description="文档列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页码")
    size: int = Field(..., description="每页数量")


class ProcessingHistoryResponse(BaseModel):
    """处理历史响应Schema"""
    history: List[Dict] = Field(..., description="处理历史列表")
    total: int = Field(..., description="总数量")
    page: int = Field(..., description="当前页码")
    size: int = Field(..., description="每页数量")
