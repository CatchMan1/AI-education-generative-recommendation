"""
文本梳理API路由
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.schemas.text_organization import (
    FileUploadRequest,
    FileUploadResponse,
    TextProcessingRequest,
    TextProcessingResponse,
    ProcessingStatusResponse,
    ProcessingResultsResponse,
    DocumentListResponse,
    ProcessingHistoryResponse
)
from app.services.text_organization_service import text_organization_service
from app.utils.response import success_response, error_response
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/text-organization", tags=["文本梳理"])


@router.post("/upload", response_model=FileUploadResponse, summary="上传文档")
async def upload_document(request: FileUploadRequest):
    """
    上传文档进行文本梳理
    
    - **title**: 文档标题
    - **file_type**: 文件类型 (txt, doc, docx, pdf)
    - **content**: 文件内容
    """
    try:
        result = await text_organization_service.upload_document(request)
        return success_response(data=result, message="文档上传成功")
    
    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/process", response_model=TextProcessingResponse, summary="开始文本处理")
async def start_text_processing(request: TextProcessingRequest):
    """
    开始文本处理任务
    
    - **document_id**: 文档ID
    - **processing_options**: 处理选项列表
      - keyword_extraction: 关键词提取
      - content_summary: 内容摘要
      - structure_organization: 结构化整理
      - grammar_check: 语法检查
    - **custom_instructions**: 自定义处理指令（可选）
    """
    try:
        result = await text_organization_service.start_processing(request)
        return success_response(data=result, message="文本处理任务已启动")
    
    except Exception as e:
        logger.error(f"启动文本处理失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status/{task_id}", response_model=ProcessingStatusResponse, summary="获取处理状态")
async def get_processing_status(task_id: int):
    """
    获取文本处理任务状态
    
    - **task_id**: 任务ID
    """
    try:
        result = await text_organization_service.get_processing_status(task_id)
        return success_response(data=result, message="获取处理状态成功")
    
    except Exception as e:
        logger.error(f"获取处理状态失败: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/results/{task_id}", response_model=ProcessingResultsResponse, summary="获取处理结果")
async def get_processing_results(task_id: int):
    """
    获取文本处理结果
    
    - **task_id**: 任务ID
    """
    try:
        result = await text_organization_service.get_processing_results(task_id)
        return success_response(data=result, message="获取处理结果成功")
    
    except Exception as e:
        logger.error(f"获取处理结果失败: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse, summary="获取文档列表")
async def get_document_list(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(10, ge=1, le=100, description="每页数量"),
    search: Optional[str] = Query(None, description="搜索关键词")
):
    """
    获取用户的文档列表
    
    - **page**: 页码（从1开始）
    - **size**: 每页数量（1-100）
    - **search**: 搜索关键词（可选）
    """
    try:
        # 模拟文档列表数据
        documents = [
            {
                "id": 1,
                "title": "机器学习基础文档",
                "file_type": "txt",
                "file_size": 1024,
                "uploaded_at": "2024-01-15T10:30:00",
                "last_processed": "2024-01-15T10:35:00",
                "processing_count": 3
            },
            {
                "id": 2,
                "title": "深度学习研究报告",
                "file_type": "pdf",
                "file_size": 2048,
                "uploaded_at": "2024-01-14T15:20:00",
                "last_processed": "2024-01-14T15:25:00",
                "processing_count": 1
            }
        ]
        
        # 应用搜索过滤
        if search:
            documents = [doc for doc in documents if search.lower() in doc["title"].lower()]
        
        # 分页处理
        start = (page - 1) * size
        end = start + size
        paginated_docs = documents[start:end]
        
        result = {
            "documents": paginated_docs,
            "total": len(documents),
            "page": page,
            "size": size
        }
        
        return success_response(data=result, message="获取文档列表成功")
    
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=ProcessingHistoryResponse, summary="获取处理历史")
async def get_processing_history(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(10, ge=1, le=100, description="每页数量"),
    document_id: Optional[int] = Query(None, description="文档ID过滤")
):
    """
    获取文本处理历史记录
    
    - **page**: 页码（从1开始）
    - **size**: 每页数量（1-100）
    - **document_id**: 文档ID过滤（可选）
    """
    try:
        # 模拟历史记录数据
        history = [
            {
                "id": 1,
                "document_id": 1,
                "task_id": 1,
                "document_title": "机器学习基础文档",
                "file_type": "txt",
                "file_size": 1024,
                "processing_options": ["keyword_extraction", "content_summary"],
                "status": "completed",
                "created_at": "2024-01-15T10:30:00",
                "completed_at": "2024-01-15T10:35:00"
            },
            {
                "id": 2,
                "document_id": 2,
                "task_id": 2,
                "document_title": "深度学习研究报告",
                "file_type": "pdf",
                "file_size": 2048,
                "processing_options": ["structure_organization"],
                "status": "completed",
                "created_at": "2024-01-14T15:20:00",
                "completed_at": "2024-01-14T15:25:00"
            }
        ]
        
        # 应用文档ID过滤
        if document_id:
            history = [h for h in history if h["document_id"] == document_id]
        
        # 分页处理
        start = (page - 1) * size
        end = start + size
        paginated_history = history[start:end]
        
        result = {
            "history": paginated_history,
            "total": len(history),
            "page": page,
            "size": size
        }
        
        return success_response(data=result, message="获取处理历史成功")
    
    except Exception as e:
        logger.error(f"获取处理历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}", summary="删除文档")
async def delete_document(document_id: int):
    """
    删除文档及其相关的处理记录
    
    - **document_id**: 文档ID
    """
    try:
        # 这里应该实现删除逻辑
        # 1. 删除文档文件
        # 2. 删除数据库记录
        # 3. 删除相关的处理任务和结果
        
        logger.info(f"删除文档: {document_id}")
        
        return success_response(message="文档删除成功")
    
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="获取统计信息")
async def get_statistics():
    """
    获取文本梳理功能的统计信息
    """
    try:
        # 模拟统计数据
        stats = {
            "total_documents": 25,
            "total_processing_tasks": 48,
            "completed_tasks": 45,
            "failed_tasks": 3,
            "most_used_options": [
                {"option": "keyword_extraction", "count": 30},
                {"option": "content_summary", "count": 25},
                {"option": "structure_organization", "count": 15},
                {"option": "grammar_check", "count": 10}
            ],
            "file_type_distribution": {
                "txt": 15,
                "pdf": 8,
                "docx": 2
            },
            "average_processing_time": 45.5  # 秒
        }
        
        return success_response(data=stats, message="获取统计信息成功")
    
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
