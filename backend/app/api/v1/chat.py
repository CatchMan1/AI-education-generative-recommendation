"""
AI聊天相关API路由
"""
from typing import List
from fastapi import APIRouter
from app.schemas.chat import AIQuestionRequest, AIQuestionResponse
from app.schemas.common import ResponseModel
from app.services.ai_service import ai_service
from app.utils.logger import get_logger
from app.utils.response import create_response, create_error_response

router = APIRouter()
logger = get_logger(__name__)


@router.post("/ask", response_model=AIQuestionResponse, summary="AI问答")
async def ask_ai(request: AIQuestionRequest):
    """
    AI智能问答
    
    - **question**: 用户问题，支持课程设计、PPT制作、知识图谱等教育相关问题
    
    返回AI生成的回答，包含针对性的建议和指导。
    """
    try:
        logger.info(f"收到AI问答请求: {request.question[:50]}...")
        
        # 调用AI服务获取回答
        response = await ai_service.get_ai_response(request)
        
        logger.info(f"AI问答成功，回答长度: {len(response.answer)}")
        return response
        
    except Exception as e:
        logger.error(f"AI问答失败: {str(e)}")
        raise create_error_response("AI服务暂时不可用，请稍后再试", 500)


@router.get("/suggestions", response_model=List[str], summary="获取问题建议")
async def get_suggestions():
    """
    获取快速问题建议
    
    返回一些常见的教育相关问题，用户可以直接点击使用。
    """
    try:
        logger.info("获取问题建议")
        
        suggestions = ai_service.get_quick_suggestions()
        
        logger.info(f"成功返回 {len(suggestions)} 个建议")
        return suggestions
        
    except Exception as e:
        logger.error(f"获取问题建议失败: {str(e)}")
        raise create_error_response("获取建议失败", 500)


# 兼容旧的API路径
@router.post("/", response_model=AIQuestionResponse, summary="AI问答（兼容接口）")
async def ask_ai_legacy(request: AIQuestionRequest):
    """
    AI智能问答（兼容旧版本API）
    
    这是为了兼容前端现有代码而保留的接口。
    """
    return await ask_ai(request)
