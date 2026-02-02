"""
聊天相关数据模型
"""
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime


class MessageBase(BaseModel):
    """消息基础模型"""
    content: str
    message_type: str  # 'user' or 'ai'


class Message(MessageBase):
    """消息模型"""
    id: int
    timestamp: datetime
    chat_id: Optional[int] = None
    
    class Config:
        from_attributes = True


class ChatBase(BaseModel):
    """聊天基础模型"""
    title: str


class Chat(ChatBase):
    """聊天模型"""
    id: int
    created_at: datetime
    updated_at: datetime
    messages: List[Message] = []
    
    class Config:
        from_attributes = True


class ChatCreate(ChatBase):
    """创建聊天模型"""
    pass


class AIQuestionRequest(BaseModel):
    """AI问答请求模型"""
    question: str
    
    class Config:
        schema_extra = {
            "example": {
                "question": "请帮我设计一个关于机器学习的课程大纲"
            }
        }


class AIQuestionResponse(BaseModel):
    """AI问答响应模型"""
    answer: str
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "好的，我来为您设计一个机器学习课程大纲..."
            }
        }
