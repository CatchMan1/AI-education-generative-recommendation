"""
聊天相关Schema
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class ChatHistoryMessage(BaseModel):
    """对话历史消息"""
    role: str = Field(..., pattern="^(user|assistant|system)$", description="角色, user, assistant, or system")
    content: str = Field(..., description="消息内容")


class AIQuestionRequest(BaseModel):
    """AI问答请求Schema"""
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    document_text: Optional[str] = Field(None, description="上传的文档内容，用于RAG")
    history: List[ChatHistoryMessage] = Field([], description="对话历史")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "请帮我设计一个关于机器学习的课程大纲",
                "document_text": "这是一些关于机器学习的文档内容",
                "history": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好，有什么可以帮你的吗？"}
                ]
            }
        }


class AIQuestionResponse(BaseModel):
    """AI问答响应Schema"""
    answer: str = Field(..., description="AI回答")
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "好的，我来为您设计一个机器学习课程大纲..."
            }
        }


class MessageBase(BaseModel):
    """消息基础Schema"""
    content: str
    message_type: str = Field(..., pattern="^(user|ai)$", description="消息类型")


class Message(MessageBase):
    """消息Schema"""
    id: int
    timestamp: datetime
    chat_id: Optional[int] = None
    
    class Config:
        from_attributes = True


class ChatBase(BaseModel):
    """聊天基础Schema"""
    title: str = Field(..., min_length=1, max_length=100)


class Chat(ChatBase):
    """聊天Schema"""
    id: int
    created_at: datetime
    updated_at: datetime
    messages: List[Message] = []
    
    class Config:
        from_attributes = True


class ChatCreate(ChatBase):
    """创建聊天Schema"""
    pass


class ChatListResponse(BaseModel):
    """聊天列表响应"""
    chats: List[Chat]
    total: int
