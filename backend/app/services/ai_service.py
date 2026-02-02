"""
AI服务层
"""
import random
from typing import Dict, List
from app.schemas.chat import AIQuestionRequest, AIQuestionResponse
import httpx
import os
import urllib.parse
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class AIService:
    """AI服务"""
    
    def __init__(self):
        # 从环境变量或安全配置中获取API密钥
        self.api_key = os.getenv("OPENROUTER_API_KEY", "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")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """将文本分割成有意义的块"""
        # 使用正则表达式按句子和段落分割
        sentences = re.split(r'(?<=[.!?。！？\n])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _get_relevant_context(self, question: str, document_text: str, top_k: int = 3) -> str:
        """从文档中检索与问题相关的上下文"""
        # 1. 将文档分割成块
        chunks = self._split_text_into_chunks(document_text)
        if not chunks:
            return ""

        # 2. 获取问题和文本块的嵌入向量
        question_embedding = self.embedding_model.encode([question])
        chunk_embeddings = self.embedding_model.encode(chunks)

        # 3. 计算余弦相似度
        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]

        # 4. 找到最相关的 top_k 个块
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_context = "\n".join([chunks[i] for i in top_k_indices])
        return relevant_context

    async def get_ai_response(self, request: AIQuestionRequest) -> AIQuestionResponse:
        """获取AI回答"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": urllib.parse.quote("AI教育助手"),
        }
        
        final_question = request.question
        
        # RAG流程：如果提供了文档，则检索相关上下文
        if request.document_text:
            try:
                relevant_context = self._get_relevant_context(request.question, request.document_text)
                if relevant_context:
                    # 将上下文和原始问题组合成新的提示
                    final_question = f"请根据以下上下文回答问题。\n\n上下文：\n---\n{relevant_context}\n---\n\n问题：{request.question}"
                    print("--- 使用RAG流程 ---")
                    print(f"原始问题: {request.question}")
                    print(f"检索到的上下文: {relevant_context[:200]}...")
            except Exception as e:
                print(f"RAG流程失败: {e}")

        # 构造发送给AI的消息列表
        messages = []
        if request.history:
            messages.extend([{"role": h.role, "content": h.content} for h in request.history])
        messages.append({"role": "user", "content": final_question})

        data = {
            "model": "moonshotai/kimi-k2:free",
            "messages": messages
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(self.api_url, json=data, headers=headers)
                response.raise_for_status()
                
                result = response.json()
                
                if not result.get("choices") or not result["choices"][0].get("message"):
                    raise ValueError("从AI API收到了无效的回复")

                answer = result["choices"][0]["message"]["content"]
                return AIQuestionResponse(answer=answer)

            except httpx.HTTPStatusError as e:
                # 更详细的日志记录
                print(f"请求AI API时发生HTTP错误: {e.response.status_code} - {e.response.text}")
                raise Exception("AI服务请求失败")
            except Exception as e:
                print(f"调用AI API时发生未知错误: {e}")
                raise Exception("AI服务暂时不可用")
    
    def _generate_specific_answer(self, question: str, base_answer: str) -> str:
        """生成具体的回答"""
        # 这里可以添加更复杂的逻辑来生成个性化回答
        # 实际项目中应该调用真实的AI模型
        
        additional_info = ""
        
        if "机器学习" in question:
            additional_info = "\n\n关于机器学习课程，建议包含以下模块：\n1. 机器学习基础理论\n2. 监督学习算法\n3. 无监督学习方法\n4. 深度学习入门\n5. 实践项目案例"
        elif "数学" in question:
            additional_info = "\n\n数学课程设计要点：\n1. 循序渐进的知识体系\n2. 理论与应用相结合\n3. 充分的练习和反馈\n4. 可视化教学工具的运用"
        elif "英语" in question:
            additional_info = "\n\n英语教学建议：\n1. 听说读写全面发展\n2. 情境化教学方法\n3. 互动式课堂活动\n4. 多媒体资源整合"
        
        return base_answer + additional_info
    
    def get_quick_suggestions(self) -> List[str]:
        """获取快速建议"""
        return [
            "如何设计一个有效的课程大纲？",
            "制作教学PPT的最佳实践是什么？",
            "怎样构建学科知识图谱？",
            "如何建立公平的作业评价体系？",
            "什么是翻转课堂教学模式？"
        ]


# 创建全局服务实例
ai_service = AIService()
