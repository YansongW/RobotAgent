"""
豆包大模型服务 - OpenAI兼容实现
实现与豆包API的交互，支持对话、嵌入等功能，完全兼容OpenAI API格式
"""

import json
import logging
import asyncio
import os
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)

class DoubaoOpenAIClient:
    """OpenAI兼容的豆包客户端"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        初始化OpenAI兼容客户端
        
        Args:
            api_key: API密钥，可从环境变量ARK_API_KEY获取
            base_url: API基础URL
        """
        self.api_key = api_key or os.environ.get("ARK_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("API key is required. Set ARK_API_KEY environment variable or pass api_key parameter.")
        
        # 初始化聊天和嵌入客户端
        self.chat = ChatCompletions(self.api_key, self.base_url)
        self.embeddings = Embeddings(self.api_key, self.base_url)


class ChatCompletions:
    """聊天完成API"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.endpoint = "/chat/completions"
    
    async def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        stream: bool = False,
        tools: List[Dict] = None,
        **kwargs
    ) -> Union[Dict, AsyncGenerator]:
        """
        创建聊天完成
        
        Args:
            model: 模型名称或推理接入点ID
            messages: 消息列表
            max_tokens: 最大token数
            temperature: 温度参数
            stream: 是否流式响应
            tools: 工具列表（Function Calling）
            **kwargs: 其他参数
            
        Returns:
            聊天响应或流式生成器
        """
        url = f"{self.base_url}{self.endpoint}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        
        # 添加工具调用支持
        if tools:
            payload["tools"] = tools
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        if stream:
                            return self._handle_stream_response(response)
                        else:
                            result = await response.json()
                            return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Chat API调用失败: {response.status}, {error_text}")
                        raise Exception(f"API call failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Chat API请求异常: {e}")
            raise
    
    async def _handle_stream_response(self, response) -> AsyncGenerator[Dict, None]:
        """处理流式响应"""
        try:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if not line:
                    continue
                    
                if line.startswith('data: '):
                    data = line[6:]  # 移除 'data: ' 前缀
                    
                    if data == '[DONE]':
                        break
                        
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"处理流式响应失败: {e}")
            raise


class Embeddings:
    """嵌入向量API"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.endpoint = "/embeddings"
    
    async def create(
        self,
        model: str,
        input: Union[str, List[str]],
        **kwargs
    ) -> Dict:
        """
        创建嵌入向量
        
        Args:
            model: 模型名称
            input: 输入文本或文本列表
            **kwargs: 其他参数
            
        Returns:
            嵌入响应
        """
        url = f"{self.base_url}{self.endpoint}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "input": input,
            **kwargs
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Embeddings API调用失败: {response.status}, {error_text}")
                        raise Exception(f"API call failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Embeddings API请求异常: {e}")
            raise


class DoubaoService:
    """豆包大模型服务类 - 保持向后兼容"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化豆包服务
        
        Args:
            config: 豆包API配置
        """
        self.config = config
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        self.models = config["models"]
        self.endpoints = config["endpoints"]
        
        # 初始化OpenAI兼容客户端
        self.client = DoubaoOpenAIClient(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 加载prompt模板
        self.prompts = self._load_prompts()
        
        logger.info("豆包服务初始化完成")
    
    def _load_prompts(self) -> Dict[str, Any]:
        """加载prompt模板"""
        try:
            prompts_dir = Path(__file__).parent.parent / "prompts"
            prompt_file = prompts_dir / "chat_prompts.json"
            
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 返回默认prompt
                return {
                    "system_prompt": "你是一个智能机器人助手，能够理解用户的自然语言指令并生成相应的ROS2控制命令。",
                    "chat_template": {
                        "type": "robot_response",
                        "response": "对用户的自然语言回复",
                        "ros2_command": {
                            "action": "动作类型",
                            "parameters": {},
                            "description": "动作描述"
                        }
                    }
                }
        except Exception as e:
            logger.error(f"加载prompt失败: {e}")
            return {}
    
    async def chat(
        self, 
        message: str, 
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        与豆包模型进行对话
        
        Args:
            message: 用户消息
            conversation_history: 对话历史
            
        Returns:
            包含响应和ROS2命令的字典
        """
        try:
            # 构建消息列表
            messages = []
            
            # 添加系统prompt
            if self.prompts.get("system_prompt"):
                messages.append({
                    "role": "system",
                    "content": self.prompts["system_prompt"]
                })
            
            # 添加对话历史
            if conversation_history:
                messages.extend(conversation_history[-10:])  # 只保留最近10轮对话
            
            # 添加当前用户消息
            messages.append({
                "role": "user",
                "content": message
            })
            
            # 调用豆包API
            response = await self._call_chat_api(messages)
            
            if response:
                # 解析响应
                parsed_response = self._parse_chat_response(response)
                return parsed_response
            else:
                return {
                    "response": "抱歉，我现在无法处理您的请求。",
                    "ros2_command": None
                }
                
        except Exception as e:
            logger.error(f"豆包对话失败: {e}")
            return {
                "response": "抱歉，服务出现了问题。",
                "ros2_command": None
            }
    
    async def _call_chat_api(self, messages: List[Dict], stream: bool = False) -> Optional[Dict]:
        """调用豆包聊天API - 兼容OpenAI API格式"""
        url = f"{self.base_url}{self.endpoints['chat']}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 基础payload - 兼容OpenAI API格式
        payload = {
            "model": self.models["chat"],
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.7,
            "stream": stream
        }
        
        # 只在非流式模式下使用结构化输出
        if not stream:
            # 使用火山引擎支持的结构化输出格式
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "robot_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "response": {
                                "type": "string",
                                "description": "对用户的自然语言回复"
                            },
                            "ros2_command": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string"},
                                    "parameters": {"type": "object"},
                                    "description": {"type": "string"}
                                },
                                "required": ["action", "parameters", "description"]
                            }
                        },
                        "required": ["response"],
                        "additionalProperties": false
                    },
                    "strict": True
                }
            }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        if stream:
                            return response  # 返回响应对象用于流式处理
                        else:
                            result = await response.json()
                            return result
                    else:
                        try:
                            error_json = await response.json()
                            logger.error(f"豆包API调用失败: {response.status}, {error_json}")
                        except:
                            error_text = await response.text()
                            logger.error(f"豆包API调用失败: {response.status}, {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"豆包API请求异常: {e}")
            return None
    
    async def _handle_stream_response(self, response):
        """处理流式响应"""
        try:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if not line:
                    continue
                    
                if line.startswith('data: '):
                    data = line[6:]  # 移除 'data: ' 前缀
                    
                    if data == '[DONE]':
                        break
                        
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"处理流式响应失败: {e}")
            yield f"流式处理错误: {e}"
    
    def _parse_chat_response(self, api_response: Dict) -> Dict[str, Any]:
        """解析豆包API响应"""
        try:
            if "choices" in api_response and api_response["choices"]:
                content = api_response["choices"][0]["message"]["content"]
                
                # 由于使用了结构化输出，content应该直接是JSON格式
                try:
                    # 直接解析JSON响应
                    parsed = json.loads(content)
                    
                    return {
                        "response": parsed.get("response", "抱歉，我无法理解您的请求。"),
                        "ros2_command": parsed.get("ros2_command")
                    }
                    
                except json.JSONDecodeError:
                    # 如果解析失败，可能是非结构化响应，尝试其他解析方式
                    logger.warning(f"JSON解析失败，尝试其他方式: {content}")
                    
                    if "```json" in content:
                        json_start = content.find("```json") + 7
                        json_end = content.find("```", json_start)
                        json_content = content[json_start:json_end].strip()
                        try:
                            parsed = json.loads(json_content)
                            return {
                                "response": parsed.get("response", content),
                                "ros2_command": parsed.get("ros2_command")
                            }
                        except json.JSONDecodeError:
                            pass
                    
                    # 如果所有解析都失败，返回原始内容
                    return {
                        "response": content,
                        "ros2_command": None
                    }
            else:
                return {
                    "response": "抱歉，我无法理解您的请求。",
                    "ros2_command": None
                }
                
        except Exception as e:
            logger.error(f"解析豆包响应失败: {e}")
            return {
                "response": "响应解析失败。",
                "ros2_command": None
            }
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本嵌入向量"""
        url = f"{self.base_url}{self.endpoints['embeddings']}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.models["embedding_text"],
            "input": text
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "data" in result and result["data"]:
                            return result["data"][0]["embedding"]
                    else:
                        error_text = await response.text()
                        logger.error(f"嵌入API调用失败: {response.status}, {error_text}")
                        
        except Exception as e:
            logger.error(f"获取嵌入向量失败: {e}")
            
        return None
    
    async def get_multimodal_embedding(
        self, 
        text: str = None, 
        image_url: str = None
    ) -> Optional[List[float]]:
        """获取多模态嵌入向量"""
        url = f"{self.base_url}{self.endpoints['embeddings_multimodal']}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        input_data = []
        if text:
            input_data.append({"type": "text", "text": text})
        if image_url:
            input_data.append({"type": "image_url", "image_url": {"url": image_url}})
        
        payload = {
            "model": self.models["embedding_vision"],
            "input": input_data
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "data" in result and result["data"]:
                            return result["data"][0]["embedding"]
                    else:
                        error_text = await response.text()
                        logger.error(f"多模态嵌入API调用失败: {response.status}, {error_text}")
                        
        except Exception as e:
            logger.error(f"获取多模态嵌入向量失败: {e}")
            
        return None
    
    async def chat_completion(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """
        兼容性接口：从消息列表中提取用户消息和对话历史
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
            stream: 是否流式响应
            **kwargs: 其他参数
            
        Returns:
            聊天响应或流式生成器
        """
        try:
            # 提取最后一条用户消息
            user_message = None
            conversation_history = []
            
            for msg in messages:
                if msg["role"] == "user":
                    user_message = msg["content"]
                elif msg["role"] == "assistant":
                    conversation_history.append({"role": "assistant", "content": msg["content"]})
            
            if not user_message:
                raise ValueError("No user message found in messages")
            
            # 调用现有的chat方法
            return await self.chat(user_message, conversation_history, stream=stream)
            
        except Exception as e:
            logger.error(f"Chat completion失败: {e}")
            raise
    
    # OpenAI兼容的新方法
    async def openai_chat_completion(
        self,
        model: str = None,
        messages: List[Dict[str, str]] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        stream: bool = False,
        tools: List[Dict] = None,
        **kwargs
    ) -> Union[Dict, AsyncGenerator]:
        """
        OpenAI兼容的聊天完成接口
        
        Args:
            model: 模型名称，默认使用配置中的聊天模型
            messages: 消息列表
            max_tokens: 最大token数
            temperature: 温度参数
            stream: 是否流式响应
            tools: 工具列表（Function Calling）
            **kwargs: 其他参数
            
        Returns:
            OpenAI格式的聊天响应或流式生成器
        """
        if not model:
            model = self.models.get("chat", "doubao-pro-4k")
        
        if not messages:
            raise ValueError("Messages are required")
        
        try:
            return await self.client.chat.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                tools=tools,
                **kwargs
            )
        except Exception as e:
            logger.error(f"OpenAI聊天完成失败: {e}")
            raise
    
    async def openai_embeddings(
        self,
        model: str = None,
        input: Union[str, List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        OpenAI兼容的嵌入向量接口
        
        Args:
            model: 模型名称，默认使用配置中的文本嵌入模型
            input: 输入文本或文本列表
            **kwargs: 其他参数
            
        Returns:
            OpenAI格式的嵌入响应
        """
        if not model:
            model = self.models.get("text_embedding", "doubao-embedding")
        
        if not input:
            raise ValueError("Input is required")
        
        try:
            return await self.client.embeddings.create(
                model=model,
                input=input,
                **kwargs
            )
        except Exception as e:
            logger.error(f"OpenAI嵌入向量失败: {e}")
            raise
    
    # 便捷方法
    async def single_turn_chat(self, user_message: str, system_message: str = None, **kwargs) -> str:
        """
        单轮对话
        
        Args:
            user_message: 用户消息
            system_message: 系统消息，默认使用配置中的系统prompt
            **kwargs: 其他参数
            
        Returns:
            助手回复
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        elif self.prompts.get("system_prompt"):
            messages.append({"role": "system", "content": self.prompts["system_prompt"]})
        
        messages.append({"role": "user", "content": user_message})
        
        response = await self.openai_chat_completion(messages=messages, **kwargs)
        
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["message"]["content"]
        else:
            raise ValueError("Unexpected response format")
    
    async def multi_turn_chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        多轮对话
        
        Args:
            messages: 消息历史列表
            **kwargs: 其他参数
            
        Returns:
            助手回复
        """
        response = await self.openai_chat_completion(messages=messages, **kwargs)
        
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["message"]["content"]
        else:
            raise ValueError("Unexpected response format")
    
    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """
        流式聊天
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Yields:
            流式文本片段
        """
        stream = await self.openai_chat_completion(messages=messages, stream=True, **kwargs)
        
        async for chunk in stream:
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
    
    async def function_call_chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        **kwargs
    ) -> Dict:
        """
        工具调用聊天
        
        Args:
            messages: 消息列表
            tools: 工具定义列表
            **kwargs: 其他参数
            
        Returns:
            包含工具调用的完整响应
        """
        response = await self.openai_chat_completion(
            messages=messages,
            tools=tools,
            **kwargs
        )
        
        return response