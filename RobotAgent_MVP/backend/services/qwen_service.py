import httpx
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from utils.config import Config
from utils.logger import CustomLogger

class QwenService:
    """Qwen模型服务类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = CustomLogger("QwenService")
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.qwen.timeout,
                read=self.config.qwen.timeout,
                write=self.config.qwen.timeout,
                pool=self.config.qwen.timeout
            )
        )
        self.api_key = self.config.qwen.api_key
        self.base_url = self.config.qwen.base_url
        self.model = self.config.qwen.model
        
        # 统计信息
        self.total_calls = 0
        self.total_tokens = 0
        self.error_count = 0
        
    async def parse_natural_language(self, text: str, user_id: str = "default") -> Dict[str, Any]:
        """
        使用Qwen模型解析自然语言输入
        
        Args:
            text: 用户输入的自然语言文本
            user_id: 用户ID
            
        Returns:
            解析结果字典
        """
        start_time = datetime.now()
        
        try:
            # 构建提示词
            prompt = self._build_prompt(text)
            
            # 调用Qwen API
            response = await self._call_qwen_api(prompt)
            
            # 解析响应
            parsed_result = self._parse_response(response)
            
            # 计算延迟
            latency = (datetime.now() - start_time).total_seconds()
            
            # 记录统计信息
            self.total_calls += 1
            
            # 记录日志
            self.logger.info(
                f"成功解析自然语言输入",
                extra={
                    "user_id": user_id,
                    "input_text": text[:100] + "..." if len(text) > 100 else text,
                    "latency": latency,
                    "intent": parsed_result.get("intent"),
                    "action": parsed_result.get("action")
                }
            )
            
            return {
                "success": True,
                "result": parsed_result,
                "latency": latency,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            self.error_count += 1
            latency = (datetime.now() - start_time).total_seconds()
            
            self.logger.error(
                f"解析自然语言输入失败: {str(e)}",
                extra={
                    "user_id": user_id,
                    "input_text": text[:100] + "..." if len(text) > 100 else text,
                    "latency": latency,
                    "error": str(e)
                }
            )
            
            return {
                "success": False,
                "error": str(e),
                "latency": latency,
                "timestamp": start_time.isoformat()
            }
    
    def _build_prompt(self, text: str) -> str:
        """构建Qwen模型的提示词"""
        prompt = f"""
你是一个专业的机器人控制指令解析器。请将用户的自然语言输入解析为标准的JSON格式。

解析规则：
1. 识别用户的意图（intent）：移动、抓取、放置、停止、查询状态等
2. 提取具体的动作（action）：move_to_position、pick_object、place_object、stop_motion、get_status等
3. 提取动作参数（parameters）：位置坐标、物体名称、速度等

输出格式（必须是有效的JSON）：
{{
    "intent": "用户意图的中文描述",
    "action": "具体的动作类型",
    "parameters": {{
        "position": {{"x": 0.0, "y": 0.0, "z": 0.0}},
        "orientation": {{"roll": 0.0, "pitch": 0.0, "yaw": 0.0}},
        "speed": 0.5,
        "object_name": "物体名称",
        "gripper_action": "open/close"
    }},
    "priority": "high/medium/low",
    "estimated_duration": 5.0
}}

用户输入：{text}

请解析上述输入并返回JSON格式的结果：
"""
        return prompt
    
    async def _call_qwen_api(self, prompt: str) -> Dict[str, Any]:
        """调用Qwen API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.config.qwen.temperature,
            "max_tokens": self.config.qwen.max_tokens,
            "top_p": self.config.qwen.top_p
        }
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"Qwen API调用失败: {response.status_code} - {response.text}")
        
        return response.json()
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """解析Qwen API响应"""
        try:
            # 提取生成的文本
            content = response["choices"][0]["message"]["content"]
            
            # 尝试解析JSON
            # 清理可能的markdown格式
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # 解析JSON
            parsed_json = json.loads(content)
            
            # 验证必要字段
            required_fields = ["intent", "action", "parameters"]
            for field in required_fields:
                if field not in parsed_json:
                    raise ValueError(f"缺少必要字段: {field}")
            
            # 设置默认值
            if "priority" not in parsed_json:
                parsed_json["priority"] = "medium"
            if "estimated_duration" not in parsed_json:
                parsed_json["estimated_duration"] = 5.0
            
            # 记录token使用情况
            if "usage" in response:
                self.total_tokens += response["usage"].get("total_tokens", 0)
            
            return parsed_json
            
        except json.JSONDecodeError as e:
            raise Exception(f"JSON解析失败: {str(e)}, 原始内容: {content}")
        except Exception as e:
            raise Exception(f"响应解析失败: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 简单的API调用测试
            test_prompt = "测试连接"
            start_time = datetime.now()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 10
            }
            
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            latency = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "latency": latency,
                "status_code": response.status_code,
                "total_calls": self.total_calls,
                "total_tokens": self.total_tokens,
                "error_count": self.error_count
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "total_calls": self.total_calls,
                "total_tokens": self.total_tokens,
                "error_count": self.error_count
            }
    
    async def close(self):
        """关闭HTTP客户端"""
        await self.client.aclose()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "success_rate": (self.total_calls - self.error_count) / max(self.total_calls, 1) * 100
        }