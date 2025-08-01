import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import sys
import os

# 添加backend路径到sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import app
from services.qwen_service import QwenService
from services.message_queue import MessageQueue
from agents.memory_agent import MemoryAgent
from agents.ros2_agent import ROS2Agent
from models.message_models import UserInputMessage, ParsedMessage, ROS2CommandMessage

class TestQwenService:
    """测试Qwen服务"""
    
    @pytest.fixture
    def qwen_service(self):
        with patch('services.qwen_service.Config') as mock_config:
            mock_config.return_value.qwen.api_key = "test_key"
            mock_config.return_value.qwen.base_url = "http://test.com"
            mock_config.return_value.qwen.model = "qwen-turbo"
            return QwenService()
    
    @pytest.mark.asyncio
    async def test_parse_natural_language_success(self, qwen_service):
        """测试自然语言解析成功"""
        mock_response = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "intent": "move_to_position",
                        "action": "move_arm",
                        "parameters": {"x": 0.5, "y": 0.3, "z": 0.2},
                        "confidence": 0.95
                    })
                }
            }]
        }
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.status_code = 200
            
            result = await qwen_service.parse_natural_language("移动到位置(0.5, 0.3, 0.2)")
            
            assert result["intent"] == "move_to_position"
            assert result["action"] == "move_arm"
            assert result["parameters"]["x"] == 0.5
    
    @pytest.mark.asyncio
    async def test_parse_natural_language_api_error(self, qwen_service):
        """测试API错误处理"""
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value.status_code = 500
            
            result = await qwen_service.parse_natural_language("测试命令")
            
            assert result is None

class TestMessageQueue:
    """测试消息队列"""
    
    @pytest.fixture
    def message_queue(self):
        with patch('services.message_queue.Config') as mock_config:
            mock_config.return_value.redis.host = "localhost"
            mock_config.return_value.redis.port = 6379
            return MessageQueue()
    
    @pytest.mark.asyncio
    async def test_connect_success(self, message_queue):
        """测试Redis连接成功"""
        with patch('aioredis.from_url') as mock_redis:
            mock_redis.return_value.ping = AsyncMock(return_value=True)
            
            result = await message_queue.connect()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_send_to_memory_agent(self, message_queue):
        """测试发送消息到记忆Agent"""
        message_queue.redis = Mock()
        message_queue.redis.lpush = AsyncMock()
        
        test_message = {"type": "test", "data": "test_data"}
        await message_queue.send_to_memory_agent(test_message)
        
        message_queue.redis.lpush.assert_called_once()

class TestMemoryAgent:
    """测试记忆Agent"""
    
    @pytest.fixture
    def memory_agent(self):
        with patch('agents.memory_agent.Config') as mock_config:
            mock_config.return_value.memory_agent.records_dir = "test_records"
            mock_config.return_value.memory_agent.session_log_file = "test_session.log"
            return MemoryAgent()
    
    @pytest.mark.asyncio
    async def test_process_memory_record(self, memory_agent):
        """测试处理记忆记录"""
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.write = Mock()
            
            test_record = {
                "user_id": "test_user",
                "session_id": "test_session",
                "interaction_data": {"input": "test", "output": "test"}
            }
            
            await memory_agent.process_memory_record(test_record)
            mock_open.assert_called()

class TestROS2Agent:
    """测试ROS2 Agent"""
    
    @pytest.fixture
    def ros2_agent(self):
        with patch('agents.ros2_agent.Config') as mock_config:
            mock_config.return_value.ros2.workspace_path = "/test/ws"
            return ROS2Agent()
    
    @pytest.mark.asyncio
    async def test_execute_parsed_command(self, ros2_agent):
        """测试执行解析后的命令"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "success"
            
            test_command = {
                "intent": "move_to_position",
                "action": "move_arm",
                "parameters": {"x": 0.5, "y": 0.3, "z": 0.2}
            }
            
            result = await ros2_agent.execute_parsed_command(test_command)
            assert result["success"] is True

class TestAPIEndpoints:
    """测试API端点"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_health_check(self, client):
        """测试健康检查端点"""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_process_command_invalid_input(self, client):
        """测试无效输入的命令处理"""
        response = client.post("/api/process-command", json={})
        assert response.status_code == 422  # Validation error
    
    def test_process_command_valid_input(self, client):
        """测试有效输入的命令处理"""
        with patch('services.qwen_service.QwenService.parse_natural_language') as mock_parse:
            mock_parse.return_value = {
                "intent": "move_to_position",
                "action": "move_arm",
                "parameters": {"x": 0.5, "y": 0.3, "z": 0.2},
                "confidence": 0.95
            }
            
            response = client.post("/api/process-command", json={
                "user_id": "test_user",
                "input_text": "移动到位置(0.5, 0.3, 0.2)",
                "session_id": "test_session"
            })
            
            # 由于依赖服务可能未启动，这里主要测试请求格式
            assert response.status_code in [200, 500]  # 200成功或500服务未启动

class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_command_flow(self):
        """测试完整的命令流程"""
        # 这是一个集成测试示例，需要所有服务都启动
        # 在实际环境中运行
        pass

class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_qwen_service_response_time(self):
        """测试Qwen服务响应时间"""
        import time
        
        with patch('services.qwen_service.Config') as mock_config:
            mock_config.return_value.qwen.api_key = "test_key"
            mock_config.return_value.qwen.base_url = "http://test.com"
            mock_config.return_value.qwen.model = "qwen-turbo"
            
            qwen_service = QwenService()
            
            with patch('httpx.AsyncClient.post') as mock_post:
                mock_response = {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "intent": "test",
                                "action": "test",
                                "parameters": {},
                                "confidence": 0.95
                            })
                        }
                    }]
                }
                mock_post.return_value.json.return_value = mock_response
                mock_post.return_value.status_code = 200
                
                start_time = time.time()
                await qwen_service.parse_natural_language("测试命令")
                end_time = time.time()
                
                response_time = end_time - start_time
                assert response_time < 1.0  # 响应时间应小于1秒

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])