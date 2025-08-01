import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from utils.config import Config
from utils.logger import CustomLogger
from services.message_queue import MessageQueue
from models.message_models import QueueMessage, MessageType, MemoryRecordMessage

class MemoryAgent:
    """记忆Agent - 负责记录和管理交互记忆"""
    
    def __init__(self, config: Config, message_queue: MessageQueue):
        self.config = config
        self.message_queue = message_queue
        self.logger = CustomLogger("MemoryAgent")
        
        # 记忆存储路径
        self.memory_dir = Path(self.config.memory_agent["storage_path"])
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # 运行状态
        self.is_running = False
        self.processed_count = 0
        
    async def start(self):
        """启动记忆Agent"""
        self.is_running = True
        self.logger.info("记忆Agent已启动")
        
        # 启动消息处理循环
        asyncio.create_task(self._message_processing_loop())
    
    async def stop(self):
        """停止记忆Agent"""
        self.is_running = False
        self.logger.info("记忆Agent已停止")
    
    async def _message_processing_loop(self):
        """消息处理循环"""
        while self.is_running:
            try:
                # 从队列接收消息
                message = await self.message_queue.receive_from_queue(
                    self.message_queue.memory_queue,
                    timeout=5
                )
                
                if message:
                    await self._process_message(message)
                    
            except Exception as e:
                self.logger.error(f"消息处理循环出错: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: QueueMessage):
        """处理接收到的消息"""
        try:
            self.logger.info(
                f"处理记忆消息",
                extra={
                    "message_id": message.message_id,
                    "message_type": message.message_type.value
                }
            )
            
            if message.message_type == MessageType.MEMORY_RECORD:
                await self._save_memory_record(message)
            elif message.message_type == MessageType.USER_INPUT:
                await self._record_user_interaction(message)
            elif message.message_type == MessageType.PARSED_COMMAND:
                await self._record_parsed_command(message)
            elif message.message_type == MessageType.ROS2_COMMAND:
                await self._record_ros2_command(message)
            else:
                self.logger.warning(f"未知的消息类型: {message.message_type}")
            
            self.processed_count += 1
            
        except Exception as e:
            self.logger.error(
                f"处理消息失败: {str(e)}",
                extra={
                    "message_id": message.message_id,
                    "error": str(e)
                }
            )
    
    async def _save_memory_record(self, message: QueueMessage):
        """保存记忆记录"""
        try:
            memory_data = message.data
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = memory_data.get("session_id", "default")
            filename = f"memory_{session_id}_{timestamp}.md"
            filepath = self.memory_dir / filename
            
            # 生成Markdown内容
            markdown_content = self._generate_markdown_content(memory_data)
            
            # 写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            self.logger.info(
                f"记忆记录已保存",
                extra={
                    "filename": filename,
                    "session_id": session_id,
                    "file_size": len(markdown_content)
                }
            )
            
        except Exception as e:
            self.logger.error(f"保存记忆记录失败: {str(e)}")
    
    async def _record_user_interaction(self, message: QueueMessage):
        """记录用户交互"""
        try:
            interaction_data = {
                "type": "user_interaction",
                "timestamp": message.timestamp,
                "message_id": message.message_id,
                "user_input": message.data.get("input_text", ""),
                "user_id": message.data.get("user_id", ""),
                "session_id": message.data.get("session_id", "")
            }
            
            await self._append_to_session_log(interaction_data)
            
        except Exception as e:
            self.logger.error(f"记录用户交互失败: {str(e)}")
    
    async def _record_parsed_command(self, message: QueueMessage):
        """记录解析后的命令"""
        try:
            command_data = {
                "type": "parsed_command",
                "timestamp": message.timestamp,
                "message_id": message.message_id,
                "intent": message.data.get("intent", ""),
                "action": message.data.get("action", ""),
                "parameters": message.data.get("parameters", {}),
                "priority": message.priority.value
            }
            
            await self._append_to_session_log(command_data)
            
        except Exception as e:
            self.logger.error(f"记录解析命令失败: {str(e)}")
    
    async def _record_ros2_command(self, message: QueueMessage):
        """记录ROS2命令"""
        try:
            ros2_data = {
                "type": "ros2_command",
                "timestamp": message.timestamp,
                "message_id": message.message_id,
                "command_type": message.data.get("command_type", ""),
                "topic": message.data.get("topic", ""),
                "parameters": message.data.get("parameters", {}),
                "execution_status": message.data.get("status", "")
            }
            
            await self._append_to_session_log(ros2_data)
            
        except Exception as e:
            self.logger.error(f"记录ROS2命令失败: {str(e)}")
    
    async def _append_to_session_log(self, data: Dict[str, Any]):
        """追加到会话日志"""
        try:
            session_id = data.get("session_id", "default")
            today = datetime.now().strftime("%Y%m%d")
            log_filename = f"session_{session_id}_{today}.md"
            log_filepath = self.memory_dir / log_filename
            
            # 生成日志条目
            log_entry = self._format_log_entry(data)
            
            # 追加到文件
            with open(log_filepath, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n\n")
            
        except Exception as e:
            self.logger.error(f"追加会话日志失败: {str(e)}")
    
    def _generate_markdown_content(self, memory_data: Dict[str, Any]) -> str:
        """生成Markdown格式的记忆内容"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"""# 机器人交互记忆记录

## 基本信息
- **记录时间**: {timestamp}
- **会话ID**: {memory_data.get('session_id', 'N/A')}
- **用户ID**: {memory_data.get('user_id', 'N/A')}
- **消息ID**: {memory_data.get('message_id', 'N/A')}

## 交互内容

### 用户输入
```
{memory_data.get('user_input', 'N/A')}
```

### 解析结果
- **意图**: {memory_data.get('intent', 'N/A')}
- **动作**: {memory_data.get('action', 'N/A')}
- **优先级**: {memory_data.get('priority', 'N/A')}

### 动作参数
```json
{json.dumps(memory_data.get('parameters', {}), indent=2, ensure_ascii=False)}
```

### ROS2命令
- **命令类型**: {memory_data.get('ros2_command_type', 'N/A')}
- **话题**: {memory_data.get('ros2_topic', 'N/A')}
- **执行状态**: {memory_data.get('execution_status', 'N/A')}

### 性能指标
- **解析延迟**: {memory_data.get('parsing_latency', 'N/A')}秒
- **执行时间**: {memory_data.get('execution_time', 'N/A')}秒
- **总处理时间**: {memory_data.get('total_time', 'N/A')}秒

### 错误信息
{memory_data.get('error_message', '无错误')}

---
*此记录由RobotAgent记忆系统自动生成*
"""
        return content
    
    def _format_log_entry(self, data: Dict[str, Any]) -> str:
        """格式化日志条目"""
        timestamp = data.get("timestamp", datetime.now().isoformat())
        entry_type = data.get("type", "unknown")
        
        if entry_type == "user_interaction":
            return f"""## 用户交互 - {timestamp}
- **消息ID**: {data.get('message_id', 'N/A')}
- **用户输入**: {data.get('user_input', 'N/A')}
- **用户ID**: {data.get('user_id', 'N/A')}"""
        
        elif entry_type == "parsed_command":
            return f"""## 命令解析 - {timestamp}
- **消息ID**: {data.get('message_id', 'N/A')}
- **意图**: {data.get('intent', 'N/A')}
- **动作**: {data.get('action', 'N/A')}
- **优先级**: {data.get('priority', 'N/A')}
- **参数**: {json.dumps(data.get('parameters', {}), ensure_ascii=False)}"""
        
        elif entry_type == "ros2_command":
            return f"""## ROS2命令 - {timestamp}
- **消息ID**: {data.get('message_id', 'N/A')}
- **命令类型**: {data.get('command_type', 'N/A')}
- **话题**: {data.get('topic', 'N/A')}
- **执行状态**: {data.get('execution_status', 'N/A')}
- **参数**: {json.dumps(data.get('parameters', {}), ensure_ascii=False)}"""
        
        else:
            return f"""## {entry_type} - {timestamp}
- **消息ID**: {data.get('message_id', 'N/A')}
- **数据**: {json.dumps(data, ensure_ascii=False, indent=2)}"""
    
    async def get_memory_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取记忆记录列表"""
        try:
            records = []
            
            # 遍历记忆目录
            for filepath in self.memory_dir.glob("*.md"):
                stat = filepath.stat()
                
                # 计算交互次数（简单统计）
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    interaction_count = content.count("## 用户交互")
                
                records.append({
                    "filename": filepath.name,
                    "created_time": datetime.fromtimestamp(stat.st_ctime),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime),
                    "size": stat.st_size,
                    "interaction_count": interaction_count
                })
            
            # 按修改时间排序
            records.sort(key=lambda x: x["modified_time"], reverse=True)
            
            return records[:limit]
            
        except Exception as e:
            self.logger.error(f"获取记忆记录失败: {str(e)}")
            return []
    
    async def get_session_log(self, session_id: str, date: str = None) -> Optional[str]:
        """获取会话日志内容"""
        try:
            if not date:
                date = datetime.now().strftime("%Y%m%d")
            
            log_filename = f"session_{session_id}_{date}.md"
            log_filepath = self.memory_dir / log_filename
            
            if log_filepath.exists():
                with open(log_filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取会话日志失败: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查存储目录
            storage_accessible = self.memory_dir.exists() and os.access(self.memory_dir, os.W_OK)
            
            # 统计记录数量
            total_records = len(list(self.memory_dir.glob("*.md")))
            
            # 计算存储大小
            total_size = sum(f.stat().st_size for f in self.memory_dir.glob("*.md"))
            
            return {
                "status": "healthy" if (self.is_running and storage_accessible) else "unhealthy",
                "is_running": self.is_running,
                "storage_accessible": storage_accessible,
                "storage_path": str(self.memory_dir),
                "total_records": total_records,
                "total_size_bytes": total_size,
                "processed_count": self.processed_count
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "is_running": self.is_running
            }