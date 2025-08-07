"""
TTS语音合成服务
基于豆包双向流式TTS WebSocket API实现文字转语音功能
"""

import logging
import asyncio
import base64
import json
import os
import uuid
import struct
import websockets
from typing import Optional
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)

class TTSService:
    """TTS语音合成服务类 - 使用双向流式WebSocket API"""
    
    def __init__(self, config: dict):
        """
        初始化TTS服务
        
        Args:
            config: TTS配置
        """
        self.config = config
        self.app_id = config["app_id"]
        self.access_token = config["access_token"]
        self.secret_key = config["secret_key"]
        self.base_url = config["base_url"]
        self.resource_id = config["resource_id"]
        self.voice_type = config.get("voice_type", "zh_female_linjianvhai_moon_bigtts")
        self.model = config.get("model", "seed-tts-1.1")
        self.format = config.get("format", "pcm")
        self.sample_rate = config.get("sample_rate", 24000)
        self.channel = config.get("channel", 1)
        
        # 确保音频输出目录存在
        self.audio_dir = Path(__file__).parent.parent / "static" / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("TTS服务初始化完成 - 使用双向流式WebSocket API")
    
    async def text_to_speech(self, text: str) -> Optional[str]:
        """
        将文字转换为语音
        
        Args:
            text: 要转换的文字
            
        Returns:
            生成的音频文件名，失败返回None
        """
        try:
            if not text or not text.strip():
                logger.warning("输入文本为空")
                return None
            
            # 调用TTS API
            audio_data = await self._synthesize_speech(text)
            
            if audio_data:
                # 保存音频文件
                filename = await self._save_audio_file(audio_data)
                if filename:
                    logger.info(f"语音合成成功: {filename}")
                    return filename
            
            logger.warning("语音合成失败")
            return None
            
        except Exception as e:
            logger.error(f"语音合成异常: {e}")
            return None
    
    async def _synthesize_speech(self, text: str) -> Optional[bytes]:
        """
        使用WebSocket双向流式TTS API合成语音
        
        Args:
            text: 要合成的文字
            
        Returns:
            音频数据（字节流）
        """
        try:
            # WebSocket连接头
            headers = {
                "X-Api-App-Key": self.app_id,
                "X-Api-Access-Key": self.access_token,
                "X-Api-Resource-Id": self.resource_id,
                "X-Api-Connect-Id": str(uuid.uuid4())
            }
            
            # 连接WebSocket
            async with websockets.connect(self.base_url, extra_headers=headers) as websocket:
                logger.info("WebSocket连接已建立")
                
                # 发送StartConnection事件
                await self._send_start_connection(websocket)
                
                # 等待ConnectionStarted响应
                await self._wait_for_connection_started(websocket)
                
                # 发送StartSession事件
                session_id = str(uuid.uuid4())
                await self._send_start_session(websocket, session_id)
                
                # 等待SessionStarted响应
                await self._wait_for_session_started(websocket)
                
                # 发送文本请求
                await self._send_task_request(websocket, text, session_id)
                
                # 发送FinishSession
                await self._send_finish_session(websocket, session_id)
                
                # 接收音频数据
                audio_data = await self._receive_audio_data(websocket)
                
                # 发送FinishConnection
                await self._send_finish_connection(websocket)
                
                return audio_data
                
        except Exception as e:
            logger.error(f"WebSocket TTS API请求异常: {e}")
            return None
    
    def _create_binary_frame(self, message_type: int, flags: int, payload: bytes, event: str = None, session_id: str = None) -> bytes:
        """创建二进制协议帧"""
        # 协议版本(4bit) + Header大小(4bit) = 0x11 (版本1, 4字节header)
        header_byte0 = 0x11
        
        # 消息类型(4bit) + 特定标志(4bit)
        header_byte1 = (message_type << 4) | flags
        
        # 序列化方法(4bit) + 压缩方法(4bit) = 0x10 (JSON, 无压缩)
        header_byte2 = 0x10
        
        # 保留字段
        header_byte3 = 0x00
        
        # 构建header
        header = struct.pack('>BBBB', header_byte0, header_byte1, header_byte2, header_byte3)
        
        # 可选字段
        optional_fields = b''
        
        # 添加event字段
        if event:
            event_bytes = event.encode('utf-8')
            optional_fields += struct.pack('>I', len(event_bytes)) + event_bytes
        
        # 添加session_id字段
        if session_id:
            session_bytes = session_id.encode('utf-8')
            optional_fields += struct.pack('>I', len(session_bytes)) + session_bytes
        
        # Payload大小
        payload_size = struct.pack('>I', len(payload))
        
        return header + optional_fields + payload_size + payload
    
    async def _send_start_connection(self, websocket):
        """发送StartConnection事件"""
        payload = json.dumps({
            "event": "StartConnection"
        }).encode('utf-8')
        
        frame = self._create_binary_frame(0x1, 0x4, payload, "StartConnection")
        await websocket.send(frame)
        logger.debug("已发送StartConnection事件")
    
    async def _send_start_session(self, websocket, session_id: str):
        """发送StartSession事件"""
        payload = json.dumps({
            "event": "StartSession",
            "user": {
                "uid": "default_user"
            },
            "req_params": {
                "voice_type": self.voice_type,
                "model": self.model,
                "audio_config": {
                    "format": self.format,
                    "sample_rate": self.sample_rate,
                    "channel": self.channel
                },
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0
            }
        }).encode('utf-8')
        
        frame = self._create_binary_frame(0x1, 0x4, payload, "StartSession", session_id)
        await websocket.send(frame)
        logger.debug("已发送StartSession事件")
    
    async def _send_task_request(self, websocket, text: str, session_id: str):
        """发送TaskRequest事件"""
        payload = json.dumps({
            "event": "TaskRequest",
            "req_params": {
                "text": text,
                "text_type": "plain"
            }
        }).encode('utf-8')
        
        frame = self._create_binary_frame(0x1, 0x4, payload, "TaskRequest", session_id)
        await websocket.send(frame)
        logger.debug(f"已发送TaskRequest事件: {text}")
    
    async def _send_finish_session(self, websocket, session_id: str):
        """发送FinishSession事件"""
        payload = json.dumps({
            "event": "FinishSession"
        }).encode('utf-8')
        
        frame = self._create_binary_frame(0x1, 0x4, payload, "FinishSession", session_id)
        await websocket.send(frame)
        logger.debug("已发送FinishSession事件")
    
    async def _send_finish_connection(self, websocket):
        """发送FinishConnection事件"""
        payload = json.dumps({
            "event": "FinishConnection"
        }).encode('utf-8')
        
        frame = self._create_binary_frame(0x1, 0x4, payload, "FinishConnection")
        await websocket.send(frame)
        logger.debug("已发送FinishConnection事件")
    
    async def _wait_for_connection_started(self, websocket):
        """等待ConnectionStarted响应"""
        try:
            response = await websocket.recv()
            logger.debug("收到ConnectionStarted响应")
        except Exception as e:
            logger.error(f"等待ConnectionStarted失败: {e}")
            raise
    
    async def _wait_for_session_started(self, websocket):
        """等待SessionStarted响应"""
        try:
            response = await websocket.recv()
            logger.debug("收到SessionStarted响应")
        except Exception as e:
            logger.error(f"等待SessionStarted失败: {e}")
            raise
    
    async def _receive_audio_data(self, websocket) -> bytes:
        """接收音频数据"""
        audio_chunks = []
        
        try:
            while True:
                response = await websocket.recv()
                
                if isinstance(response, bytes):
                    # 解析二进制响应
                    if len(response) >= 4:
                        # 检查消息类型
                        header_byte1 = response[1]
                        message_type = (header_byte1 >> 4) & 0xF
                        
                        if message_type == 0xB:  # Audio-only response
                            # 提取音频数据 (跳过header和payload size)
                            audio_data = response[8:]  # 假设header+payload_size=8字节
                            if audio_data:
                                audio_chunks.append(audio_data)
                                logger.debug(f"收到音频数据块: {len(audio_data)} 字节")
                        elif message_type == 0x9:  # Full server response
                            # 可能包含SessionFinished等事件
                            logger.debug("收到服务器响应")
                            break
                else:
                    # JSON响应
                    try:
                        data = json.loads(response)
                        if data.get("event") == "SessionFinished":
                            logger.debug("收到SessionFinished事件")
                            break
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            logger.error(f"接收音频数据失败: {e}")
        
        if audio_chunks:
            total_audio = b''.join(audio_chunks)
            logger.info(f"音频合成完成，总大小: {len(total_audio)} 字节")
            return total_audio
        else:
            logger.warning("未收到音频数据")
            return None
    
    async def _save_audio_file(self, audio_data: bytes) -> Optional[str]:
        """
        保存音频文件
        
        Args:
            audio_data: 音频数据
            
        Returns:
            保存的文件名
        """
        try:
            # 生成唯一文件名
            filename = f"tts_{uuid.uuid4().hex[:8]}.{self.format}"
            file_path = self.audio_dir / filename
            
            # 写入文件
            with open(file_path, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"音频文件已保存: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"保存音频文件失败: {e}")
            return None
    
    async def get_available_voices(self) -> list:
        """
        获取可用的语音列表
        
        Returns:
            可用语音列表
        """
        url = f"{self.base_url}/voices"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("voices", [])
                    else:
                        logger.error(f"获取语音列表失败: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"获取语音列表异常: {e}")
            return []
    
    def get_default_voices(self) -> dict:
        """
        获取默认语音配置
        
        Returns:
            默认语音配置字典
        """
        return {
            "zh_female_tianmei": "天美（女声）",
            "zh_male_chunhou": "淳厚（男声）",
            "zh_female_wenwen": "文文（女声）",
            "zh_male_haoming": "浩明（男声）",
            "en_female_sara": "Sara（英文女声）",
            "en_male_john": "John（英文男声）"
        }
    
    async def cleanup_old_files(self, max_age_hours: int = 24):
        """
        清理旧的音频文件
        
        Args:
            max_age_hours: 文件最大保留时间（小时）
        """
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for file_path in self.audio_dir.glob("tts_*.wav"):
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    logger.info(f"已删除旧音频文件: {file_path.name}")
                    
        except Exception as e:
            logger.error(f"清理音频文件失败: {e}")
    
    def get_audio_info(self, filename: str) -> dict:
        """
        获取音频文件信息
        
        Args:
            filename: 音频文件名
            
        Returns:
            音频文件信息
        """
        try:
            file_path = self.audio_dir / filename
            if file_path.exists():
                stat = file_path.stat()
                return {
                    "filename": filename,
                    "size": stat.st_size,
                    "created_time": stat.st_ctime,
                    "format": self.format,
                    "sample_rate": self.sample_rate
                }
            else:
                return {"error": "文件不存在"}
                
        except Exception as e:
            logger.error(f"获取音频文件信息失败: {e}")
            return {"error": str(e)}