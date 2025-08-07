"""
ASR语音识别服务
基于火山引擎SAUC WebSocket协议实现语音转文字功能
"""

import json
import struct
import time
import uuid
import gzip
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# 协议常量定义
class ProtocolVersion:
    V1 = 0b0001

class MessageType:
    CLIENT_FULL_REQUEST = 0b0001
    CLIENT_AUDIO_ONLY_REQUEST = 0b0010
    SERVER_FULL_RESPONSE = 0b1001
    SERVER_ERROR_RESPONSE = 0b1111

class MessageTypeSpecificFlags:
    NO_SEQUENCE = 0b0000
    POS_SEQUENCE = 0b0001
    NEG_SEQUENCE = 0b0010
    NEG_WITH_SEQUENCE = 0b0011

class SerializationType:
    NO_SERIALIZATION = 0b0000
    JSON = 0b0001

class CompressionType:
    GZIP = 0b0001

class ASRRequestHeader:
    """ASR请求头"""
    
    def __init__(self):
        self.message_type = MessageType.CLIENT_FULL_REQUEST
        self.message_type_specific_flags = MessageTypeSpecificFlags.POS_SEQUENCE
        self.serialization_type = SerializationType.JSON
        self.compression_type = CompressionType.GZIP
        self.reserved_data = bytes([0x00])

    def with_message_type(self, message_type: int):
        self.message_type = message_type
        return self

    def with_message_type_specific_flags(self, flags: int):
        self.message_type_specific_flags = flags
        return self

    def to_bytes(self) -> bytes:
        header = bytearray()
        header.append((ProtocolVersion.V1 << 4) | 1)
        header.append((self.message_type << 4) | self.message_type_specific_flags)
        header.append((self.serialization_type << 4) | self.compression_type)
        header.extend(self.reserved_data)
        return bytes(header)

    @staticmethod
    def default_header():
        return ASRRequestHeader()

class ASRResponse:
    """ASR响应"""
    
    def __init__(self):
        self.code = 0
        self.event = 0
        self.is_last_package = False
        self.payload_sequence = 0
        self.payload_size = 0
        self.payload_msg = None

class ASRService:
    """ASR语音识别服务类"""
    
    def __init__(self, config: dict):
        """
        初始化ASR服务
        
        Args:
            config: ASR配置
        """
        self.config = config
        self.app_id = config["app_id"]
        self.access_token = config["access_token"]
        self.secret_key = config["secret_key"]
        self.base_url = config["base_url"]
        self.model = config["model"]
        self.resource_id = config.get("resource_id", "volc.bigasr.sauc.duration")
        self.language = config.get("language", "zh-CN")
        self.format = config.get("format", "wav")
        self.sample_rate = config.get("sample_rate", 16000)
        self.channel = config.get("channel", 1)
        self.bits = config.get("bits", 16)
        
        # WebSocket连接相关
        self.seq = 1
        self.conn = None
        self.session = None
        
        logger.info("ASR服务初始化完成")
    
    async def speech_to_text(self, audio_data: bytes) -> Optional[str]:
        """
        将语音转换为文字
        
        Args:
            audio_data: 音频数据（字节流）
            
        Returns:
            识别出的文字，失败返回None
        """
        try:
            # 使用SAUC WebSocket协议进行识别
            result = await self._sauc_recognition(audio_data)
            
            if result:
                recognized_text = result.strip()
                logger.info(f"语音识别成功: {recognized_text}")
                return recognized_text
            else:
                logger.warning("语音识别结果为空")
                return None
                
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return None
    
    async def _sauc_recognition(self, audio_data: bytes) -> Optional[str]:
        """
        使用SAUC协议进行语音识别
        
        Args:
            audio_data: 音频数据
            
        Returns:
            识别结果文本
        """
        self.seq = 1
        result_text = ""
        
        try:
            # 创建HTTP会话
            self.session = aiohttp.ClientSession()
            
            # 建立WebSocket连接
            await self._create_connection()
            
            # 发送完整客户端请求
            await self._send_full_client_request()
            
            # 发送音频数据并接收结果
            result_text = await self._send_audio_and_receive_result(audio_data)
            
            return result_text
            
        except Exception as e:
            logger.error(f"SAUC识别失败: {e}")
            return None
        finally:
            await self._cleanup_connection()
    
    async def _create_connection(self):
        """创建WebSocket连接"""
        headers = self._build_auth_headers()
        
        # 使用正确的WebSocket URL
        ws_url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
        
        try:
            self.conn = await self.session.ws_connect(ws_url, headers=headers)
            logger.info(f"已连接到 {ws_url}")
        except Exception as e:
            logger.error(f"WebSocket连接失败: {e}")
            raise
    
    def _build_auth_headers(self) -> Dict[str, str]:
        """构建认证头"""
        reqid = str(uuid.uuid4())
        return {
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": reqid,
            "X-Api-Access-Key": self.access_token,
            "X-Api-App-Key": self.app_id
        }
    
    async def _send_full_client_request(self):
        """发送完整客户端请求"""
        header = ASRRequestHeader.default_header().with_message_type_specific_flags(
            MessageTypeSpecificFlags.POS_SEQUENCE
        )
        
        payload = {
            "user": {
                "uid": "demo_uid"
            },
            "audio": {
                "format": self.format,
                "codec": "raw",
                "rate": self.sample_rate,
                "bits": self.bits,
                "channel": self.channel
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                "enable_ddc": True,
                "show_utterances": True,
                "enable_nonstream": False
            }
        }
        
        payload_bytes = json.dumps(payload).encode('utf-8')
        compressed_payload = gzip.compress(payload_bytes)
        payload_size = len(compressed_payload)
        
        request = bytearray()
        request.extend(header.to_bytes())
        request.extend(struct.pack('>i', self.seq))
        request.extend(struct.pack('>I', payload_size))
        request.extend(compressed_payload)
        
        await self.conn.send_bytes(bytes(request))
        logger.info(f"已发送完整客户端请求，seq: {self.seq}")
        
        self.seq += 1
        
        # 接收响应
        msg = await self.conn.receive()
        if msg.type == aiohttp.WSMsgType.BINARY:
            response = self._parse_response(msg.data)
            logger.info(f"收到响应: code={response.code}")
        else:
            logger.error(f"意外的消息类型: {msg.type}")
    
    async def _send_audio_and_receive_result(self, audio_data: bytes) -> str:
        """发送音频数据并接收识别结果"""
        result_text = ""
        
        # 计算分段大小（每200ms的音频数据）
        segment_duration = 200  # ms
        bytes_per_ms = self.sample_rate * self.channel * (self.bits // 8) // 1000
        segment_size = bytes_per_ms * segment_duration
        
        # 分割音频数据
        audio_segments = self._split_audio(audio_data, segment_size)
        total_segments = len(audio_segments)
        
        # 创建发送和接收任务
        send_task = asyncio.create_task(
            self._send_audio_segments(audio_segments)
        )
        receive_task = asyncio.create_task(
            self._receive_results()
        )
        
        try:
            # 等待接收任务完成
            results = await receive_task
            result_text = "".join(results)
        finally:
            # 取消发送任务
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
        
        return result_text
    
    async def _send_audio_segments(self, segments: List[bytes]):
        """发送音频分段"""
        total_segments = len(segments)
        
        for i, segment in enumerate(segments):
            is_last = (i == total_segments - 1)
            
            header = ASRRequestHeader.default_header()
            if is_last:
                header.with_message_type_specific_flags(MessageTypeSpecificFlags.NEG_WITH_SEQUENCE)
                seq = -self.seq
            else:
                header.with_message_type_specific_flags(MessageTypeSpecificFlags.POS_SEQUENCE)
                seq = self.seq
            
            header.with_message_type(MessageType.CLIENT_AUDIO_ONLY_REQUEST)
            
            request = bytearray()
            request.extend(header.to_bytes())
            request.extend(struct.pack('>i', seq))
            
            compressed_segment = gzip.compress(segment)
            request.extend(struct.pack('>I', len(compressed_segment)))
            request.extend(compressed_segment)
            
            await self.conn.send_bytes(bytes(request))
            logger.debug(f"已发送音频分段 {i+1}/{total_segments}, seq: {seq}")
            
            if not is_last:
                self.seq += 1
                await asyncio.sleep(0.2)  # 模拟实时流
    
    async def _receive_results(self) -> List[str]:
        """接收识别结果"""
        results = []
        
        try:
            async for msg in self.conn:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    response = self._parse_response(msg.data)
                    
                    if response.payload_msg:
                        # 提取识别文本
                        text = self._extract_text_from_response(response.payload_msg)
                        if text:
                            results.append(text)
                    
                    if response.is_last_package or response.code != 0:
                        break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket错误: {msg.data}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket连接已关闭")
                    break
        except Exception as e:
            logger.error(f"接收结果时出错: {e}")
        
        return results
    
    def _parse_response(self, data: bytes) -> 'ASRResponse':
        """解析响应数据"""
        try:
            # 解析消息头（前4字节）
            header_bytes = data[:4]
            header = ASRRequestHeader.from_bytes(header_bytes)
            
            # 解析序列号（4字节）
            seq = struct.unpack('>i', data[4:8])[0]
            
            # 解析payload大小（4字节）
            payload_size = struct.unpack('>I', data[8:12])[0]
            
            # 解析payload
            payload_data = data[12:12+payload_size]
            
            # 解压缩payload
            if header.compression_type == CompressionType.GZIP:
                payload_data = gzip.decompress(payload_data)
            
            # 解析JSON
            payload_msg = None
            if payload_data:
                try:
                    payload_msg = json.loads(payload_data.decode('utf-8'))
                except json.JSONDecodeError:
                    logger.warning("无法解析payload JSON")
            
            # 创建响应对象
            response = ASRResponse()
            response.header = header
            response.seq = seq
            response.payload_size = payload_size
            response.payload_msg = payload_msg
            
            # 设置响应状态
            if payload_msg:
                response.code = payload_msg.get('code', 0)
                response.message = payload_msg.get('message', '')
                response.is_last_package = payload_msg.get('is_last_package', False)
            
            return response
            
        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            response = ASRResponse()
            response.code = -1
            response.message = str(e)
            return response
    
    def _extract_text_from_response(self, payload_msg: dict) -> str:
        """从响应中提取识别文本"""
        try:
            if 'result' in payload_msg:
                result = payload_msg['result']
                if 'utterances' in result:
                    utterances = result['utterances']
                    if utterances and len(utterances) > 0:
                        # 获取最新的识别结果
                        latest_utterance = utterances[-1]
                        if 'text' in latest_utterance:
                            return latest_utterance['text']
                        elif 'words' in latest_utterance:
                            # 如果没有text字段，从words中拼接
                            words = latest_utterance['words']
                            return ''.join([word.get('text', '') for word in words])
            return ""
        except Exception as e:
            logger.error(f"提取文本失败: {e}")
            return ""
    
    def _split_audio(self, audio_data: bytes, segment_size: int) -> List[bytes]:
        """分割音频数据"""
        segments = []
        for i in range(0, len(audio_data), segment_size):
            segment = audio_data[i:i + segment_size]
            segments.append(segment)
        return segments
    
    async def _cleanup_connection(self):
        """清理连接"""
        try:
            if hasattr(self, 'conn') and self.conn:
                await self.conn.close()
            if hasattr(self, 'session') and self.session:
                await self.session.close()
        except Exception as e:
            logger.error(f"清理连接时出错: {e}")
    
    def _validate_audio_format(self, audio_data: bytes) -> bool:
        """
        验证音频格式
        
        Args:
            audio_data: 音频数据
            
        Returns:
            是否为有效格式
        """
        try:
            # 检查音频数据长度
            if len(audio_data) < 1024:  # 至少1KB
                logger.warning("音频数据太短")
                return False
            
            # 对于PCM格式，检查数据长度是否符合采样率和位深
            if self.format.lower() == "pcm":
                # 简单检查：数据长度应该是采样位数的倍数
                bytes_per_sample = self.bits // 8 * self.channel
                if len(audio_data) % bytes_per_sample != 0:
                    logger.warning("PCM音频数据长度不符合格式要求")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"音频格式验证失败: {e}")
            return False
    
    async def get_supported_languages(self) -> list:
        """
        获取支持的语言列表
        
        Returns:
            支持的语言代码列表
        """
        # 火山引擎ASR支持的语言
        return [
            "zh-CN",  # 中文（简体）
            "en-US",  # 英语（美国）
            "ja-JP",  # 日语
            "ko-KR",  # 韩语
        ]
    
    async def get_supported_formats(self) -> list:
        """
        获取支持的音频格式列表
        
        Returns:
            支持的音频格式列表
        """
        return [
            "pcm",    # 原始PCM格式（推荐）
            "wav",    # WAV格式
            "opus",   # Opus格式
        ]
    
    async def get_supported_sample_rates(self) -> list:
        """
        获取支持的采样率列表
        
        Returns:
            支持的采样率列表
        """
        return [8000, 16000, 24000, 48000]