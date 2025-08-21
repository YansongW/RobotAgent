# -*- coding: utf-8 -*-

# TTS处理器 (Text-to-Speech Handler)
# 负责文本转语音功能和语音输出管理
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21

# 导入标准库
import os
import json
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from queue import Queue, Empty

# 导入第三方库 (可选)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# 导入项目基础组件
from ..communication.protocols import MessageType, AgentMessage


class TTSEngine(Enum):
    """TTS引擎类型"""
    SYSTEM = "system"           # 系统内置TTS
    PYTTSX3 = "pyttsx3"         # pyttsx3库
    ONLINE = "online"           # 在线TTS服务
    CUSTOM = "custom"           # 自定义TTS


class TTSStatus(Enum):
    """TTS状态"""
    IDLE = "idle"
    SPEAKING = "speaking"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class TTSRequest:
    """TTS请求数据结构"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    voice: Optional[str] = None
    rate: Optional[int] = None  # 语速
    volume: Optional[float] = None  # 音量 (0.0-1.0)
    pitch: Optional[int] = None  # 音调
    language: str = "zh-CN"
    priority: int = 0  # 优先级，数字越大优先级越高
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TTSResponse:
    """TTS响应数据结构"""
    request_id: str
    success: bool
    message: str = ""
    audio_file: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.now)


class TTSHandler:
    """TTS处理器"""
    
    def __init__(self, engine: TTSEngine = TTSEngine.SYSTEM,
                 output_dir: Optional[str] = None,
                 max_queue_size: int = 100):
        """初始化TTS处理器
        
        Args:
            engine: TTS引擎类型
            output_dir: 音频文件输出目录
            max_queue_size: 最大队列大小
        """
        self.engine_type = engine
        self.output_dir = Path(output_dir) if output_dir else Path("./tts_output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.max_queue_size = max_queue_size
        self.status = TTSStatus.IDLE
        self.current_request: Optional[TTSRequest] = None
        
        # 请求队列
        self.request_queue = Queue(maxsize=max_queue_size)
        self.response_callbacks: Dict[str, Callable] = {}
        
        # TTS引擎
        self.tts_engine = None
        self.engine_config = {
            'rate': 200,      # 语速
            'volume': 0.8,    # 音量
            'voice': None     # 语音
        }
        
        # 工作线程
        self.worker_thread = None
        self.is_running = False
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化TTS引擎
        self._initialize_engine()
    
    def _initialize_engine(self):
        """初始化TTS引擎"""
        try:
            if self.engine_type == TTSEngine.PYTTSX3 and PYTTSX3_AVAILABLE:
                self.tts_engine = pyttsx3.init()
                self._configure_pyttsx3()
                self.logger.info("初始化pyttsx3 TTS引擎成功")
            elif self.engine_type == TTSEngine.SYSTEM:
                # 使用系统默认TTS (Windows SAPI)
                if os.name == 'nt':  # Windows
                    self.logger.info("使用Windows系统TTS")
                else:
                    self.logger.warning("当前系统不支持系统TTS，切换到文本输出模式")
            else:
                self.logger.warning(f"TTS引擎 {self.engine_type} 不可用，使用文本输出模式")
        except Exception as e:
            self.logger.error(f"初始化TTS引擎失败: {e}")
            self.tts_engine = None
    
    def _configure_pyttsx3(self):
        """配置pyttsx3引擎"""
        if not self.tts_engine:
            return
        
        try:
            # 设置语速
            self.tts_engine.setProperty('rate', self.engine_config['rate'])
            
            # 设置音量
            self.tts_engine.setProperty('volume', self.engine_config['volume'])
            
            # 设置语音
            voices = self.tts_engine.getProperty('voices')
            if voices and self.engine_config['voice']:
                for voice in voices:
                    if self.engine_config['voice'] in voice.name:
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.logger.info("配置pyttsx3引擎完成")
        except Exception as e:
            self.logger.error(f"配置pyttsx3引擎失败: {e}")
    
    def start(self):
        """启动TTS处理器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        self.logger.info("TTS处理器已启动")
    
    def stop(self):
        """停止TTS处理器"""
        self.is_running = False
        
        # 停止当前播放
        self.stop_speaking()
        
        # 等待工作线程结束
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        self.logger.info("TTS处理器已停止")
    
    def speak(self, text: str, voice: Optional[str] = None,
             rate: Optional[int] = None, volume: Optional[float] = None,
             priority: int = 0, callback: Optional[Callable] = None,
             **kwargs) -> str:
        """添加TTS请求
        
        Args:
            text: 要转换的文本
            voice: 语音类型
            rate: 语速
            volume: 音量
            priority: 优先级
            callback: 完成回调
            **kwargs: 其他参数
            
        Returns:
            请求ID
        """
        request = TTSRequest(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume,
            priority=priority,
            callback=callback,
            metadata=kwargs
        )
        
        try:
            # 根据优先级插入队列
            if priority > 0:
                # 高优先级请求，插入到队列前面
                temp_queue = Queue()
                temp_queue.put(request)
                
                while not self.request_queue.empty():
                    try:
                        item = self.request_queue.get_nowait()
                        temp_queue.put(item)
                    except Empty:
                        break
                
                # 重新填充队列
                while not temp_queue.empty():
                    self.request_queue.put(temp_queue.get())
            else:
                self.request_queue.put(request, timeout=1)
            
            if callback:
                self.response_callbacks[request.request_id] = callback
            
            self.logger.info(f"添加TTS请求: {request.request_id[:8]}... 文本: {text[:50]}...")
            return request.request_id
            
        except Exception as e:
            self.logger.error(f"添加TTS请求失败: {e}")
            return ""
    
    def speak_immediately(self, text: str, **kwargs) -> str:
        """立即播放文本（高优先级）"""
        return self.speak(text, priority=10, **kwargs)
    
    def stop_speaking(self):
        """停止当前播放"""
        try:
            if self.tts_engine and hasattr(self.tts_engine, 'stop'):
                self.tts_engine.stop()
            
            self.status = TTSStatus.IDLE
            self.current_request = None
            
            self.logger.info("停止TTS播放")
        except Exception as e:
            self.logger.error(f"停止TTS播放失败: {e}")
    
    def pause_speaking(self):
        """暂停播放"""
        try:
            if self.status == TTSStatus.SPEAKING:
                self.status = TTSStatus.PAUSED
                self.logger.info("暂停TTS播放")
        except Exception as e:
            self.logger.error(f"暂停TTS播放失败: {e}")
    
    def resume_speaking(self):
        """恢复播放"""
        try:
            if self.status == TTSStatus.PAUSED:
                self.status = TTSStatus.SPEAKING
                self.logger.info("恢复TTS播放")
        except Exception as e:
            self.logger.error(f"恢复TTS播放失败: {e}")
    
    def clear_queue(self):
        """清空请求队列"""
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except Empty:
                break
        
        self.logger.info("清空TTS请求队列")
    
    def get_status(self) -> Dict[str, Any]:
        """获取TTS状态"""
        return {
            'status': self.status.value,
            'engine': self.engine_type.value,
            'queue_size': self.request_queue.qsize(),
            'current_request': self.current_request.request_id if self.current_request else None,
            'is_running': self.is_running,
            'engine_available': self.tts_engine is not None
        }
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """获取可用语音列表"""
        voices = []
        
        try:
            if self.tts_engine and hasattr(self.tts_engine, 'getProperty'):
                engine_voices = self.tts_engine.getProperty('voices')
                if engine_voices:
                    for voice in engine_voices:
                        voices.append({
                            'id': voice.id,
                            'name': voice.name,
                            'languages': getattr(voice, 'languages', []),
                            'gender': getattr(voice, 'gender', 'unknown')
                        })
        except Exception as e:
            self.logger.error(f"获取语音列表失败: {e}")
        
        return voices
    
    def set_engine_config(self, **config):
        """设置引擎配置"""
        self.engine_config.update(config)
        
        if self.tts_engine:
            self._configure_pyttsx3()
        
        self.logger.info(f"更新TTS引擎配置: {config}")
    
    def _worker_loop(self):
        """工作线程主循环"""
        self.logger.info("TTS工作线程启动")
        
        while self.is_running:
            try:
                # 获取请求
                request = self.request_queue.get(timeout=1)
                
                if not self.is_running:
                    break
                
                # 处理请求
                self._process_request(request)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"TTS工作线程错误: {e}")
        
        self.logger.info("TTS工作线程结束")
    
    def _process_request(self, request: TTSRequest):
        """处理TTS请求"""
        self.current_request = request
        self.status = TTSStatus.SPEAKING
        
        start_time = datetime.now()
        response = TTSResponse(
            request_id=request.request_id,
            success=False
        )
        
        try:
            # 应用请求的配置
            if request.rate:
                self.engine_config['rate'] = request.rate
            if request.volume:
                self.engine_config['volume'] = request.volume
            if request.voice:
                self.engine_config['voice'] = request.voice
            
            # 重新配置引擎
            if self.tts_engine:
                self._configure_pyttsx3()
            
            # 执行TTS
            success = self._execute_tts(request.text)
            
            if success:
                response.success = True
                response.message = "TTS播放完成"
                response.duration = (datetime.now() - start_time).total_seconds()
            else:
                response.error = "TTS播放失败"
            
        except Exception as e:
            response.error = str(e)
            self.logger.error(f"处理TTS请求失败: {e}")
        
        finally:
            self.status = TTSStatus.IDLE
            self.current_request = None
            
            # 调用回调
            if request.callback:
                try:
                    request.callback(response)
                except Exception as e:
                    self.logger.error(f"TTS回调执行失败: {e}")
            
            # 移除回调
            if request.request_id in self.response_callbacks:
                del self.response_callbacks[request.request_id]
    
    def _execute_tts(self, text: str) -> bool:
        """执行TTS转换"""
        try:
            if self.engine_type == TTSEngine.PYTTSX3 and self.tts_engine:
                # 使用pyttsx3
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return True
            
            elif self.engine_type == TTSEngine.SYSTEM and os.name == 'nt':
                # 使用Windows SAPI
                import subprocess
                # 使用PowerShell的语音合成
                cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak(\"{text}\")"'
                subprocess.run(cmd, shell=True, capture_output=True)
                return True
            
            else:
                # 文本输出模式
                print(f"[TTS] {text}")
                self.logger.info(f"文本输出: {text}")
                return True
                
        except Exception as e:
            self.logger.error(f"执行TTS失败: {e}")
            return False
    
    def save_to_file(self, text: str, filename: Optional[str] = None,
                    format: str = "wav") -> Optional[str]:
        """保存TTS到文件
        
        Args:
            text: 要转换的文本
            filename: 文件名
            format: 音频格式
            
        Returns:
            保存的文件路径
        """
        try:
            if not filename:
                filename = f"tts_{uuid.uuid4().hex[:8]}.{format}"
            
            file_path = self.output_dir / filename
            
            if self.tts_engine and hasattr(self.tts_engine, 'save_to_file'):
                self.tts_engine.save_to_file(text, str(file_path))
                self.logger.info(f"保存TTS到文件: {file_path}")
                return str(file_path)
            else:
                # 创建文本文件作为替代
                text_file = file_path.with_suffix('.txt')
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.logger.info(f"保存文本到文件: {text_file}")
                return str(text_file)
                
        except Exception as e:
            self.logger.error(f"保存TTS文件失败: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_requests': len(self.response_callbacks),
            'queue_size': self.request_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'current_status': self.status.value,
            'engine_type': self.engine_type.value,
            'is_running': self.is_running,
            'output_directory': str(self.output_dir)
        }