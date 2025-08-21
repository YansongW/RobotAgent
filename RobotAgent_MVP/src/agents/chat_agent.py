# -*- coding: utf-8 -*-

# 对话智能体 (Chat Agent)
# 基于CAMEL框架的自然语言处理智能体，专注于多轮对话管理和自然语言理解
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-15


# 对话智能体实现

# 基于CAMEL框架构建的智能对话系统，提供：
# 1. 多轮对话管理和上下文维护
# 2. 自然语言理解和生成
# 3. 情感分析和意图识别
# 4. 与其他智能体的协作通信
# 5. 动态提示优化和学习能力

# 变更历史:
# v0.0.1 (2025-08-15):
# - 初始实现
# - 基础对话功能
# - CAMEL框架集成
# - 多轮对话上下文管理
# - 情感分析和意图识别

# 导入标准库
import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# 导入项目基础组件
from .base_agent import (
    BaseRobotAgent, AgentState, TaskStatus,
    TaskDefinition, AgentCapability, ToolDefinition
)
from config import (
    MessageType, AgentMessage, MessagePriority, TaskMessage, ResponseMessage,
    IntentType, MessageAnalysis
)
from src.communication.protocols import (
    CollaborationMode, StatusMessage
)
from src.communication.message_bus import get_message_bus

# 导入CAMEL框架组件
try:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.models import ModelFactory
    from camel.types import RoleType, ModelType
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    logging.warning("CAMEL框架未安装，使用模拟实现")


class ConversationState(Enum):
    # 对话状态枚举
    IDLE = "idle"                    # 空闲状态
    LISTENING = "listening"          # 监听用户输入
    PROCESSING = "processing"        # 处理用户消息
    GENERATING = "generating"        # 生成回复
    WAITING_CLARIFICATION = "waiting_clarification"  # 等待澄清
    COLLABORATING = "collaborating"  # 与其他智能体协作


class IntentType(Enum):
    # 意图类型枚举
    QUESTION = "question"            # 问题询问
    REQUEST = "request"              # 请求执行
    COMMAND = "command"              # 命令指令
    CONVERSATION = "conversation"    # 日常对话
    CLARIFICATION = "clarification"  # 澄清说明
    COLLABORATION = "collaboration"  # 协作请求
    UNKNOWN = "unknown"              # 未知意图


class EmotionType(Enum):
    # 情感类型枚举
    POSITIVE = "positive"            # 积极情感
    NEGATIVE = "negative"            # 消极情感
    NEUTRAL = "neutral"              # 中性情感
    EXCITED = "excited"              # 兴奋
    FRUSTRATED = "frustrated"        # 沮丧
    CONFUSED = "confused"            # 困惑
    SATISFIED = "satisfied"          # 满意


@dataclass
class ConversationContext:
    # 对话上下文数据结构
    conversation_id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    current_topic: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    context_summary: Optional[str] = None
    emotion_history: List[EmotionType] = field(default_factory=list)
    intent_history: List[IntentType] = field(default_factory=list)


@dataclass
class MessageAnalysis:
    # 消息分析结果
    intent: IntentType
    emotion: EmotionType
    confidence: float
    key_entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    requires_clarification: bool = False
    suggested_actions: List[str] = field(default_factory=list)


class ChatAgent(BaseRobotAgent):
    
    # 对话智能体 (Chat Agent)
    
    # 基于CAMEL框架的自然语言处理智能体，专注于：
    # 1. 多轮对话管理和上下文维护
    # 2. 自然语言理解和生成
    # 3. 情感分析和意图识别
    # 4. 与其他智能体的协作通信
    # 5. 动态学习和提示优化
    
    # Attributes:
    #    conversation_state (ConversationState): 当前对话状态
    #    active_conversations (Dict[str, ConversationContext]): 活跃对话上下文
    #    camel_agent (ChatAgent): CAMEL框架的ChatAgent实例
    #    response_templates (Dict[str, str]): 响应模板库
        
    # Example:
    #   >>> agent = ChatAgent("chat_001")
    #    >>> await agent.start()
    #    >>> response = await agent.process_message("你好，我需要帮助")
    #    >>> await agent.stop()
        
    # Note:
    #    需要配置有效的语言模型才能正常工作.
    #    支持多种模型后端(OpenAI, Claude, 本地模型等).
        
    #See Also:
    #    BaseRobotAgent: 基础智能体类
    #    ConversationContext: 对话上下文管理
    #    MessageAnalysis: 消息分析结果
    
    
    def __init__(self, 
                 agent_id: str,
                 config: Dict[str, Any] = None,
                 **kwargs):
        
        # 初始化对话智能体
        
        # Args:
        #    agent_id: 智能体唯一标识符
        #    config: 配置参数字典, 包含模型配置, 提示模板等
        #    **kwargs: 其他初始化参数
            
        # Raises:
        #    ValueError: 当配置参数无效时
        #    ImportError: 当CAMEL框架未安装时
        
        # 调用父类初始化，设置智能体类型为"chat"
        super().__init__(agent_id, "chat", config, **kwargs)
        
        # 对话状态管理
        self.conversation_state = ConversationState.IDLE
        self.active_conversations: Dict[str, ConversationContext] = {}
        
        # CAMEL框架集成
        self._camel_agent: Optional[ChatAgent] = None
        self._model_backend = None
        
        # 对话处理组件
        self._response_templates = self._load_response_templates()
        self._intent_patterns = self._load_intent_patterns()
        self._emotion_keywords = self._load_emotion_keywords()
        
        # 性能监控
        self._conversation_metrics = {
            'total_conversations': 0,
            'total_messages': 0,
            'average_response_time': 0.0,
            'satisfaction_score': 0.0
        }
        
        # 初始化专业化组件
        self._init_specialized_components()
        
        # 注册专业化工具
        self._register_specialized_tools()
        
        # 添加专业化能力
        self._add_specialized_capabilities()
        
        self.logger.info(f"对话智能体 {agent_id} 初始化完成")
    
    def _load_and_validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # 加载和验证配置
        
        # 确保配置参数的完整性和有效性.
        
        # Args:
        #    config: 原始配置字典
            
        # Returns:
        #    Dict[str, Any]: 验证后的配置字典
            
        # Raises:
        #    ValueError: 当配置参数无效时

        # 默认配置
        default_config = {
            'model_name': 'gpt-3.5-turbo',
            'model_type': 'openai',
            'temperature': 0.7,
            'max_tokens': 1000,
            'timeout': 30.0,
            'retry_count': 3,
            'message_window_size': 10,
            'context_retention_hours': 24,
            'enable_emotion_analysis': True,
            'enable_intent_recognition': True,
            'enable_learning': True,
            'response_style': 'friendly',
            'language': 'zh-CN'
        }
        
        # 合并配置
        merged_config = {**default_config, **(config or {})}
        
        # 验证配置
        self._validate_config(merged_config)
        
        return merged_config
    
    def _validate_config(self, config: Dict[str, Any]):
        """
        验证配置参数
        
        Args:
            config: 配置字典
            
        Raises:
            ValueError: 当配置参数无效时
        """
        required_keys = ['model_name', 'temperature']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"缺少必需的配置参数: {key}")
        
        # 参数范围验证
        if not 0 <= config['temperature'] <= 2:
            raise ValueError("temperature必须在0-2之间")
        
        if config['max_tokens'] <= 0:
            raise ValueError("max_tokens必须大于0")
        
        if config['timeout'] <= 0:
            raise ValueError("timeout必须大于0")
    
    def _init_specialized_components(self):
        """
        初始化专业化组件
        
        包括CAMEL框架集成、模型后端、对话管理器等。
        """
        try:
            # 初始化CAMEL智能体
            self._init_camel_agent()
            
            # 初始化对话管理器
            self._init_conversation_manager()
            
            # 初始化分析引擎
            self._init_analysis_engine()
            
            self.logger.info("专业化组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"专业化组件初始化失败: {e}")
            raise
    
    def _init_camel_agent(self):
        """
        初始化CAMEL ChatAgent
        
        基于配置创建CAMEL ChatAgent实例, 用于自然语言处理.
        如果CAMEL框架不可用, 将使用模拟实现.
        """
        if not CAMEL_AVAILABLE:
            self.logger.warning("CAMEL框架不可用，使用模拟实现")
            self._camel_agent = None
            return
        
        try:
            # 构建系统提示
            system_prompt = self._build_system_prompt()
            
            # 创建模型后端
            self._model_backend = self._create_model_backend()
            
            # 创建CAMEL ChatAgent
            self._camel_agent = ChatAgent(
                system_message=BaseMessage.make_assistant_message(
                    role_name="对话助手",
                    content=system_prompt
                ),
                model=self._model_backend,
                message_window_size=self.config.get('message_window_size', 10)
            )
            
            self.logger.info("CAMEL ChatAgent初始化成功")
            
        except Exception as e:
            self.logger.error(f"CAMEL ChatAgent初始化失败: {e}")
            self._camel_agent = None
    
    def _create_model_backend(self):
        """
        创建模型后端
        
        Returns:
            模型后端实例
        """
        if not CAMEL_AVAILABLE:
            return None
        
        model_type = self.config.get('model_type', 'openai')
        model_name = self.config.get('model_name', 'gpt-3.5-turbo')
        
        if model_type.lower() == 'openai':
            model_type_enum = ModelType.GPT_3_5_TURBO
        else:
            model_type_enum = ModelType.GPT_3_5_TURBO  # 默认使用GPT-3.5
        
        return ModelFactory.create(
            model_platform=model_type_enum,
            model_type=model_name,
            model_config_dict={
                'temperature': self.config.get('temperature', 0.7),
                'max_tokens': self.config.get('max_tokens', 1000)
            }
        )
    
    def _build_system_prompt(self) -> str:
        """
        构建系统提示
        
        从chat_agent_prompt_template.json文件加载提示词模板，
        如果加载失败则使用默认提示词。
        
        Returns:
            str: 系统提示文本
        """
        try:
            # 尝试从JSON配置文件加载提示词模板
            import os
            from pathlib import Path
            
            # 获取项目根目录
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            config_path = project_root / "config" / "chat_agent_prompt_template.json"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    prompt_config = json.load(f)
                
                # 将JSON配置转换为系统提示词
                system_prompt = f"""
你是一个基于多维行为树结构的智能助手。

设计理念：{prompt_config['overview']['设计理念']}

核心目标：
{chr(10).join(f"- {goal}" for goal in prompt_config['overview']['核心目标'])}

工作流程：
"""
                
                for stage in prompt_config['process']:
                    system_prompt += f"\n{stage['阶段']}：{stage['目标']}\n"
                    for detail in stage['细节']:
                        system_prompt += f"  • {detail['项']}：{detail['说明']}\n"
                        if '范例' in detail:
                            system_prompt += f"    范例：{detail['范例']}\n"
                
                system_prompt += f"\n期望效果：{prompt_config['expectation']}"
                
                self.logger.info("成功加载chat_agent_prompt_template.json提示词模板")
                return system_prompt
            else:
                self.logger.warning(f"提示词模板文件不存在: {config_path}")
                
        except Exception as e:
            self.logger.error(f"加载提示词模板失败: {e}")
        
        # 如果加载失败，使用默认提示词
        language = self.config.get('language', 'zh-CN')
        response_style = self.config.get('response_style', 'friendly')
        
        if language == 'zh-CN':
            prompt = f"""
你是一个智能对话助手，具有以下特点和能力：

## 角色定位
- 友好、专业、有帮助的AI助手
- 具备多领域知识和问题解决能力
- 能够理解上下文并维持连贯对话
- 支持多轮对话和复杂任务处理

## 对话风格
- 风格：{response_style}
- 语言：简洁明了，易于理解
- 态度：积极主动，乐于助人
- 回应：及时准确，有针对性

## 核心能力
1. 自然语言理解和生成
2. 上下文感知和记忆管理
3. 情感识别和适应性回应
4. 意图识别和任务规划
5. 多智能体协作和任务委派

## 行为准则
- 始终保持礼貌和专业
- 承认不确定性，必要时寻求澄清
- 保护用户隐私和数据安全
- 提供准确、有用的信息和建议
- 在复杂任务中主动寻求协作

请根据用户的输入提供恰当的回应。
"""
        else:
            prompt = f"""
You are an intelligent conversational assistant with the following characteristics:

## Role
- Friendly, professional, and helpful AI assistant
- Multi-domain knowledge and problem-solving capabilities
- Context understanding and coherent conversation maintenance
- Multi-turn dialogue and complex task processing support

## Conversation Style
- Style: {response_style}
- Language: Clear and easy to understand
- Attitude: Proactive and helpful
- Response: Timely, accurate, and targeted

## Core Capabilities
1. Natural language understanding and generation
2. Context awareness and memory management
3. Emotion recognition and adaptive responses
4. Intent recognition and task planning
5. Multi-agent collaboration and task delegation

## Behavioral Guidelines
- Always maintain politeness and professionalism
- Acknowledge uncertainty and seek clarification when necessary
- Protect user privacy and data security
- Provide accurate and useful information and advice
- Actively seek collaboration for complex tasks

Please provide appropriate responses based on user input.
"""
        
        return prompt
    
    def _init_conversation_manager(self):
        """
        初始化对话管理器
        
        设置对话上下文管理, 会话清理等功能.
        """
        # 对话清理任务
        self._cleanup_interval = self.config.get('context_retention_hours', 24) * 3600
        
        # 注意：定期清理任务将在异步环境中启动
        self._cleanup_task = None
        
        self.logger.info("对话管理器初始化完成")
    
    def _init_analysis_engine(self):
        """
        初始化分析引擎
        
        设置意图识别, 情感分析等功能.
        """
        self._enable_emotion_analysis = self.config.get('enable_emotion_analysis', True)
        self._enable_intent_recognition = self.config.get('enable_intent_recognition', True)
        
        self.logger.info("分析引擎初始化完成")
    
    def _register_specialized_tools(self):
        """
        注册专业化工具
        
        注册对话相关的工具和功能.
        """
        # 注册对话分析工具
        self.register_tool(
            name="conversation_analysis",
            description="分析用户消息的意图、情感和关键信息",
            function=self._analyze_conversation,
            parameters_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "要分析的消息"}
                },
                "required": ["message"]
            },
            category="conversation"
        )
        
        # 注册上下文总结工具
        self.register_tool(
            name="context_summarization",
            description="总结对话历史和关键信息",
            function=self._summarize_context,
            parameters_schema={
                "type": "object",
                "properties": {
                    "conversation_id": {"type": "string", "description": "对话ID"}
                },
                "required": ["conversation_id"]
            },
            category="conversation"
        )
        
        # 注册回复生成工具
        self.register_tool(
            name="response_generation",
            description="基于上下文生成合适的回复",
            function=self._generate_response,
            parameters_schema={
                "type": "object",
                "properties": {
                    "context": {"type": "object", "description": "对话上下文"},
                    "message": {"type": "string", "description": "用户消息"}
                },
                "required": ["context", "message"]
            },
            category="conversation"
        )
        
        self.logger.info("注册了 3 个专业化工具")
    
    def _analyze_conversation(self, message: str) -> dict:
        """
        分析对话消息
        
        Args:
            message: 要分析的消息
            
        Returns:
            分析结果字典
        """
        # 简单的意图和情感分析实现
        return {
            "intent": "general_conversation",
            "sentiment": "neutral",
            "keywords": message.split()[:5],
            "confidence": 0.8
        }
    
    def _summarize_context(self, conversation_id: str) -> str:
        """
        总结对话上下文
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            上下文总结
        """
        # 简单的上下文总结实现
        return f"对话 {conversation_id} 的上下文总结"
    
    def _generate_response(self, context: dict, message: str) -> str:
        """
        生成回复
        
        Args:
            context: 对话上下文
            message: 用户消息
            
        Returns:
            生成的回复
        """
        # 简单的回复生成实现
        return f"基于上下文对消息 '{message}' 的回复"
    
    def _add_specialized_capabilities(self):
        """
        添加专业化能力
        
        定义对话智能体的核心能力.
        """
        capabilities = [
            AgentCapability(
                name="自然语言理解",
                description="理解和解析自然语言输入",
                input_types=["text", "voice"],
                output_types=["intent", "entities"],
                confidence=0.9
            ),
            AgentCapability(
                name="对话管理",
                description="管理多轮对话和上下文",
                input_types=["conversation_context"],
                output_types=["dialogue_state"],
                confidence=0.85
            ),
            AgentCapability(
                name="情感识别",
                description="识别和分析用户情感状态",
                input_types=["text", "voice"],
                output_types=["emotion", "sentiment"],
                confidence=0.75
            ),
            AgentCapability(
                name="意图分类",
                description="识别用户意图和需求",
                input_types=["text"],
                output_types=["intent", "confidence"],
                confidence=0.9
            ),
            AgentCapability(
                name="回复生成",
                description="生成自然、合适的回复",
                input_types=["context", "intent"],
                output_types=["response"],
                confidence=0.8
            )
        ]
        
        for capability in capabilities:
            self.add_capability(capability)
        
        self.logger.info(f"添加了 {len(capabilities)} 个专业化能力")
    
    async def start(self) -> bool:
        """
        启动对话智能体
        
        Returns:
            bool: 启动是否成功
        """
        try:
            # 调用父类启动方法
            if not await super().start():
                return False
            
            # 设置对话状态
            self.conversation_state = ConversationState.IDLE
            
            # 启动CAMEL智能体
            if self._camel_agent:
                # CAMEL智能体通常不需要显式启动
                pass
            
            self.logger.info(f"对话智能体 {self.agent_id} 启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"对话智能体启动失败: {e}")
            await self._set_state(AgentState.ERROR)
            return False
    
    async def stop(self) -> bool:
        """
        停止对话智能体
        
        Returns:
            bool: 停止是否成功
        """
        try:
            # 清理活跃对话
            await self._cleanup_conversations()
            
            # 取消清理任务
            if hasattr(self, '_cleanup_task'):
                self._cleanup_task.cancel()
            
            # 调用父类停止方法
            result = await super().stop()
            
            self.logger.info(f"对话智能体 {self.agent_id} 停止成功")
            return result
            
        except Exception as e:
            self.logger.error(f"对话智能体停止失败: {e}")
            return False
    
    async def execute_task(self, task: TaskDefinition) -> Any:
        """
        执行任务的核心方法实现
        
        这是BaseRobotAgent的抽象方法, 必须在子类中实现.
        对话智能体的任务执行流程包括:
        1. 任务分析和预处理
        2. 消息处理和分析
        3. 回复生成和后处理
        
        Args:
            task: 任务定义对象，包含任务类型、参数等信息
            
        Returns:
            Any: 任务执行结果，通常是生成的回复文本
            
        Raises:
            ValueError: 当任务参数无效时
            RuntimeError: 当任务执行失败时
        """
        try:
            # 更新智能体状态
            await self._set_state(AgentState.EXECUTING)
            self.conversation_state = ConversationState.PROCESSING
            
            # 记录任务开始
            self.logger.info(f"开始执行对话任务: {task.task_id}")
            
            # 任务类型分发
            if task.task_type == "chat":
                result = await self._handle_chat_task(task)
            elif task.task_type == "analysis":
                result = await self._handle_analysis_task(task)
            elif task.task_type == "summarization":
                result = await self._handle_summarization_task(task)
            else:
                raise ValueError(f"不支持的任务类型: {task.task_type}")
            
            # 更新任务状态
            task.status = TaskStatus.COMPLETED
            task.result = result
            
            # 更新性能指标
            self._update_metrics(task)
            
            # 记录任务完成
            self.logger.info(f"对话任务执行完成: {task.task_id}")
            
            return result
            
        except Exception as e:
            # 错误处理
            task.status = TaskStatus.FAILED
            task.error_info = str(e)
            await self._set_state(AgentState.ERROR)
            
            self.logger.error(f"对话任务执行失败: {task.task_id}, 错误: {e}")
            raise
        
        finally:
            # 清理和状态恢复
            self.conversation_state = ConversationState.IDLE
            await self._set_state(AgentState.IDLE)
    
    async def _handle_chat_task(self, task: TaskDefinition) -> str:
        # 处理对话任务
        # Args: task: 对话任务定义
        # Returns: str: 生成的回复
        # 提取任务参数
        message = task.parameters.get('message', '')
        user_id = task.parameters.get('user_id', 'anonymous')
        conversation_id = task.parameters.get('conversation_id')
        
        if not message:
            raise ValueError("消息内容不能为空")
        
        # 处理消息
        response = await self.process_message(
            message=message,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        return response
    
    async def _handle_analysis_task(self, task: TaskDefinition) -> MessageAnalysis:
        # 处理分析任务
        # Args: task: 分析任务定义
        # Returns: MessageAnalysis: 分析结果
        message = task.parameters.get('message', '')
        
        if not message:
            raise ValueError("分析消息不能为空")
        
        # 执行消息分析
        analysis = await self._analyze_message(message)
        
        return analysis
    
    async def _handle_summarization_task(self, task: TaskDefinition) -> str:
        # 处理总结任务
        # Args: task: 总结任务定义
        # Returns: str: 总结结果
        conversation_id = task.parameters.get('conversation_id')
        
        if not conversation_id:
            raise ValueError("对话ID不能为空")
        
        # 执行对话总结
        summary = await self._summarize_conversation(conversation_id)
        
        return summary
    
    async def process_message(self, 
                            message: str, 
                            user_id: str = "anonymous",
                            conversation_id: Optional[str] = None) -> str:
        # 处理用户消息并生成回复
        # 这是对话智能体的核心方法，处理完整的对话流程。
        # Args: message: 用户消息内容, user_id: 用户标识符, conversation_id: 对话标识符，如果为None则创建新对话
        # Returns: str: 生成的回复消息
        # Raises: ValueError: 当消息参数无效时, RuntimeError: 当处理失败时
        try:
            # 参数验证
            if not message or not message.strip():
                raise ValueError("消息内容不能为空")
            
            # 更新对话状态
            self.conversation_state = ConversationState.PROCESSING
            
            # 获取或创建对话上下文
            context = await self._get_or_create_conversation(
                user_id=user_id,
                conversation_id=conversation_id
            )
            
            # 分析消息
            analysis = await self._analyze_message(message)
            
            # 更新对话上下文
            await self._update_conversation_context(context, message, analysis)
            
            # 生成回复
            self.conversation_state = ConversationState.GENERATING
            response = await self._generate_response(context, message, analysis)
            
            # 记录回复到上下文
            await self._record_response(context, response)
            
            # 更新指标
            self._conversation_metrics['total_messages'] += 1
            
            self.logger.info(f"消息处理完成: {len(message)} 字符 -> {len(response)} 字符")
            
            return response
            
        except Exception as e:
            self.logger.error(f"消息处理失败: {e}")
            self.conversation_state = ConversationState.IDLE
            
            # 返回错误回复
            return await self._generate_error_response(str(e))
        
        finally:
            # 恢复状态
            self.conversation_state = ConversationState.IDLE
    
    async def _get_or_create_conversation(self, 
                                        user_id: str,
                                        conversation_id: Optional[str] = None) -> ConversationContext:
        # 获取或创建对话上下文
        # Args: user_id: 用户ID, conversation_id: 对话ID
        # Returns: ConversationContext: 对话上下文对象
        # 如果没有指定对话ID，创建新的
        if conversation_id is None:
            conversation_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 检查是否存在活跃对话
        if conversation_id in self.active_conversations:
            context = self.active_conversations[conversation_id]
            context.last_activity = datetime.now()
            return context
        
        # 创建新的对话上下文
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            start_time=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.active_conversations[conversation_id] = context
        self._conversation_metrics['total_conversations'] += 1
        
        self.logger.info(f"创建新对话: {conversation_id}")
        
        return context
    
    async def _analyze_message(self, message: str) -> MessageAnalysis:
        # 分析用户消息
        # Args: message: 用户消息
        # Returns: MessageAnalysis: 分析结果
        try:
            # 意图识别
            intent = await self._recognize_intent(message)
            
            # 情感分析
            emotion = await self._analyze_emotion(message)
            
            # 实体提取
            entities = await self._extract_entities(message)
            
            # 主题识别
            topics = await self._identify_topics(message)
            
            # 计算置信度
            confidence = await self._calculate_confidence(message, intent, emotion)
            
            # 检查是否需要澄清
            requires_clarification = await self._check_clarification_needed(message, intent)
            
            # 生成建议动作
            suggested_actions = await self._generate_suggested_actions(intent, entities)
            
            analysis = MessageAnalysis(
                intent=intent,
                emotion=emotion,
                confidence=confidence,
                key_entities=entities,
                topics=topics,
                requires_clarification=requires_clarification,
                suggested_actions=suggested_actions
            )
            
            self.logger.debug(f"消息分析完成: 意图={intent.value}, 情感={emotion.value}, 置信度={confidence:.2f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"消息分析失败: {e}")
            # 返回默认分析结果
            return MessageAnalysis(
                intent=IntentType.UNKNOWN,
                emotion=EmotionType.NEUTRAL,
                confidence=0.0
            )
    
    async def _recognize_intent(self, message: str) -> IntentType:
        # 识别用户意图
        # Args: message: 用户消息
        # Returns: IntentType: 识别的意图类型
        if not self._enable_intent_recognition:
            return IntentType.CONVERSATION
        
        message_lower = message.lower().strip()
        
        # 基于关键词的简单意图识别
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return IntentType(intent)
        
        # 基于问号判断问题
        if '?' in message or '？' in message:
            return IntentType.QUESTION
        
        # 基于祈使句判断命令
        command_words = ['请', '帮我', '给我', '执行', '运行', '开始']
        if any(word in message for word in command_words):
            return IntentType.REQUEST
        
        # 默认为对话
        return IntentType.CONVERSATION
    
    async def _analyze_emotion(self, message: str) -> EmotionType:
        # 分析用户情感
        # Args: message: 用户消息
        # Returns: EmotionType: 识别的情感类型
        if not self._enable_emotion_analysis:
            return EmotionType.NEUTRAL
        
        message_lower = message.lower()
        
        # 基于关键词的简单情感分析
        for emotion, keywords in self._emotion_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return EmotionType(emotion)
        
        # 基于标点符号判断
        if '!' in message or '！' in message:
            return EmotionType.EXCITED
        
        # 默认为中性
        return EmotionType.NEUTRAL
    
    async def _extract_entities(self, message: str) -> List[str]:
        # 提取关键实体
        # Args: message: 用户消息
        # Returns: List[str]: 提取的实体列表
        entities = []
        
        # 简单的实体提取（可以集成更复杂的NER模型）
        # 提取数字
        numbers = re.findall(r'\d+', message)
        entities.extend([f"数字:{num}" for num in numbers])
        
        # 提取时间表达
        time_patterns = ['今天', '明天', '昨天', '现在', '稍后', '马上']
        for pattern in time_patterns:
            if pattern in message:
                entities.append(f"时间:{pattern}")
        
        return entities
    
    async def _identify_topics(self, message: str) -> List[str]:
        # 识别消息主题
        # Args: message: 用户消息
        # Returns: List[str]: 识别的主题列表
        topics = []
        
        # 基于关键词的主题识别
        topic_keywords = {
            '技术': ['编程', '代码', '软件', '系统', '算法', '数据库'],
            '生活': ['吃饭', '睡觉', '购物', '旅游', '健康', '运动'],
            '工作': ['项目', '会议', '报告', '任务', '同事', '老板'],
            '学习': ['学习', '考试', '课程', '知识', '技能', '培训']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in message for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ['通用']
    
    async def _calculate_confidence(self, 
                                  message: str, 
                                  intent: IntentType, 
                                  emotion: EmotionType) -> float:
        # 计算分析置信度
        # Args: message: 用户消息, intent: 识别的意图, emotion: 识别的情感
        # Returns: float: 置信度分数 (0.0-1.0)
        confidence = 0.5  # 基础置信度
        
        # 基于消息长度调整
        if len(message) > 10:
            confidence += 0.1
        if len(message) > 50:
            confidence += 0.1
        
        # 基于意图明确性调整
        if intent != IntentType.UNKNOWN:
            confidence += 0.2
        
        # 基于情感明确性调整
        if emotion != EmotionType.NEUTRAL:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _check_clarification_needed(self, 
                                        message: str, 
                                        intent: IntentType) -> bool:
        # 检查是否需要澄清
        # Args: message: 用户消息, intent: 识别的意图
        # Returns: bool: 是否需要澄清
        # 意图不明确时需要澄清
        if intent == IntentType.UNKNOWN:
            return True
        
        # 消息过短可能需要澄清
        if len(message.strip()) < 5:
            return True
        
        # 包含模糊词汇
        ambiguous_words = ['这个', '那个', '它', '他们', '什么', '怎么']
        if any(word in message for word in ambiguous_words):
            return True
        
        return False
    
    async def _generate_suggested_actions(self, 
                                        intent: IntentType, 
                                        entities: List[str]) -> List[str]:
        # 生成建议动作
        # Args: intent: 用户意图, entities: 提取的实体
        # Returns: List[str]: 建议动作列表
        actions = []
        
        if intent == IntentType.QUESTION:
            actions.append("提供详细回答")
            actions.append("搜索相关信息")
        elif intent == IntentType.REQUEST:
            actions.append("执行请求的操作")
            actions.append("确认操作参数")
        elif intent == IntentType.COMMAND:
            actions.append("解析命令参数")
            actions.append("执行命令")
        elif intent == IntentType.COLLABORATION:
            actions.append("联系相关智能体")
            actions.append("协调任务分配")
        
        return actions
    
    async def _update_conversation_context(self, 
                                         context: ConversationContext,
                                         message: str,
                                         analysis: MessageAnalysis):
        # 更新对话上下文
        # Args: context: 对话上下文, message: 用户消息, analysis: 消息分析结果
        # 添加消息到历史
        context.message_history.append({
            'timestamp': datetime.now().isoformat(),
            'role': 'user',
            'content': message,
            'intent': analysis.intent.value,
            'emotion': analysis.emotion.value,
            'entities': analysis.key_entities
        })
        
        # 更新情感和意图历史
        context.emotion_history.append(analysis.emotion)
        context.intent_history.append(analysis.intent)
        
        # 保持历史长度限制
        max_history = self.config.get('message_window_size', 10)
        if len(context.message_history) > max_history:
            context.message_history = context.message_history[-max_history:]
        
        if len(context.emotion_history) > max_history:
            context.emotion_history = context.emotion_history[-max_history:]
        
        if len(context.intent_history) > max_history:
            context.intent_history = context.intent_history[-max_history:]
        
        # 更新当前主题
        if analysis.topics:
            context.current_topic = analysis.topics[0]
        
        # 更新活动时间
        context.last_activity = datetime.now()
    
    async def _generate_response(self, 
                               context: ConversationContext,
                               message: str,
                               analysis: MessageAnalysis) -> str:
        # 生成回复
        # Args: context: 对话上下文, message: 用户消息, analysis: 消息分析结果
        # Returns: str: 生成的回复
        try:
            # 如果需要澄清
            if analysis.requires_clarification:
                return await self._generate_clarification_response(message, analysis)
            
            # 如果需要协作
            if analysis.intent == IntentType.COLLABORATION:
                return await self._handle_collaboration_request(context, message, analysis)
            
            # 使用CAMEL生成回复
            if self._camel_agent:
                response = await self._generate_camel_response(context, message)
            else:
                response = await self._generate_fallback_response(context, message, analysis)
            
            # 后处理回复
            response = await self._post_process_response(response, analysis)
            
            return response
            
        except Exception as e:
            self.logger.error(f"回复生成失败: {e}")
            return await self._generate_error_response(str(e))
    
    async def _generate_camel_response(self, 
                                     context: ConversationContext,
                                     message: str) -> str:
        # 使用CAMEL生成回复
        # Args: context: 对话上下文, message: 用户消息
        # Returns: str: 生成的回复
        if not self._camel_agent:
            raise RuntimeError("CAMEL智能体未初始化")
        
        try:
            # 构建上下文消息
            context_messages = []
            for msg in context.message_history[-5:]:  # 最近5条消息
                role = "user" if msg['role'] == 'user' else "assistant"
                context_messages.append(f"{role}: {msg['content']}")
            
            # 构建完整提示
            full_prompt = f"""
对话历史:
{chr(10).join(context_messages)}

当前用户消息: {message}

请基于对话历史和当前消息生成合适的回复。
"""
            
            # 使用CAMEL生成回复
            user_message = BaseMessage.make_user_message(
                role_name="用户",
                content=full_prompt
            )
            
            response = self._camel_agent.step(user_message)
            
            if response and hasattr(response, 'msg') and response.msg:
                return response.msg.content
            else:
                raise RuntimeError("CAMEL返回空回复")
                
        except Exception as e:
            self.logger.error(f"CAMEL回复生成失败: {e}")
            raise
    
    async def _generate_fallback_response(self, 
                                        context: ConversationContext,
                                        message: str,
                                        analysis: MessageAnalysis) -> str:
        # 生成备用回复（当CAMEL不可用时）
        # Args: context: 对话上下文, message: 用户消息, analysis: 消息分析结果
        # Returns: str: 生成的回复
        # 基于意图选择回复模板
        if analysis.intent == IntentType.QUESTION:
            templates = self._response_templates.get('question', [
                "这是一个很好的问题。让我来为您解答。",
                "关于您的问题，我需要更多信息才能给出准确答案。",
                "我理解您的疑问，让我尽力帮助您。"
            ])
        elif analysis.intent == IntentType.REQUEST:
            templates = self._response_templates.get('request', [
                "我会尽力帮助您完成这个请求。",
                "让我来处理您的请求。",
                "我明白您的需求，正在为您处理。"
            ])
        elif analysis.intent == IntentType.COMMAND:
            templates = self._response_templates.get('command', [
                "收到指令，正在执行。",
                "我会按照您的指令进行操作。",
                "指令已接收，开始处理。"
            ])
        else:
            templates = self._response_templates.get('conversation', [
                "谢谢您的分享，我很高兴与您交流。",
                "我理解您的想法。",
                "这很有趣，请继续。"
            ])
        
        # 选择合适的模板
        import random
        template = random.choice(templates)
        
        # 个性化回复
        if context.current_topic:
            template += f" 关于{context.current_topic}，我可以为您提供更多帮助。"
        
        return template
    
    async def _generate_clarification_response(self, 
                                             message: str,
                                             analysis: MessageAnalysis) -> str:
        """
        生成澄清回复
        
        Args:
            message: 用户消息
            analysis: 消息分析结果
            
        Returns:
            str: 澄清回复
        """
        clarification_templates = [
            "我需要更多信息来更好地帮助您。您能详细说明一下吗？",
            "为了给您提供准确的帮助，请您再详细描述一下您的需求。",
            "我想确保理解正确，您是想要...？",
            "抱歉，我没有完全理解您的意思。您能换个方式表达吗？"
        ]
        
        import random
        return random.choice(clarification_templates)
    
    async def _handle_collaboration_request(self, 
                                          context: ConversationContext,
                                          message: str,
                                          analysis: MessageAnalysis) -> str:
        """
        处理协作请求
        
        Args:
            context: 对话上下文
            message: 用户消息
            analysis: 消息分析结果
            
        Returns:
            str: 协作回复
        """
        # 这里可以实现与其他智能体的协作逻辑
        # 目前返回简单回复
        return "我理解您需要更专业的帮助。让我联系相关的专家智能体来协助您。"
    
    async def _post_process_response(self, 
                                   response: str,
                                   analysis: MessageAnalysis) -> str:
        """
        后处理回复
        
        Args:
            response: 原始回复
            analysis: 消息分析结果
            
        Returns:
            str: 处理后的回复
        """
        # 基于情感调整语调
        if analysis.emotion == EmotionType.FRUSTRATED:
            response = "我理解您的困扰。" + response
        elif analysis.emotion == EmotionType.EXCITED:
            response = response + " 我也很兴奋能帮助您！"
        
        # 确保回复不为空
        if not response.strip():
            response = "抱歉，我现在无法给出合适的回复。请稍后再试。"
        
        return response.strip()
    
    async def _generate_error_response(self, error_message: str) -> str:
        """
        生成错误回复
        
        Args:
            error_message: 错误信息
            
        Returns:
            str: 错误回复
        """
        error_templates = [
            "抱歉，我遇到了一些技术问题。请稍后再试。",
            "很抱歉，我现在无法正常处理您的请求。",
            "系统出现了一些问题，我正在尝试解决。请您稍等。"
        ]
        
        import random
        return random.choice(error_templates)
    
    async def _record_response(self, 
                             context: ConversationContext,
                             response: str):
        # 记录回复到对话上下文
        # Args:
        #     context: 对话上下文
        #     response: 生成的回复
        context.message_history.append({
            'timestamp': datetime.now().isoformat(),
            'role': 'assistant',
            'content': response
        })
        
        # 更新活动时间
        context.last_activity = datetime.now()
    
    async def _summarize_conversation(self, conversation_id: str) -> str:
        # 总结对话内容
        # Args:
        #     conversation_id: 对话ID
        # Returns:
        #     str: 对话总结
        if conversation_id not in self.active_conversations:
            return "未找到指定的对话。"
        
        context = self.active_conversations[conversation_id]
        
        if not context.message_history:
            return "对话暂无内容。"
        
        # 简单的总结逻辑
        total_messages = len(context.message_history)
        user_messages = [msg for msg in context.message_history if msg['role'] == 'user']
        assistant_messages = [msg for msg in context.message_history if msg['role'] == 'assistant']
        
        # 提取主要主题
        topics = set()
        for msg in user_messages:
            if 'entities' in msg:
                topics.update(msg.get('entities', []))
        
        summary = f"""
对话总结 (ID: {conversation_id}):
- 开始时间: {context.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- 消息总数: {total_messages} (用户: {len(user_messages)}, 助手: {len(assistant_messages)})
- 主要主题: {', '.join(list(topics)[:5]) if topics else '无特定主题'}
- 当前状态: {'活跃' if context.conversation_id in self.active_conversations else '已结束'}
"""
        
        return summary
    
    async def _update_metrics(self, task: TaskDefinition):
        # 更新性能指标
        # Args:
        #     task: 已完成的任务
        if hasattr(task, 'execution_time'):
            # 更新平均响应时间
            current_avg = self._conversation_metrics['average_response_time']
            total_messages = self._conversation_metrics['total_messages']
            
            new_avg = (current_avg * total_messages + task.execution_time) / (total_messages + 1)
            self._conversation_metrics['average_response_time'] = new_avg
    
    async def _periodic_cleanup(self):
        # 定期清理过期对话
        # 这是一个后台任务，定期清理不活跃的对话上下文。
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时检查一次
                
                current_time = datetime.now()
                expired_conversations = []
                
                for conv_id, context in self.active_conversations.items():
                    # 检查是否过期
                    time_diff = current_time - context.last_activity
                    if time_diff.total_seconds() > self._cleanup_interval:
                        expired_conversations.append(conv_id)
                
                # 清理过期对话
                for conv_id in expired_conversations:
                    del self.active_conversations[conv_id]
                    self.logger.info(f"清理过期对话: {conv_id}")
                
                if expired_conversations:
                    self.logger.info(f"清理了 {len(expired_conversations)} 个过期对话")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"对话清理任务出错: {e}")
    
    async def _cleanup_conversations(self):
        # 清理所有活跃对话
        # 在智能体停止时调用。
        conversation_count = len(self.active_conversations)
        self.active_conversations.clear()
        
        if conversation_count > 0:
            self.logger.info(f"清理了 {conversation_count} 个活跃对话")
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        # 加载回复模板
        # Returns: Dict[str, List[str]]: 回复模板字典
        return {
            'question': [
                "这是一个很好的问题。让我来为您解答。",
                "关于您的问题，我需要更多信息才能给出准确答案。",
                "我理解您的疑问，让我尽力帮助您。",
                "您提出了一个有趣的问题，让我仔细思考一下。"
            ],
            'request': [
                "我会尽力帮助您完成这个请求。",
                "让我来处理您的请求。",
                "我明白您的需求，正在为您处理。",
                "收到您的请求，我会立即开始处理。"
            ],
            'command': [
                "收到指令，正在执行。",
                "我会按照您的指令进行操作。",
                "指令已接收，开始处理。",
                "明白，我会立即执行您的指令。"
            ],
            'conversation': [
                "谢谢您的分享，我很高兴与您交流。",
                "我理解您的想法。",
                "这很有趣，请继续。",
                "我很乐意继续我们的对话。"
            ]
        }
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        # 加载意图识别模式
        # Returns: Dict[str, List[str]]: 意图模式字典
        return {
            'question': [
                r'什么是|什么叫|如何|怎么|为什么|哪里|哪个|谁|何时',
                r'能否|可以.*吗|是否|有没有',
                r'\?|？'
            ],
            'request': [
                r'请.*帮|帮我|帮助我|协助我',
                r'我需要|我想要|我希望',
                r'能不能|可不可以|麻烦您'
            ],
            'command': [
                 r'^(执行|运行|开始|启动|停止|结束)',
                 r'立即|马上|现在就',
                 r'命令|指令|操作'
             ],
             'collaboration': [
                 r'协作|合作|配合',
                 r'其他.*智能体|专家.*帮助',
                 r'转交|委托|分配'
             ]
         }
    
    def _load_emotion_keywords(self) -> Dict[str, List[str]]:
        # 加载情感关键词
        # Returns: Dict[str, List[str]]: 情感关键词字典
        return {
            'positive': [
                '好', '棒', '赞', '优秀', '满意', '开心', '高兴', '喜欢',
                '感谢', '谢谢', '不错', '很好', '完美', '太好了'
            ],
            'negative': [
                '不好', '糟糕', '失望', '生气', '愤怒', '讨厌', '烦躁',
                '不满', '抱怨', '问题', '错误', '失败', '糟糕'
            ],
            'excited': [
                '太棒了', 'amazing', '惊喜', '兴奋', '激动', '哇',
                '太好了', 'fantastic', 'wonderful', 'excellent'
            ],
            'frustrated': [
                '烦死了', '郁闷', '无语', '崩溃', '受不了', '气死了',
                '为什么', '怎么回事', '搞什么', '真是的'
            ],
            'confused': [
                '不懂', '不明白', '搞不清', '糊涂', '迷惑', '不理解',
                '什么意思', '怎么回事', '为什么', '不知道'
            ],
            'satisfied': [
                '满意', '满足', '够了', '可以了', '行了', '好的',
                '没问题', 'ok', '好吧', '同意'
            ]
        }
    
    # 公共接口方法
    
    def get_conversation_metrics(self) -> Dict[str, Any]:
        # 获取对话性能指标
        # Returns: Dict[str, Any]: 性能指标字典
        return self._conversation_metrics.copy()
    
    def get_active_conversations_count(self) -> int:
        # 获取活跃对话数量
        # Returns: int: 活跃对话数量
        return len(self.active_conversations)
    
    def get_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        # 获取指定对话的上下文
        # Args:
        #     conversation_id: 对话ID
        # Returns:
        #     Optional[ConversationContext]: 对话上下文，如果不存在则返回None
        return self.active_conversations.get(conversation_id)
    
    async def end_conversation(self, conversation_id: str) -> bool:
        # 结束指定对话
        # Args:
        #     conversation_id: 对话ID
        # Returns:
        #     bool: 是否成功结束对话
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
            self.logger.info(f"手动结束对话: {conversation_id}")
            return True
        return False
    
    async def clear_all_conversations(self) -> int:
        # 清理所有对话
        # Returns: int: 清理的对话数量
        count = len(self.active_conversations)
        self.active_conversations.clear()
        self.logger.info(f"清理了所有 {count} 个对话")
        return count
    
    def set_response_style(self, style: str):
        # 设置回复风格
        # Args: style: 回复风格 ('friendly', 'professional', 'casual', 'formal')
        self.config['response_style'] = style
        self.logger.info(f"回复风格已设置为: {style}")
    
    def enable_emotion_analysis(self, enabled: bool = True):
        # 启用或禁用情感分析
        # Args: enabled: 是否启用情感分析
        self._enable_emotion_analysis = enabled
        self.config['enable_emotion_analysis'] = enabled
        self.logger.info(f"情感分析已{'启用' if enabled else '禁用'}")
    
    def enable_intent_recognition(self, enabled: bool = True):
        # 启用或禁用意图识别
        # Args: enabled: 是否启用意图识别
        self._enable_intent_recognition = enabled
        self.config['enable_intent_recognition'] = enabled
        self.logger.info(f"意图识别已{'启用' if enabled else '禁用'}")
    
    def get_conversation_state(self) -> ConversationState:
        # 获取当前对话状态
        # Returns: ConversationState: 当前对话状态
        return self.conversation_state
    
    def is_camel_available(self) -> bool:
        # 检查CAMEL框架是否可用
        # Returns: bool: CAMEL框架是否可用
        return CAMEL_AVAILABLE and self._camel_agent is not None
    
    async def update_user_preferences(self, 
                                    conversation_id: str,
                                    preferences: Dict[str, Any]) -> bool:
        # 更新用户偏好设置
        # Args: conversation_id: 对话ID, preferences: 用户偏好字典
        # Returns: bool: 是否成功更新
        if conversation_id in self.active_conversations:
            context = self.active_conversations[conversation_id]
            context.user_preferences.update(preferences)
            self.logger.info(f"更新对话 {conversation_id} 的用户偏好")
            return True
        return False
    
    def get_supported_languages(self) -> List[str]:
        # 获取支持的语言列表
        # Returns: List[str]: 支持的语言代码列表
        return ['zh-CN', 'en-US', 'ja-JP', 'ko-KR']
    
    def set_language(self, language: str):
        # 设置对话语言
        # Args: language: 语言代码 (如 'zh-CN', 'en-US')
        if language in self.get_supported_languages():
            self.config['language'] = language
            self.logger.info(f"对话语言已设置为: {language}")
            
            # 重新构建系统提示
            if self._camel_agent:
                try:
                    system_prompt = self._build_system_prompt()
                    # 注意：这里可能需要重新初始化CAMEL智能体
                    # 具体实现取决于CAMEL框架的API
                except Exception as e:
                    self.logger.warning(f"更新系统提示失败: {e}")
        else:
            raise ValueError(f"不支持的语言: {language}")
    
    # 调试和监控方法
    
    def get_debug_info(self) -> Dict[str, Any]:
        # 获取调试信息
        # Returns: Dict[str, Any]: 调试信息字典
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self.state.value,
            'conversation_state': self.conversation_state.value,
            'active_conversations': len(self.active_conversations),
            'camel_available': self.is_camel_available(),
            'config': self.config,
            'metrics': self._conversation_metrics,
            'capabilities_count': len(self.capabilities),
            'tools_count': len(self.tools)
        }
    
    def export_conversation_history(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        # 导出对话历史
        # Args: conversation_id: 对话ID
        # Returns: Optional[Dict[str, Any]]: 对话历史数据，如果对话不存在则返回None
        if conversation_id not in self.active_conversations:
            return None
        
        context = self.active_conversations[conversation_id]
        
        return {
            'conversation_id': context.conversation_id,
            'user_id': context.user_id,
            'start_time': context.start_time.isoformat(),
            'last_activity': context.last_activity.isoformat(),
            'message_history': context.message_history,
            'current_topic': context.current_topic,
            'user_preferences': context.user_preferences,
            'context_summary': context.context_summary,
            'emotion_history': [e.value for e in context.emotion_history],
            'intent_history': [i.value for i in context.intent_history]
        }
    
    async def import_conversation_history(self, history_data: Dict[str, Any]) -> bool:
        # 导入对话历史
        # Args: history_data: 对话历史数据
        # Returns: bool: 是否成功导入
        try:
            conversation_id = history_data['conversation_id']
            
            # 创建对话上下文
            context = ConversationContext(
                conversation_id=conversation_id,
                user_id=history_data['user_id'],
                start_time=datetime.fromisoformat(history_data['start_time']),
                last_activity=datetime.fromisoformat(history_data['last_activity']),
                message_history=history_data.get('message_history', []),
                current_topic=history_data.get('current_topic'),
                user_preferences=history_data.get('user_preferences', {}),
                context_summary=history_data.get('context_summary')
            )
            
            # 恢复情感和意图历史
            if 'emotion_history' in history_data:
                context.emotion_history = [EmotionType(e) for e in history_data['emotion_history']]
            
            if 'intent_history' in history_data:
                context.intent_history = [IntentType(i) for i in history_data['intent_history']]
            
            # 添加到活跃对话
            self.active_conversations[conversation_id] = context
            
            self.logger.info(f"成功导入对话历史: {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"导入对话历史失败: {e}")
            return False


# 模块级别的工具函数

def create_chat_agent(agent_id: str, 
                     config: Dict[str, Any] = None) -> ChatAgent:
    """
    创建对话智能体的便捷函数
    
    Args:
        agent_id: 智能体ID
        config: 配置参数
        
    Returns:
        ChatAgent: 创建的对话智能体实例
    """
    return ChatAgent(agent_id, config)


def get_default_chat_config() -> Dict[str, Any]:
    """
    获取默认的对话智能体配置
    
    Returns:
        Dict[str, Any]: 默认配置字典
    """
    return {
        'model_name': 'gpt-3.5-turbo',
        'model_type': 'openai',
        'temperature': 0.7,
        'max_tokens': 1000,
        'timeout': 30.0,
        'retry_count': 3,
        'message_window_size': 10,
        'context_retention_hours': 24,
        'enable_emotion_analysis': True,
        'enable_intent_recognition': True,
        'enable_learning': True,
        'response_style': 'friendly',
        'language': 'zh-CN'
    }


# 异常类定义

class ChatAgentError(Exception):
    # 对话智能体基础异常类
    pass


class ConversationNotFoundError(ChatAgentError):
    # 对话未找到异常
    pass


class MessageAnalysisError(ChatAgentError):
    # 消息分析异常
    pass


class ResponseGenerationError(ChatAgentError):
    # 回复生成异常
    pass


# 导出的公共接口
__all__ = [
    'ChatAgent',
    'ConversationState',
    'IntentType', 
    'EmotionType',
    'ConversationContext',
    'MessageAnalysis',
    'create_chat_agent',
    'get_default_chat_config',
    'ChatAgentError',
    'ConversationNotFoundError',
    'MessageAnalysisError',
    'ResponseGenerationError'
]