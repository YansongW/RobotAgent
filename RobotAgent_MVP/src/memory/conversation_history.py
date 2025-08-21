# -*- coding: utf-8 -*-

# 对话历史管理 (Conversation History Manager)
# 管理用户对话历史、上下文和会话状态
# 作者: RobotAgent开发团队
# 版本: 0.0.1 (Initial Release)
# 更新时间: 2025-08-21

# 导入标准库
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
import logging

# 导入项目基础组件
from ..communication.protocols import MessageType, AgentMessage


@dataclass
class ConversationTurn:
    """对话轮次数据结构"""
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_input: str = ""
    agent_response: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """从字典创建实例"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ConversationSession:
    """对话会话数据结构"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    turns: List[ConversationTurn] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def add_turn(self, user_input: str, agent_response: str, 
                 context: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> ConversationTurn:
        """添加对话轮次"""
        turn = ConversationTurn(
            user_input=user_input,
            agent_response=agent_response,
            context=context or {},
            metadata=metadata or {}
        )
        self.turns.append(turn)
        self.last_activity = datetime.now()
        return turn
    
    def get_recent_turns(self, count: int = 5) -> List[ConversationTurn]:
        """获取最近的对话轮次"""
        return self.turns[-count:] if self.turns else []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'start_time': self.start_time.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'turns': [turn.to_dict() for turn in self.turns],
            'context': self.context,
            'metadata': self.metadata,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        """从字典创建实例"""
        turns_data = data.pop('turns', [])
        if 'start_time' in data and isinstance(data['start_time'], str):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if 'last_activity' in data and isinstance(data['last_activity'], str):
            data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        
        session = cls(**data)
        session.turns = [ConversationTurn.from_dict(turn_data) for turn_data in turns_data]
        return session


class ConversationHistoryManager:
    """对话历史管理器"""
    
    def __init__(self, max_sessions: int = 100, max_turns_per_session: int = 1000):
        """初始化对话历史管理器
        
        Args:
            max_sessions: 最大会话数量
            max_turns_per_session: 每个会话最大轮次数
        """
        self.max_sessions = max_sessions
        self.max_turns_per_session = max_turns_per_session
        self.sessions: Dict[str, ConversationSession] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> session_ids
        self.logger = logging.getLogger(__name__)
    
    def create_session(self, user_id: str, 
                      context: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> ConversationSession:
        """创建新的对话会话"""
        session = ConversationSession(
            user_id=user_id,
            context=context or {},
            metadata=metadata or {}
        )
        
        # 添加到会话管理
        self.sessions[session.session_id] = session
        
        # 添加到用户会话列表
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session.session_id)
        
        # 清理旧会话
        self._cleanup_old_sessions()
        
        self.logger.info(f"创建新会话: {session.session_id} for user: {user_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """获取指定会话"""
        return self.sessions.get(session_id)
    
    def get_active_session(self, user_id: str) -> Optional[ConversationSession]:
        """获取用户的活跃会话"""
        user_session_ids = self.user_sessions.get(user_id, [])
        for session_id in reversed(user_session_ids):  # 从最新的开始查找
            session = self.sessions.get(session_id)
            if session and session.is_active:
                return session
        return None
    
    def add_conversation_turn(self, session_id: str, user_input: str, 
                            agent_response: str,
                            context: Optional[Dict[str, Any]] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> Optional[ConversationTurn]:
        """添加对话轮次"""
        session = self.get_session(session_id)
        if not session:
            self.logger.warning(f"会话不存在: {session_id}")
            return None
        
        turn = session.add_turn(user_input, agent_response, context, metadata)
        
        # 检查轮次数量限制
        if len(session.turns) > self.max_turns_per_session:
            # 移除最旧的轮次
            removed_turns = session.turns[:len(session.turns) - self.max_turns_per_session]
            session.turns = session.turns[-self.max_turns_per_session:]
            self.logger.info(f"会话 {session_id} 移除了 {len(removed_turns)} 个旧轮次")
        
        return turn
    
    def get_conversation_context(self, session_id: str, 
                               recent_turns: int = 5) -> Dict[str, Any]:
        """获取对话上下文"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        recent_turns_data = session.get_recent_turns(recent_turns)
        
        return {
            'session_id': session_id,
            'user_id': session.user_id,
            'session_context': session.context,
            'recent_turns': [turn.to_dict() for turn in recent_turns_data],
            'total_turns': len(session.turns),
            'last_activity': session.last_activity.isoformat()
        }
    
    def end_session(self, session_id: str) -> bool:
        """结束会话"""
        session = self.get_session(session_id)
        if session:
            session.is_active = False
            self.logger.info(f"结束会话: {session_id}")
            return True
        return False
    
    def search_conversations(self, user_id: Optional[str] = None,
                           keyword: Optional[str] = None,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: int = 50) -> List[ConversationTurn]:
        """搜索对话记录"""
        results = []
        
        for session in self.sessions.values():
            # 用户过滤
            if user_id and session.user_id != user_id:
                continue
            
            # 时间过滤
            if start_time and session.start_time < start_time:
                continue
            if end_time and session.last_activity > end_time:
                continue
            
            # 搜索轮次
            for turn in session.turns:
                # 关键词过滤
                if keyword:
                    if (keyword.lower() not in turn.user_input.lower() and 
                        keyword.lower() not in turn.agent_response.lower()):
                        continue
                
                results.append(turn)
                
                if len(results) >= limit:
                    return results
        
        return results
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计信息"""
        user_session_ids = self.user_sessions.get(user_id, [])
        user_sessions = [self.sessions[sid] for sid in user_session_ids if sid in self.sessions]
        
        total_turns = sum(len(session.turns) for session in user_sessions)
        active_sessions = sum(1 for session in user_sessions if session.is_active)
        
        return {
            'user_id': user_id,
            'total_sessions': len(user_sessions),
            'active_sessions': active_sessions,
            'total_turns': total_turns,
            'first_session': min((s.start_time for s in user_sessions), default=None),
            'last_activity': max((s.last_activity for s in user_sessions), default=None)
        }
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """导出会话数据"""
        session = self.get_session(session_id)
        return session.to_dict() if session else None
    
    def import_session(self, session_data: Dict[str, Any]) -> bool:
        """导入会话数据"""
        try:
            session = ConversationSession.from_dict(session_data)
            self.sessions[session.session_id] = session
            
            # 更新用户会话列表
            if session.user_id not in self.user_sessions:
                self.user_sessions[session.user_id] = []
            if session.session_id not in self.user_sessions[session.user_id]:
                self.user_sessions[session.user_id].append(session.session_id)
            
            return True
        except Exception as e:
            self.logger.error(f"导入会话失败: {e}")
            return False
    
    def _cleanup_old_sessions(self):
        """清理旧会话"""
        if len(self.sessions) <= self.max_sessions:
            return
        
        # 按最后活动时间排序，移除最旧的会话
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].last_activity
        )
        
        sessions_to_remove = len(self.sessions) - self.max_sessions
        for i in range(sessions_to_remove):
            session_id, session = sorted_sessions[i]
            
            # 从用户会话列表中移除
            if session.user_id in self.user_sessions:
                if session_id in self.user_sessions[session.user_id]:
                    self.user_sessions[session.user_id].remove(session_id)
                if not self.user_sessions[session.user_id]:
                    del self.user_sessions[session.user_id]
            
            # 从会话字典中移除
            del self.sessions[session_id]
            
            self.logger.info(f"清理旧会话: {session_id}")
    
    def clear_user_history(self, user_id: str) -> int:
        """清除用户历史记录"""
        user_session_ids = self.user_sessions.get(user_id, [])
        removed_count = 0
        
        for session_id in user_session_ids:
            if session_id in self.sessions:
                del self.sessions[session_id]
                removed_count += 1
        
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        
        self.logger.info(f"清除用户 {user_id} 的 {removed_count} 个会话")
        return removed_count
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        total_turns = sum(len(session.turns) for session in self.sessions.values())
        active_sessions = sum(1 for session in self.sessions.values() if session.is_active)
        
        return {
            'total_sessions': len(self.sessions),
            'active_sessions': active_sessions,
            'total_users': len(self.user_sessions),
            'total_turns': total_turns,
            'max_sessions': self.max_sessions,
            'max_turns_per_session': self.max_turns_per_session
        }