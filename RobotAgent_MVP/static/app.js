/**
 * RobotAgent MVP 前端应用
 * 实现ASR + 大模型 + TTS的三段式处理
 */

class RobotAgentApp {
    constructor() {
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.currentConversationId = this.generateConversationId();
        this.isConnected = false;
        this.currentMode = 'text'; // 'text' 或 'voice'
        this.voiceStreamingEnabled = false; // 语音流式模式
        this.isSending = false; // 防止重复提交
        
        this.init();
    }

    async init() {
        try {
            this.bindEvents();
            await this.checkConnection();
            await this.loadROS2Actions();
            this.loadConversationHistory();
            this.setupAudioContext();
            
            // 初始化语音流式模式状态
            this.updateVoiceStreamingStatus();
            
            // 显示欢迎消息
            this.showNotification('RobotAgent已就绪', 'success');
            
        } catch (error) {
            console.error('初始化失败:', error);
            this.showNotification('初始化失败，部分功能可能不可用', 'error');
        }
    }

    generateConversationId() {
        return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    bindEvents() {
        // 文本输入事件
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendTextMessage();
            }
        });
        
        sendBtn.addEventListener('click', () => this.sendTextMessage());

        // 模式切换
        document.getElementById('textModeBtn').addEventListener('click', () => {
            this.currentMode = 'text';
            this.voiceStreamingEnabled = false;
            this.switchMode('text');
            this.toggleVoiceStreaming(false);
        });
        
        document.getElementById('voiceModeBtn').addEventListener('click', () => {
            this.currentMode = 'voice';
            this.voiceStreamingEnabled = true;
            this.switchMode('voice');
            this.toggleVoiceStreaming(true);
        });

        // 语音录制
        document.getElementById('recordBtn').addEventListener('click', () => {
            this.toggleRecording();
        });

        // 快捷指令
        document.querySelectorAll('.quick-cmd-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const command = btn.getAttribute('data-command');
                this.sendQuickCommand(command);
            });
        });

        // 聊天控制
        document.getElementById('clearChat').addEventListener('click', () => {
            this.clearChat();
        });
        
        document.getElementById('exportChat').addEventListener('click', () => {
            this.exportChat();
        });
    }

    switchMode(mode) {
        const textArea = document.getElementById('textInputArea');
        const voiceArea = document.getElementById('voiceInputArea');
        const textBtn = document.getElementById('textModeBtn');
        const voiceBtn = document.getElementById('voiceModeBtn');

        if (mode === 'text') {
            textArea.style.display = 'block';
            voiceArea.style.display = 'none';
            textBtn.classList.add('active');
            voiceBtn.classList.remove('active');
        } else {
            textArea.style.display = 'none';
            voiceArea.style.display = 'block';
            textBtn.classList.remove('active');
            voiceBtn.classList.add('active');
        }
    }

    async toggleVoiceStreaming(enabled) {
        try {
            const response = await fetch('/api/toggle_voice_streaming', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ enabled: enabled })
            });
            
            const result = await response.json();
            if (result.success) {
                this.voiceStreamingEnabled = result.data.voice_streaming_enabled;
                this.showNotification(result.data.message, 'success');
            } else {
                this.showNotification('切换语音模式失败', 'error');
            }
        } catch (error) {
            console.error('切换语音流式模式失败:', error);
            this.showNotification('切换语音模式失败', 'error');
        }
    }

    async checkConnection() {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5秒超时
            
            const response = await fetch('/api/status', {
                signal: controller.signal,
                headers: {
                    'Cache-Control': 'no-cache'
                }
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) {
                const data = await response.json();
                this.updateConnectionStatus(true);
                this.updateSystemStatus(data);
                return true;
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.error('连接检查失败:', error);
            this.updateConnectionStatus(false);
            
            if (error.name === 'AbortError') {
                this.showNotification('连接超时', 'warning');
            } else {
                this.showNotification('连接失败', 'error');
            }
            return false;
        }
    }

    updateConnectionStatus(connected) {
        this.isConnected = connected;
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const connectionStatus = document.getElementById('connectionStatus');

        if (connected) {
            statusDot.className = 'status-dot connected';
            statusText.textContent = '已连接';
            connectionStatus.textContent = '已连接';
        } else {
            statusDot.className = 'status-dot disconnected';
            statusText.textContent = '连接失败';
            connectionStatus.textContent = '未连接';
        }
    }

    async sendTextMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message) {
            this.showNotification('请输入消息内容', 'warning');
            return;
        }
        
        // 防止重复提交
        if (this.isProcessing) {
            this.showNotification('正在处理中，请稍候...', 'warning');
            return;
        }
        
        this.isProcessing = true;
        this.showLoading(true);
        
        // 添加用户消息到聊天界面
        this.addMessage('user', message);
        messageInput.value = '';
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: this.currentConversationId
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }
            
            const data = await response.json();
            
            // 验证响应数据格式
            if (!data || typeof data !== 'object') {
                throw new Error('服务器返回的数据格式无效');
            }
            
            // 处理后端返回的ChatResponse格式
            this.handleChatResponse(data);
            
            // 更新对话ID
            if (data.conversation_id) {
                this.currentConversationId = data.conversation_id;
            }
            
        } catch (error) {
            console.error('发送消息失败:', error);
            this.addMessage('bot', `抱歉，发生了错误：${error.message}`);
            this.showNotification(`发送消息失败: ${error.message}`, 'error');
        } finally {
            this.isProcessing = false;
            this.showLoading(false);
        }
    }

    async sendQuickCommand(command) {
        // 防止重复提交
        if (this.isSending) return;
        
        // 设置输入框值
        document.getElementById('messageInput').value = command;
        
        // 直接调用sendTextMessage，它内部已有防重复逻辑
        this.sendTextMessage();
    }

    async setupAudioContext() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.audioStream = stream;
        } catch (error) {
            console.error('无法访问麦克风:', error);
            this.showNotification('无法访问麦克风，语音功能不可用', 'error');
        }
    }

    async toggleRecording() {
        if (!this.isRecording) {
            await this.startRecording();
        } else {
            await this.stopRecording();
        }
    }

    async startRecording() {
        try {
            // 检查浏览器支持
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('浏览器不支持录音功能');
            }
            
            if (!this.audioStream) {
                await this.setupAudioContext();
            }

            this.audioChunks = [];
            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4'
            });
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.processRecording();
            };

            this.mediaRecorder.onerror = (event) => {
                console.error('录音错误:', event.error);
                this.showNotification('录音过程中出现错误', 'error');
                this.stopRecording();
            };

            this.mediaRecorder.start(100); // 每100ms收集一次数据
            this.isRecording = true;
            
            // 更新UI
            const recordBtn = document.getElementById('recordBtn');
            const recordingStatus = document.getElementById('recordingStatus');
            
            recordBtn.classList.add('recording');
            recordBtn.innerHTML = '<i class="fas fa-stop"></i><span>停止录音</span>';
            recordBtn.title = '点击停止录音';
            recordingStatus.style.display = 'flex';

            this.showNotification('开始录音...', 'info');
            
        } catch (error) {
            console.error('开始录音失败:', error);
            let errorMessage = '录音失败';
            
            if (error.name === 'NotAllowedError') {
                errorMessage = '请允许访问麦克风权限';
            } else if (error.name === 'NotFoundError') {
                errorMessage = '未找到麦克风设备';
            } else if (error.name === 'NotSupportedError') {
                errorMessage = '浏览器不支持录音功能';
            } else {
                errorMessage = '录音失败: ' + error.message;
            }
            
            this.showNotification(errorMessage, 'error');
        }
    }

    async stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // 更新UI
            const recordBtn = document.getElementById('recordBtn');
            const recordingStatus = document.getElementById('recordingStatus');
            
            recordBtn.classList.remove('recording');
            recordBtn.innerHTML = '<i class="fas fa-microphone"></i><span>点击录音</span>';
            recordingStatus.style.display = 'none';
        }
    }

    async processRecording() {
        if (!this.audioChunks || this.audioChunks.length === 0) {
            this.showNotification('录音数据为空，请重新录音', 'warning');
            return;
        }

        this.showLoading(true);
        this.showNotification('正在处理语音...', 'info');
        document.getElementById('voiceFeedback').textContent = '正在处理语音...';

        try {
            // 创建音频blob
            const mimeType = this.mediaRecorder.mimeType || 'audio/webm';
            const audioBlob = new Blob(this.audioChunks, { type: mimeType });
            
            // 检查音频文件大小
            if (audioBlob.size === 0) {
                throw new Error('录音文件为空');
            }
            
            if (audioBlob.size > 10 * 1024 * 1024) { // 10MB限制
                throw new Error('录音文件过大，请录制较短的音频');
            }

            const formData = new FormData();
            formData.append('audio', audioBlob, `recording.${mimeType.split('/')[1]}`);
            formData.append('conversation_id', this.currentConversationId);

            const response = await fetch('/api/voice_chat', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                let errorMessage = `HTTP ${response.status}`;
                
                try {
                    const errorData = JSON.parse(errorText);
                    errorMessage = errorData.detail || errorData.message || errorMessage;
                } catch {
                    errorMessage = errorText || errorMessage;
                }
                
                throw new Error(errorMessage);
            }

            const data = await response.json();
            
            // 验证响应数据
            if (!data || typeof data !== 'object') {
                throw new Error('服务器返回的数据格式无效');
            }
            
            // 显示识别的文本
            if (data.recognized_text) {
                document.getElementById('voiceFeedback').textContent = 
                    '识别结果: ' + data.recognized_text;
                this.addMessage('user', data.recognized_text);
            }
            
            this.handleChatResponse(data);
            this.showNotification('语音处理完成', 'success');

        } catch (error) {
            console.error('处理录音失败:', error);
            this.addMessage('bot', `语音处理失败：${error.message}`);
            this.showNotification(`语音处理失败: ${error.message}`, 'error');
            document.getElementById('voiceFeedback').textContent = '处理错误';
        } finally {
            this.showLoading(false);
            // 清理音频数据
            this.audioChunks = [];
        }
    }

    async handleChatResponse(data) {
        console.log('收到聊天响应:', data);
        
        try {
            // 验证响应数据结构
            if (!data || typeof data !== 'object') {
                throw new Error('响应数据格式无效');
            }
            
            // 处理后端ChatResponse格式
            if (data.success) {
                // 显示机器人回复
                const userReply = data.response || '收到您的消息';
                this.addMessage('bot', userReply, data.ros2_command, data.execution_status);
                
                // 播放TTS音频
                if (data.audio_url) {
                    await this.playAudio(data.audio_url).catch(error => {
                        console.error('播放音频失败:', error);
                        this.showNotification('音频播放失败', 'warning');
                    });
                }
                
                // 更新机器人状态
                if (data.ros2_command) {
                    const taskDescription = data.ros2_command.description || data.ros2_command.action || '执行中...';
                    this.updateRobotStatus('执行中: ' + taskDescription);
                } else {
                    this.updateRobotStatus('空闲');
                }
            } else {
                // 处理错误响应
                const errorMessage = data.error || '处理请求时出现错误';
                this.addMessage('bot', errorMessage);
                this.showNotification(errorMessage, 'error');
            }
            
        } catch (error) {
            console.error('处理聊天响应失败:', error);
            this.addMessage('bot', '处理响应时出现错误，请重试');
            this.showNotification(`响应处理错误: ${error.message}`, 'error');
        }
    }
    
    // 简化的响应处理 - 移除复杂的响应类型处理
    // 后端现在返回简单的ChatResponse格式，不需要复杂的分类处理

    addMessage(type, content, ros2Command = null, executionStatus = null) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const time = new Date().toLocaleTimeString();
        
        let messageHTML = `
            <div class="message-avatar">
                <i class="fas ${type === 'user' ? 'fa-user' : 'fa-robot'}"></i>
            </div>
            <div class="message-content">
                <p>${content}</p>
                ${ros2Command ? this.formatROS2Command(ros2Command) : ''}
                ${executionStatus ? `<div class="execution-status">${this.getStatusText(executionStatus)}</div>` : ''}
                <div class="message-time">${time}</div>
            </div>
        `;
        
        messageDiv.innerHTML = messageHTML;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    formatROS2Command(command) {
        if (!command) return '';
        
        return `
            <div class="ros2-command">
                <h5>ROS2 命令</h5>
                <p><strong>动作:</strong> ${command.action || 'N/A'}</p>
                <p><strong>描述:</strong> ${command.description || 'N/A'}</p>
                ${command.parameters ? `<p><strong>参数:</strong> ${JSON.stringify(command.parameters)}</p>` : ''}
            </div>
        `;
    }

    getStatusText(status) {
        const statusMap = {
            'pending': '等待执行',
            'executing': '执行中',
            'completed': '已完成',
            'failed': '执行失败'
        };
        return statusMap[status] || status;
    }

    async playAudio(audioUrl) {
        try {
            const audioPlayer = document.getElementById('audioPlayer');
            audioPlayer.src = audioUrl;
            await audioPlayer.play();
        } catch (error) {
            console.error('播放音频失败:', error);
        }
    }

    updateRobotStatus(task) {
        document.getElementById('currentTask').textContent = task;
    }
    
    updateRobotEmotion(emotion) {
        // 更新机器人情感状态显示
        const emotionElement = document.getElementById('robotEmotion');
        if (emotionElement) {
            emotionElement.textContent = emotion;
        }
    }
    
    // 移除复杂的显示方法 - 后端现在返回简单格式，不需要这些复杂的显示逻辑
    
    addSystemMessage(htmlContent) {
        // 添加系统消息（用于显示分析结果等）
        const messagesContainer = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system-message';
        messageDiv.innerHTML = htmlContent;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    async loadROS2Actions() {
        try {
            const response = await fetch('/api/ros2_actions');
            const actions = await response.json();
            
            const actionsList = document.getElementById('actionsList');
            actionsList.innerHTML = '';

            Object.entries(actions).forEach(([actionName, actionInfo]) => {
                const actionDiv = document.createElement('div');
                actionDiv.className = 'action-item';
                actionDiv.innerHTML = `
                    <div class="action-name">${actionInfo.name}</div>
                    <div class="action-description">${actionInfo.description}</div>
                    <span class="action-category">${actionInfo.category}</span>
                `;
                actionsList.appendChild(actionDiv);
            });
        } catch (error) {
            console.error('加载ROS2动作失败:', error);
            document.getElementById('actionsList').innerHTML = '<div class="loading">加载失败</div>';
        }
    }

    async loadConversationHistory() {
        try {
            const response = await fetch(`/api/conversation/${this.currentConversationId}`);
            const history = await response.json();
            
            // 这里可以加载历史对话
            console.log('对话历史:', history);
        } catch (error) {
            console.error('加载对话历史失败:', error);
        }
    }

    clearChat() {
        const messagesContainer = document.getElementById('chatMessages');
        messagesContainer.innerHTML = `
            <div class="welcome-message">
                <div class="message bot-message">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <p>您好！我是RobotAgent，您的智能机器人助手。我可以帮助您执行各种任务，包括机器人控制、语音交互等。有什么我可以帮助您的吗？</p>
                    </div>
                </div>
            </div>
        `;
        
        // 生成新的对话ID
        this.currentConversationId = this.generateConversationId();
    }

    exportChat() {
        const messages = document.querySelectorAll('.message');
        const chatData = [];
        
        messages.forEach(message => {
            const type = message.classList.contains('user-message') ? 'user' : 'bot';
            const content = message.querySelector('.message-content p').textContent;
            const time = message.querySelector('.message-time')?.textContent || '';
            
            chatData.push({
                type,
                content,
                time
            });
        });

        const dataStr = JSON.stringify(chatData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `chat_export_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
    }

    showLoading(show) {
        const loadingOverlay = document.getElementById('loadingOverlay');
        loadingOverlay.style.display = show ? 'flex' : 'none';
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notificationContainer');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        container.appendChild(notification);
        
        // 自动移除通知
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new RobotAgentApp();
});

// 定期检查连接状态
setInterval(() => {
    if (window.robotApp) {
        window.robotApp.checkConnection();
    }
}, 30000);

// 保存应用实例到全局
window.addEventListener('load', () => {
    if (!window.robotApp) {
        window.robotApp = new RobotAgentApp();
    }
});