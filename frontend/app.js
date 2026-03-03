/**
 * Hexa-Ops 前端应用
 *
 * 适配 Python FastAPI 后端接口：
 *   POST /chat          → 非流式对话
 *   POST /chat/stream   → SSE 流式对话（event: data: {"type":"token"|"done"|"error"}）
 *   POST /ops/diagnose  → AI Ops 诊断（含 HITL：status=interrupted|done）
 *   POST /ops/approve   → HITL 人工审批（approved: true/false）
 *   POST /knowledge/index → 重建知识库索引
 */
class HexaOpsApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8001';
        this.currentMode = 'quick';
        this.sessionId = this.generateSessionId();
        this.isStreaming = false;
        this.currentChatHistory = [];
        this.chatHistories = this.loadChatHistories();
        this.isCurrentChatFromHistory = false;
        // 当前 ops 诊断的 thread_id（用于 HITL resume）
        this.currentOpsThreadId = null;

        this.initializeElements();
        this.bindEvents();
        this.updateUI();
        this.initMarkdown();
        this.checkAndSetCentered();
        this.renderChatHistory();
    }

    // -------------------------------------------------------------------------
    // Markdown 初始化
    // -------------------------------------------------------------------------
    initMarkdown() {
        const check = () => {
            if (typeof marked !== 'undefined') {
                try {
                    marked.setOptions({ breaks: true, gfm: true, headerIds: false, mangle: false });
                    if (typeof hljs !== 'undefined') {
                        marked.setOptions({
                            highlight: (code, lang) => {
                                if (lang && hljs.getLanguage(lang)) {
                                    try { return hljs.highlight(code, { language: lang }).value; } catch (_) {}
                                }
                                return code;
                            }
                        });
                    }
                } catch (e) { console.error('Markdown init error:', e); }
            } else {
                setTimeout(check, 100);
            }
        };
        check();
    }

    renderMarkdown(content) {
        if (!content) return '';
        if (typeof marked === 'undefined') return this.escapeHtml(content);
        try { return marked.parse(content); } catch (_) { return this.escapeHtml(content); }
    }

    highlightCodeBlocks(container) {
        if (typeof hljs !== 'undefined' && container) {
            container.querySelectorAll('pre code').forEach(block => {
                if (!block.classList.contains('hljs')) hljs.highlightElement(block);
            });
        }
    }

    // -------------------------------------------------------------------------
    // DOM 初始化
    // -------------------------------------------------------------------------
    initializeElements() {
        this.sidebar           = document.querySelector('.sidebar');
        this.newChatBtn        = document.getElementById('newChatBtn');
        this.aiOpsSidebarBtn   = document.getElementById('aiOpsSidebarBtn');
        this.messageInput      = document.getElementById('messageInput');
        this.sendButton        = document.getElementById('sendButton');
        this.toolsBtn          = document.getElementById('toolsBtn');
        this.toolsMenu         = document.getElementById('toolsMenu');
        this.rebuildIndexItem  = document.getElementById('rebuildIndexItem');
        this.modeSelectorBtn   = document.getElementById('modeSelectorBtn');
        this.modeDropdown      = document.getElementById('modeDropdown');
        this.currentModeText   = document.getElementById('currentModeText');
        this.chatMessages      = document.getElementById('chatMessages');
        this.chatContainer     = document.getElementById('chatContainer');
        this.welcomeGreeting   = document.getElementById('welcomeGreeting');
        this.chatHistoryList   = document.getElementById('chatHistoryList');
        this.loadingOverlay    = document.getElementById('loadingOverlay');
        // Modals
        this.opsInputModal     = document.getElementById('opsInputModal');
        this.opsAlertInput     = document.getElementById('opsAlertInput');
        this.opsInputClose     = document.getElementById('opsInputClose');
        this.opsInputCancel    = document.getElementById('opsInputCancel');
        this.opsInputConfirm   = document.getElementById('opsInputConfirm');
        this.hitlModal         = document.getElementById('hitlModal');
        this.hitlReportPreview = document.getElementById('hitlReportPreview');
        this.hitlApprove       = document.getElementById('hitlApprove');
        this.hitlReject        = document.getElementById('hitlReject');
    }

    // -------------------------------------------------------------------------
    // 事件绑定
    // -------------------------------------------------------------------------
    bindEvents() {
        this.newChatBtn?.addEventListener('click', () => this.newChat());
        this.aiOpsSidebarBtn?.addEventListener('click', () => this.openOpsInputModal());
        this.sendButton?.addEventListener('click', () => this.sendMessage());
        this.messageInput?.addEventListener('keypress', e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.sendMessage(); }
        });

        // 工具菜单
        this.toolsBtn?.addEventListener('click', e => { e.stopPropagation(); this.toggleToolsMenu(); });
        this.rebuildIndexItem?.addEventListener('click', () => { this.closeToolsMenu(); this.rebuildIndex(); });
        document.addEventListener('click', e => {
            if (this.toolsBtn && this.toolsMenu &&
                !this.toolsBtn.contains(e.target) && !this.toolsMenu.contains(e.target)) {
                this.closeToolsMenu();
            }
        });

        // 模式选择
        this.modeSelectorBtn?.addEventListener('click', e => { e.stopPropagation(); this.toggleModeDropdown(); });
        document.querySelectorAll('.dropdown-item').forEach(item => {
            item.addEventListener('click', () => {
                this.selectMode(item.getAttribute('data-mode'));
                this.closeModeDropdown();
            });
        });
        document.addEventListener('click', e => {
            if (this.modeSelectorBtn && this.modeDropdown &&
                !this.modeSelectorBtn.contains(e.target) && !this.modeDropdown.contains(e.target)) {
                this.closeModeDropdown();
            }
        });

        // AI Ops 输入 Modal
        this.opsInputClose?.addEventListener('click', () => this.closeOpsInputModal());
        this.opsInputCancel?.addEventListener('click', () => this.closeOpsInputModal());
        this.opsInputModal?.addEventListener('click', e => {
            if (e.target === this.opsInputModal) this.closeOpsInputModal();
        });
        this.opsInputConfirm?.addEventListener('click', () => this.startOpsDiagnose());
        this.opsAlertInput?.addEventListener('keydown', e => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) this.startOpsDiagnose();
        });

        // HITL 审批 Modal
        this.hitlApprove?.addEventListener('click', () => this.submitApproval(true));
        this.hitlReject?.addEventListener('click', () => this.submitApproval(false));
    }

    // -------------------------------------------------------------------------
    // 模式 / 工具菜单
    // -------------------------------------------------------------------------
    toggleToolsMenu() {
        this.toolsBtn?.closest('.tools-btn-wrapper')?.classList.toggle('active');
    }
    closeToolsMenu() {
        this.toolsBtn?.closest('.tools-btn-wrapper')?.classList.remove('active');
    }
    toggleModeDropdown() {
        this.modeSelectorBtn?.closest('.mode-selector-wrapper')?.classList.toggle('active');
    }
    closeModeDropdown() {
        this.modeSelectorBtn?.closest('.mode-selector-wrapper')?.classList.remove('active');
    }
    selectMode(mode) {
        if (this.isStreaming) { this.showNotification('请等待当前对话完成后再切换', 'warning'); return; }
        this.currentMode = mode;
        this.updateUI();
        this.showNotification(`已切换到${mode === 'quick' ? '快速' : '流式'}模式`, 'info');
    }

    updateUI() {
        if (this.currentModeText) {
            this.currentModeText.textContent = this.currentMode === 'quick' ? '快速' : '流式';
        }
        document.querySelectorAll('.dropdown-item').forEach(item => {
            item.classList.toggle('active', item.getAttribute('data-mode') === this.currentMode);
        });
        if (this.sendButton) this.sendButton.disabled = this.isStreaming;
        if (this.messageInput) this.messageInput.disabled = this.isStreaming;
    }

    generateSessionId() {
        return 'sess_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    // -------------------------------------------------------------------------
    // 对话（普通）
    // -------------------------------------------------------------------------
    async sendMessage() {
        const message = this.messageInput?.value.trim();
        if (!message) { this.showNotification('请输入消息内容', 'warning'); return; }
        if (this.isStreaming) { this.showNotification('请等待当前对话完成', 'warning'); return; }

        this.addMessage('user', message);
        if (this.messageInput) this.messageInput.value = '';
        this.isStreaming = true;
        this.updateUI();

        try {
            if (this.currentMode === 'quick') {
                await this.sendQuickMessage(message);
            } else {
                await this.sendStreamMessage(message);
            }
        } catch (error) {
            console.error('发送失败:', error);
            this.addMessage('assistant', '抱歉，发送消息时出错：' + error.message);
        } finally {
            this.isStreaming = false;
            this.updateUI();
            if (this.isCurrentChatFromHistory && this.currentChatHistory.length > 0) {
                this.updateCurrentChatHistory();
                this.renderChatHistory();
            }
        }
    }

    async sendQuickMessage(message) {
        const resp = await fetch(`${this.apiBaseUrl}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: this.sessionId }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        this.addMessage('assistant', data.answer || '（无回答）');
    }

    async sendStreamMessage(message) {
        const resp = await fetch(`${this.apiBaseUrl}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: this.sessionId }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        const el = this.addMessage('assistant', '', true);
        let fullAnswer = '';
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const evt = JSON.parse(line.slice(6));
                        if (evt.type === 'token') {
                            fullAnswer += evt.content || '';
                            const mc = el.querySelector('.message-content');
                            if (mc) mc.textContent = fullAnswer;
                            this.scrollToBottom();
                        } else if (evt.type === 'done') {
                            const mc = el.querySelector('.message-content');
                            if (mc) { mc.innerHTML = this.renderMarkdown(fullAnswer); this.highlightCodeBlocks(mc); }
                            el.classList.remove('streaming');
                            if (fullAnswer) {
                                this.currentChatHistory.push({ type: 'assistant', content: fullAnswer, timestamp: new Date().toISOString() });
                            }
                            return;
                        } else if (evt.type === 'error') {
                            throw new Error(evt.message);
                        }
                    } catch (_) {}
                }
            }
            // 流自然结束
            const mc = el.querySelector('.message-content');
            if (mc && fullAnswer) { mc.innerHTML = this.renderMarkdown(fullAnswer); this.highlightCodeBlocks(mc); }
            el.classList.remove('streaming');
            if (fullAnswer) {
                this.currentChatHistory.push({ type: 'assistant', content: fullAnswer, timestamp: new Date().toISOString() });
            }
        } finally {
            reader.releaseLock();
        }
    }

    // -------------------------------------------------------------------------
    // AI Ops 诊断
    // -------------------------------------------------------------------------
    openOpsInputModal() {
        if (this.isStreaming) { this.showNotification('请等待当前操作完成', 'warning'); return; }
        if (this.opsInputModal) {
            this.opsInputModal.style.display = 'flex';
            this.opsAlertInput?.focus();
        }
    }

    closeOpsInputModal() {
        if (this.opsInputModal) this.opsInputModal.style.display = 'none';
    }

    async startOpsDiagnose() {
        const alertInput = this.opsAlertInput?.value.trim();
        if (!alertInput) { this.showNotification('请输入告警描述', 'warning'); return; }

        this.closeOpsInputModal();
        this.newChat();

        // 展示用户输入
        this.addMessage('user', `[AI Ops 诊断] ${alertInput}`);

        // 展示加载消息
        const loadingEl = this.addLoadingMessage('正在进行多维度根因分析...');
        this.isStreaming = true;
        this.updateUI();

        try {
            const threadId = this.generateSessionId();
            this.currentOpsThreadId = threadId;

            const resp = await fetch(`${this.apiBaseUrl}/ops/diagnose`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ alert_input: alertInput, session_id: threadId }),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();

            if (data.status === 'interrupted') {
                // HITL: 高危操作，弹出审批框
                this.updateOpsMessage(loadingEl, data.diagnosis_report || '', data, 'pending');
                this.showHitlModal(data.diagnosis_report || '', threadId);
            } else {
                // 正常完成
                this.updateOpsMessage(loadingEl, data.diagnosis_report || '', data, 'done');
                this.currentOpsThreadId = null;
            }
        } catch (error) {
            console.error('Ops 诊断失败:', error);
            const mc = loadingEl?.querySelector('.message-content');
            if (mc) mc.textContent = '诊断过程出错：' + error.message;
        } finally {
            this.isStreaming = false;
            this.updateUI();
        }
    }

    // -------------------------------------------------------------------------
    // HITL 人工审批
    // -------------------------------------------------------------------------
    showHitlModal(diagnosisReport, threadId) {
        this.currentOpsThreadId = threadId;
        if (this.hitlReportPreview) {
            // 只展示前 600 字
            this.hitlReportPreview.textContent = diagnosisReport.slice(0, 600) + (diagnosisReport.length > 600 ? '\n...' : '');
        }
        if (this.hitlModal) this.hitlModal.style.display = 'flex';
    }

    closeHitlModal() {
        if (this.hitlModal) this.hitlModal.style.display = 'none';
    }

    async submitApproval(approved) {
        this.closeHitlModal();
        const threadId = this.currentOpsThreadId;
        if (!threadId) return;

        const loadingEl = this.addLoadingMessage(approved ? '审批已通过，正在生成最终报告...' : '已拒绝，正在记录...');
        this.isStreaming = true;
        this.updateUI();

        try {
            const resp = await fetch(`${this.apiBaseUrl}/ops/approve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ thread_id: threadId, approved }),
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            this.updateOpsMessage(loadingEl, data.diagnosis_report || '', data, approved ? 'approved' : 'rejected');
        } catch (error) {
            const mc = loadingEl?.querySelector('.message-content');
            if (mc) mc.textContent = '审批请求失败：' + error.message;
        } finally {
            this.isStreaming = false;
            this.currentOpsThreadId = null;
            this.updateUI();
        }
    }

    // -------------------------------------------------------------------------
    // 重建知识库索引
    // -------------------------------------------------------------------------
    async rebuildIndex() {
        if (this.isStreaming) { this.showNotification('请等待当前操作完成', 'warning'); return; }
        this.isStreaming = true;
        this.updateUI();
        this.showNotification('正在重建知识库索引，请稍候...', 'info');

        try {
            const resp = await fetch(`${this.apiBaseUrl}/knowledge/index`, { method: 'POST' });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            this.showNotification(`索引重建完成：${JSON.stringify(data)}`, 'success');
        } catch (error) {
            this.showNotification('索引重建失败：' + error.message, 'error');
        } finally {
            this.isStreaming = false;
            this.updateUI();
        }
    }

    // -------------------------------------------------------------------------
    // 消息 UI 帮助方法
    // -------------------------------------------------------------------------
    /**
     * 渲染 Ops 诊断结果（含日志/指标折叠 + 主报告）
     * approvalStatus: 'done' | 'pending' | 'approved' | 'rejected'
     */
    updateOpsMessage(el, report, data, approvalStatus = 'done') {
        if (!el) return;
        el.classList.add('aiops-message');
        const wrapper = el.querySelector('.message-content-wrapper');
        if (!wrapper) return;
        const mc = el.querySelector('.message-content');
        if (!mc) return;

        mc.classList.remove('loading-message-content');
        mc.textContent = '';

        // 折叠详情（日志/指标）
        const details = [];
        if (data.log_summary)     details.push({ label: '日志分析', content: data.log_summary });
        if (data.metrics_summary) details.push({ label: '指标分析', content: data.metrics_summary });
        if (data.risk_level)      details.push({ label: `风险等级：${data.risk_level === 'high' ? '⚠️ HIGH' : '✅ LOW'}`, content: '' });

        if (details.length > 0) {
            let detailsContainer = el.querySelector('.aiops-details');
            if (!detailsContainer) {
                detailsContainer = document.createElement('div');
                detailsContainer.className = 'aiops-details';
                wrapper.insertBefore(detailsContainer, mc);
            } else {
                detailsContainer.innerHTML = '';
            }

            const toggle = document.createElement('div');
            toggle.className = 'details-toggle';
            toggle.innerHTML = `
                <svg class="toggle-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 18L15 12L9 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>查看分析详情（${details.length} 项）</span>`;

            const content = document.createElement('div');
            content.className = 'details-content';
            details.forEach(d => {
                const item = document.createElement('div');
                item.className = 'detail-item';
                item.innerHTML = `<strong>${this.escapeHtml(d.label)}</strong>${d.content ? '：' + this.escapeHtml(d.content) : ''}`;
                content.appendChild(item);
            });

            toggle.addEventListener('click', () => {
                content.classList.toggle('expanded');
                toggle.classList.toggle('expanded');
            });
            detailsContainer.appendChild(toggle);
            detailsContainer.appendChild(content);
        }

        // 主报告 Markdown 渲染
        mc.innerHTML = this.renderMarkdown(report);
        this.highlightCodeBlocks(mc);

        // 审批状态徽章
        if (approvalStatus === 'approved') {
            const badge = document.createElement('div');
            badge.className = 'approval-badge approved';
            badge.innerHTML = '✅ 人工已批准';
            mc.appendChild(badge);
        } else if (approvalStatus === 'rejected') {
            const badge = document.createElement('div');
            badge.className = 'approval-badge rejected';
            badge.innerHTML = '⛔ 人工已拒绝';
            mc.appendChild(badge);
        } else if (approvalStatus === 'pending') {
            const badge = document.createElement('div');
            badge.className = 'approval-badge';
            badge.style.cssText = 'background:rgba(251,188,4,0.12);color:#f9a825;';
            badge.innerHTML = '⏳ 等待人工审批...';
            mc.appendChild(badge);
        }

        // 保存到历史
        this.currentChatHistory.push({ type: 'assistant', content: report, timestamp: new Date().toISOString() });
        this.scrollToBottom();
    }

    addMessage(type, content, isStreaming = false, saveToHistory = true) {
        const isFirstMessage = this.chatMessages?.querySelectorAll('.message').length === 0;
        if (!isStreaming && saveToHistory && content) {
            this.currentChatHistory.push({ type, content, timestamp: new Date().toISOString() });
        }

        const div = document.createElement('div');
        div.className = `message ${type}${isStreaming ? ' streaming' : ''}`;

        if (type === 'assistant') {
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="white"/>
            </svg>`;
            div.appendChild(avatar);
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'message-content-wrapper';
        const mc = document.createElement('div');
        mc.className = 'message-content';

        if (type === 'assistant' && !isStreaming) {
            mc.innerHTML = this.renderMarkdown(content);
            this.highlightCodeBlocks(mc);
        } else {
            mc.textContent = content;
        }
        wrapper.appendChild(mc);
        div.appendChild(wrapper);

        if (this.chatMessages) {
            this.chatMessages.appendChild(div);
            if (isFirstMessage && this.chatContainer) {
                this.chatContainer.classList.remove('centered');
                this.chatContainer.style.transition = 'all 0.5s ease';
            }
            this.scrollToBottom();
        }
        return div;
    }

    addLoadingMessage(text = '分析中...') {
        const div = document.createElement('div');
        div.className = 'message assistant';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="white"/>
        </svg>`;
        div.appendChild(avatar);

        const wrapper = document.createElement('div');
        wrapper.className = 'message-content-wrapper';
        const mc = document.createElement('div');
        mc.className = 'message-content loading-message-content';

        const textSpan = document.createElement('span');
        textSpan.textContent = text;
        const spinner = document.createElement('span');
        spinner.className = 'loading-spinner-icon';
        spinner.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z" fill="currentColor" opacity="0.2"/>
            <path d="M12 2C6.48 2 2 6.48 2 12c0 1.54.36 3 1 4.28l2.6-1.5C15.62 13.64 16 12.84 16 12c0-4.41-3.59-8-8-8z" fill="currentColor"/>
        </svg>`;
        mc.appendChild(textSpan);
        mc.appendChild(spinner);
        wrapper.appendChild(mc);
        div.appendChild(wrapper);

        if (this.chatMessages) {
            this.chatMessages.appendChild(div);
            const isFirst = this.chatMessages.querySelectorAll('.message').length === 1;
            if (isFirst && this.chatContainer) {
                this.chatContainer.classList.remove('centered');
                this.chatContainer.style.transition = 'all 0.5s ease';
            }
            this.scrollToBottom();
        }
        return div;
    }

    checkAndSetCentered() {
        if (this.chatMessages && this.chatContainer) {
            const hasMessages = this.chatMessages.querySelectorAll('.message').length > 0;
            this.chatContainer.classList.toggle('centered', !hasMessages);
        }
    }

    scrollToBottom() {
        if (this.chatMessages) this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    escapeHtml(text) {
        const d = document.createElement('div');
        d.textContent = text;
        return d.innerHTML;
    }

    // -------------------------------------------------------------------------
    // 通知
    // -------------------------------------------------------------------------
    showNotification(message, type = 'info') {
        const colors = { info: '#1a73e8', success: '#34a853', warning: '#fbbc04', error: '#ea4335' };
        const n = document.createElement('div');
        n.className = `notification ${type}`;
        n.textContent = message;
        Object.assign(n.style, {
            position: 'fixed', top: '20px', right: '20px',
            padding: '12px 18px', borderRadius: '10px',
            color: type === 'warning' ? '#202124' : 'white',
            fontWeight: '500', zIndex: '50000',
            animation: 'slideIn 0.3s ease',
            maxWidth: '320px', boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            backgroundColor: colors[type] || colors.info,
        });
        document.body.appendChild(n);
        setTimeout(() => {
            n.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => n.parentNode?.removeChild(n), 300);
        }, 3000);
    }

    // -------------------------------------------------------------------------
    // 对话历史管理
    // -------------------------------------------------------------------------
    newChat() {
        if (this.isStreaming) { this.showNotification('请等待当前对话完成后再新建', 'warning'); return; }
        if (this.currentChatHistory.length > 0) {
            if (this.isCurrentChatFromHistory) this.updateCurrentChatHistory();
            else this.saveCurrentChat();
        }
        this.isStreaming = false;
        if (this.messageInput) this.messageInput.value = '';
        this.currentChatHistory = [];
        this.isCurrentChatFromHistory = false;
        if (this.chatMessages) this.chatMessages.innerHTML = '';
        this.sessionId = this.generateSessionId();
        this.currentMode = 'quick';
        this.updateUI();
        this.checkAndSetCentered();
        this.renderChatHistory();
    }

    saveCurrentChat() {
        if (!this.currentChatHistory.length) return;
        if (this.chatHistories.some(h => h.id === this.sessionId)) { this.updateCurrentChatHistory(); return; }
        const first = this.currentChatHistory.find(m => m.type === 'user');
        const title = first ? first.content.substring(0, 30) + (first.content.length > 30 ? '...' : '') : '新对话';
        this.chatHistories.unshift({ id: this.sessionId, title, messages: [...this.currentChatHistory], createdAt: new Date().toISOString(), updatedAt: new Date().toISOString() });
        if (this.chatHistories.length > 50) this.chatHistories = this.chatHistories.slice(0, 50);
        this.saveChatHistories();
    }

    updateCurrentChatHistory() {
        if (!this.currentChatHistory.length) return;
        const idx = this.chatHistories.findIndex(h => h.id === this.sessionId);
        if (idx === -1) { this.saveCurrentChat(); return; }
        const h = this.chatHistories[idx];
        h.messages = [...this.currentChatHistory];
        h.updatedAt = new Date().toISOString();
        const first = this.currentChatHistory.find(m => m.type === 'user');
        if (first) h.title = first.content.substring(0, 30) + (first.content.length > 30 ? '...' : '');
        this.saveChatHistories();
    }

    loadChatHistories() {
        try { return JSON.parse(localStorage.getItem('hexaops_chatHistories') || '[]'); }
        catch (_) { return []; }
    }

    saveChatHistories() {
        try { localStorage.setItem('hexaops_chatHistories', JSON.stringify(this.chatHistories)); }
        catch (_) {}
    }

    renderChatHistory() {
        if (!this.chatHistoryList) return;
        this.chatHistoryList.innerHTML = '';
        this.chatHistories.forEach(history => {
            const item = document.createElement('div');
            item.className = 'history-item';
            item.dataset.historyId = history.id;
            item.innerHTML = `
                <div class="history-item-content">
                    <span class="history-item-title">${this.escapeHtml(history.title)}</span>
                </div>
                <button class="history-item-delete" data-history-id="${history.id}" title="删除">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </button>`;
            item.addEventListener('click', e => {
                if (!e.target.closest('.history-item-delete')) this.loadChatHistory(history.id);
            });
            item.querySelector('.history-item-delete').addEventListener('click', e => {
                e.stopPropagation();
                this.deleteChatHistory(history.id);
            });
            this.chatHistoryList.appendChild(item);
        });
    }

    loadChatHistory(id) {
        const history = this.chatHistories.find(h => h.id === id);
        if (!history) return;
        if (this.currentChatHistory.length > 0 && this.sessionId !== id) {
            if (this.isCurrentChatFromHistory) this.updateCurrentChatHistory();
            else this.saveCurrentChat();
        }
        this.sessionId = history.id;
        this.currentChatHistory = [...history.messages];
        this.isCurrentChatFromHistory = true;
        if (this.chatMessages) {
            this.chatMessages.innerHTML = '';
            history.messages.forEach(msg => this.addMessage(msg.type, msg.content, false, false));
        }
        this.checkAndSetCentered();
        this.renderChatHistory();
    }

    deleteChatHistory(id) {
        this.chatHistories = this.chatHistories.filter(h => h.id !== id);
        this.saveChatHistories();
        this.renderChatHistory();
        if (this.sessionId === id) {
            this.currentChatHistory = [];
            if (this.chatMessages) this.chatMessages.innerHTML = '';
            this.sessionId = this.generateSessionId();
            this.checkAndSetCentered();
        }
    }
}

// 启动
document.addEventListener('DOMContentLoaded', () => { new HexaOpsApp(); });
