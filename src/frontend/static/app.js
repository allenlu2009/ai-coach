/**
 * AI Coach - Pose Analysis System Frontend
 * Handles video upload, pose analysis, coaching feedback, and real-time chat
 */

class AICoachApp {
    constructor() {
        this.currentVideoId = null;
        this.websocket = null;
        this.sessionId = this.generateSessionId();
        this.isProcessing = false;
        
        // API endpoints
        this.API_BASE = '';
        this.WS_BASE = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        
        this.init();
    }
    
    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 16);
    }
    
    init() {
        this.setupEventListeners();
        this.setupWebSocket();
        this.updateConnectionStatus(false);
    }
    
    setupEventListeners() {
        // Upload functionality
        const dropZone = document.getElementById('dropZone');
        const videoInput = document.getElementById('videoInput');
        const browseBtn = document.getElementById('browseBtn');
        const uploadBtn = document.getElementById('uploadBtn');
        
        // Drag and drop
        dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        dropZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        dropZone.addEventListener('drop', this.handleDrop.bind(this));
        dropZone.addEventListener('click', () => videoInput.click());
        
        // File input
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            videoInput.click();
        });
        videoInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Upload button
        uploadBtn.addEventListener('click', this.handleUpload.bind(this));
        
        // Chat functionality
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        
        chatInput.addEventListener('input', this.handleChatInput.bind(this));
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        sendBtn.addEventListener('click', this.sendMessage.bind(this));
        
        // Chat suggestions
        document.querySelectorAll('.suggestion-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const message = e.target.dataset.message;
                this.sendMessage(message);
            });
        });
        
        // Modal controls
        document.getElementById('helpBtn').addEventListener('click', () => {
            this.showModal('helpModal');
        });
        
        // Modal close buttons
        document.querySelectorAll('.close-btn, #errorOkBtn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const modal = e.target.closest('.modal');
                if (modal) this.hideModal(modal.id);
            });
        });
        
        // Download button
        document.getElementById('downloadBtn').addEventListener('click', this.downloadAnalysis.bind(this));
        
        // Click outside modal to close
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideModal(modal.id);
                }
            });
        });
    }
    
    setupWebSocket() {
        try {
            const wsUrl = `${this.WS_BASE}//${window.location.host}/chat/${this.sessionId}`;
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => {
                    if (!this.websocket || this.websocket.readyState === WebSocket.CLOSED) {
                        this.setupWebSocket();
                    }
                }, 3000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusIndicator = document.getElementById('connectionStatus');
        const statusText = document.getElementById('connectionText');
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        
        if (connected) {
            statusIndicator.className = 'status-indicator online';
            statusText.textContent = 'Connected';
            chatInput.disabled = false;
            sendBtn.disabled = false;
        } else {
            statusIndicator.className = 'status-indicator offline';
            statusText.textContent = 'Disconnected';
            chatInput.disabled = true;
            sendBtn.disabled = true;
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'message':
                this.addChatMessage(data.role, data.content);
                break;
            case 'progress_update':
                this.updateProgress(data.data);
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }
    
    // File upload handlers
    handleDragOver(e) {
        e.preventDefault();
        document.getElementById('dropZone').classList.add('drag-over');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('dropZone').classList.remove('drag-over');
    }
    
    handleDrop(e) {
        e.preventDefault();
        document.getElementById('dropZone').classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.selectFile(files[0]);
        }
    }
    
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.selectFile(file);
        }
    }
    
    selectFile(file) {
        // Validate file type
        const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo'];
        if (!allowedTypes.includes(file.type)) {
            this.showError('Please select a valid video file (MP4, AVI, MOV)');
            return;
        }
        
        // Validate file size (100MB max)
        const maxSize = 100 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('File size must be less than 100MB');
            return;
        }
        
        // Update UI
        const dropZoneContent = document.querySelector('.drop-zone-content');
        dropZoneContent.innerHTML = `
            <i class="fas fa-file-video"></i>
            <h3>${file.name}</h3>
            <p>Size: ${this.formatFileSize(file.size)}</p>
            <small>Click "Upload & Analyze" to begin pose analysis</small>
        `;
        
        // Enable upload button
        document.getElementById('uploadBtn').disabled = false;
        this.selectedFile = file;
        
        // Show video preview
        this.showVideoPreview(file);
    }
    
    showVideoPreview(file) {
        const previewSection = document.getElementById('videoPreviewSection');
        const previewVideo = document.getElementById('previewVideo');
        const previewFileName = document.getElementById('previewFileName');
        const previewFileSize = document.getElementById('previewFileSize');
        const previewDuration = document.getElementById('previewDuration');
        
        // Create object URL for the video file
        const videoUrl = URL.createObjectURL(file);
        
        // Update video source
        previewVideo.src = videoUrl;
        
        // Update file info
        previewFileName.textContent = file.name;
        previewFileSize.textContent = this.formatFileSize(file.size);
        
        // Get video duration when metadata loads
        previewVideo.addEventListener('loadedmetadata', () => {
            const duration = previewVideo.duration;
            const minutes = Math.floor(duration / 60);
            const seconds = Math.floor(duration % 60);
            previewDuration.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        });
        
        // Show the preview section
        previewSection.classList.remove('hidden');
        
        // Cleanup old object URLs to prevent memory leaks
        previewVideo.addEventListener('load', () => {
            URL.revokeObjectURL(videoUrl);
        });
    }
    
    async handleUpload() {
        if (!this.selectedFile || this.isProcessing) return;
        
        this.isProcessing = true;
        
        // Reset all analysis elements for new upload
        const video = document.getElementById('analysisVideo');
        const placeholder = document.getElementById('videoPlaceholder');
        if (video && placeholder) {
            video.src = '';
            video.load();
            video.classList.add('hidden');
            placeholder.classList.remove('hidden');
        }
        
        // Reset progress
        document.getElementById('progressFill').style.width = '0%';
        document.getElementById('progressText').textContent = '0%';
        document.getElementById('statusMessage').textContent = 'Preparing analysis...';
        
        // Hide feedback panel and video info
        document.getElementById('feedbackPanel').classList.add('hidden');
        const videoInfo = document.getElementById('videoInfo');
        if (videoInfo) videoInfo.classList.add('hidden');
        
        // Reset download button
        document.getElementById('downloadBtn').disabled = true;
        
        this.showLoadingOverlay('Uploading video...');
        
        try {
            const formData = new FormData();
            formData.append('video_file', this.selectedFile);
            
            const movementType = document.getElementById('movementSelect').value;
            formData.append('movement_type', movementType);
            
            const response = await fetch(`${this.API_BASE}/upload`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.detail || 'Upload failed');
            }
            
            this.currentVideoId = result.video_id;
            this.showAnalysisSection();
            this.hideLoadingOverlay();
            
            // Start polling for results
            this.pollAnalysisResults();
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showError(error.message || 'Failed to upload video');
            this.hideLoadingOverlay();
        } finally {
            this.isProcessing = false;
        }
    }
    
    async pollAnalysisResults() {
        if (!this.currentVideoId) return;
        
        try {
            const response = await fetch(`${this.API_BASE}/analyze/${this.currentVideoId}`);
            const result = await response.json();
            
            if (result.status === 'completed') {
                this.handleAnalysisComplete(result);
                this.loadCoachingFeedback();
            } else if (result.status === 'failed') {
                this.showError(result.error_message || 'Analysis failed');
                this.updateProgress({ status: 'failed', progress: 0, message: 'Analysis failed' });
            } else {
                // Still processing, continue polling
                setTimeout(() => this.pollAnalysisResults(), 2000);
            }
            
        } catch (error) {
            console.error('Error polling analysis results:', error);
            setTimeout(() => this.pollAnalysisResults(), 5000); // Retry after longer delay
        }
    }
    
    handleAnalysisComplete(analysis) {
        // Update video info
        this.updateVideoInfo(analysis.metadata);
        
        // Load processed video
        this.loadProcessedVideo(analysis.video_id);
        
        // Update progress to complete
        this.updateProgress({ 
            status: 'completed', 
            progress: 100, 
            message: 'Analysis complete!',
            poses_detected: analysis.poses_detected_count
        });
    }
    
    updateVideoInfo(metadata) {
        document.getElementById('videoDuration').textContent = `${metadata.duration_seconds.toFixed(1)}s`;
        document.getElementById('videoFPS').textContent = `${metadata.fps.toFixed(1)} FPS`;
        document.getElementById('videoResolution').textContent = `${metadata.resolution_width}x${metadata.resolution_height}`;
        
        document.getElementById('videoInfo').classList.remove('hidden');
    }
    
    async loadProcessedVideo(videoId) {
        try {
            const video = document.getElementById('analysisVideo');
            const placeholder = document.getElementById('videoPlaceholder');
            
            console.log('Loading processed video for ID:', videoId);
            
            // Use the actual processed video for this specific video ID
            const videoUrl = `${this.API_BASE}/videos/${videoId}/preview`;
            console.log('Loading processed video URL:', videoUrl);
            
            // Test if video URL is accessible
            const response = await fetch(videoUrl, { method: 'HEAD' });
            if (!response.ok) {
                throw new Error(`Video not accessible: ${response.status}`);
            }
            
            // Force reload with cache busting and explicit load
            video.src = '';  // Clear first to force reload
            video.load();    // Reset video element
            video.src = `${videoUrl}?t=${Date.now()}`;  // Cache busting
            video.load();    // Force load new source
            
            video.classList.remove('hidden');
            placeholder.classList.add('hidden');
            
            // Add comprehensive error handling and debugging
            video.onerror = function(e) {
                console.error('Video loading error:', e);
                console.error('Failed to load:', video.src);
                console.error('Video error details:', video.error);
            };
            
            video.onloadstart = function() {
                console.log('Video loading started for:', video.src);
            };
            
            video.oncanplay = function() {
                console.log('Video can play:', video.src);
            };
            
            video.onloadeddata = function() {
                console.log('Video data loaded:', video.src);
                console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
            };
            
            video.onstalled = function() {
                console.warn('Video loading stalled:', video.src);
            };
            
            console.log('Video element setup complete. Current src:', video.src);
            
            // Enable download button
            document.getElementById('downloadBtn').disabled = false;
            
            console.log('✅ Video loaded successfully');
            
        } catch (error) {
            console.error('Error loading processed video:', error);
        }
    }
    
    async loadCoachingFeedback() {
        if (!this.currentVideoId) return;
        
        try {
            const movementType = document.getElementById('movementSelect').value;
            const response = await fetch(`${this.API_BASE}/feedback/${this.currentVideoId}?movement_type=${movementType}`);
            const feedback = await response.json();
            
            if (!response.ok) {
                throw new Error(feedback.detail || 'Failed to load feedback');
            }
            
            this.displayCoachingFeedback(feedback);
            
        } catch (error) {
            console.error('Error loading coaching feedback:', error);
            this.showError('Failed to load coaching feedback');
        }
    }
    
    displayCoachingFeedback(feedback) {
        // Summary
        document.getElementById('feedbackSummary').textContent = feedback.analysis_summary;
        
        // Key issues
        const issuesList = document.getElementById('keyIssues');
        issuesList.innerHTML = '';
        feedback.key_issues.forEach(issue => {
            const li = document.createElement('li');
            li.textContent = issue;
            issuesList.appendChild(li);
        });
        
        // Suggestions
        const suggestionsList = document.getElementById('suggestions');
        suggestionsList.innerHTML = '';
        feedback.improvement_suggestions.forEach(suggestion => {
            const li = document.createElement('li');
            li.textContent = suggestion;
            suggestionsList.appendChild(li);
        });
        
        // Technical metrics
        this.displayMetrics(feedback.coaching_metrics);
        
        // Show feedback panel
        document.getElementById('feedbackPanel').classList.remove('hidden');
    }
    
    displayMetrics(metrics) {
        // Map API metric keys to HTML element IDs
        const metricMapping = {
            'movement_smoothness': { bar: 'smoothnessBar', value: 'smoothnessValue' },
            'posture_stability': { bar: 'stabilityBar', value: 'stabilityValue' },
            'balance_score': { bar: 'balanceBar', value: 'balanceValue' },
            'symmetry_score': { bar: 'symmetryBar', value: 'symmetryValue' }
        };
        
        Object.entries(metricMapping).forEach(([key, ids]) => {
            const value = metrics[key];
            if (value !== null && value !== undefined) {
                const percentage = Math.round(value * 100);
                
                const bar = document.getElementById(ids.bar);
                const valueSpan = document.getElementById(ids.value);
                
                if (bar && valueSpan) {
                    bar.style.width = `${percentage}%`;
                    valueSpan.textContent = `${percentage}%`;
                    
                    // Color coding
                    if (percentage >= 80) {
                        bar.style.backgroundColor = 'var(--success-color)';
                    } else if (percentage >= 60) {
                        bar.style.backgroundColor = 'var(--accent-color)';
                    } else {
                        bar.style.backgroundColor = 'var(--error-color)';
                    }
                }
            }
        });
    }
    
    updateProgress(data) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const statusMessage = document.getElementById('statusMessage');
        const posesDetected = document.getElementById('posesDetected');
        
        progressFill.style.width = `${data.progress}%`;
        progressText.textContent = `${Math.round(data.progress)}%`;
        statusMessage.textContent = data.message || 'Processing...';
        
        // Update poses detected if available
        if (data.poses_detected !== undefined) {
            posesDetected.textContent = data.poses_detected;
        }
        
        // Handle status-specific styling
        if (data.status === 'failed') {
            statusMessage.style.color = 'var(--error-color)';
        } else if (data.status === 'completed') {
            statusMessage.style.color = 'var(--success-color)';
        } else {
            statusMessage.style.color = 'var(--gray-600)';
        }
    }
    
    // Chat functionality
    handleChatInput() {
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        
        sendBtn.disabled = !chatInput.value.trim() || !this.websocket || this.websocket.readyState !== WebSocket.OPEN;
    }
    
    sendMessage(message = null) {
        const chatInput = document.getElementById('chatInput');
        const messageText = message || chatInput.value.trim();
        
        if (!messageText || !this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            return;
        }
        
        // Add user message to chat
        this.addChatMessage('user', messageText);
        
        // Send to backend
        const messageData = {
            content: messageText,
            video_id: this.currentVideoId,
            timestamp: new Date().toISOString()
        };
        
        this.websocket.send(JSON.stringify(messageData));
        
        // Clear input
        if (!message) {
            chatInput.value = '';
            this.handleChatInput();
        }
    }
    
    addChatMessage(role, content) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            <div class="message-content">${this.escapeHtml(content)}</div>
            <div class="message-time">${now}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Utility functions
    showAnalysisSection() {
        document.getElementById('analysisSection').classList.remove('hidden');
        document.getElementById('analysisSection').scrollIntoView({ behavior: 'smooth' });
    }
    
    showLoadingOverlay(text) {
        document.getElementById('loadingText').textContent = text;
        document.getElementById('loadingOverlay').classList.remove('hidden');
    }
    
    hideLoadingOverlay() {
        document.getElementById('loadingOverlay').classList.add('hidden');
    }
    
    showModal(modalId) {
        document.getElementById(modalId).classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
    
    hideModal(modalId) {
        document.getElementById(modalId).classList.add('hidden');
        document.body.style.overflow = '';
    }
    
    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        this.showModal('errorModal');
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    downloadAnalysis() {
        if (!this.currentVideoId) return;
        
        // Use the dedicated download endpoint
        const link = document.createElement('a');
        link.href = `${this.API_BASE}/videos/${this.currentVideoId}/download`;
        link.download = `pose_analysis_${this.currentVideoId}.mp4`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AICoachApp();
});