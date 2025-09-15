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
        this.setup3dVisualization();
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
        
        // Check and enable 3D visualization if available
        this.check3dVisualizationAvailability(analysis.video_id);
        
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
            
            if (!video) {
                throw new Error('Video element not found');
            }
            
            console.log('Loading processed video for ID:', videoId);
            console.log('Video element found:', video);
            console.log('Placeholder element found:', placeholder);
            
            // Use the actual processed video for this specific video ID
            const videoUrl = `${this.API_BASE}/videos/${videoId}/preview`;
            console.log('Loading processed video URL:', videoUrl);
            
            // Test if video URL is accessible with retry mechanism
            let response;
            let retries = 0;
            const maxRetries = 30;  // Increased to 30 for longer video processing times
            
            while (retries < maxRetries) {
                response = await fetch(videoUrl, { method: 'HEAD' });
                if (response.ok) {
                    console.log('✅ HEAD request successful');
                    break;
                }
                
                retries++;
                console.log(`🔄 HEAD request failed (${response.status}), retrying... (${retries}/${maxRetries})`);
                
                if (retries < maxRetries) {
                    // Wait 5 seconds before retry (increased for longer video processing)
                    await new Promise(resolve => setTimeout(resolve, 5000));
                } else {
                    throw new Error(`Video not accessible after ${maxRetries} retries: ${response.status}`);
                }
            }
            
            // Clear any existing event handlers
            video.onerror = null;
            video.onloadstart = null;
            video.oncanplay = null;
            video.onloadeddata = null;
            video.onstalled = null;
            
            // Set up event handlers BEFORE setting source
            video.onerror = function(e) {
                console.error('❌ Video loading error:', e);
                console.error('Failed to load:', video.src);
                console.error('Video error details:', video.error);
                if (video.error) {
                    console.error('Error code:', video.error.code);
                    console.error('Error message:', video.error.message);
                }
                
                // Show fallback message with option to open video in new tab
                this.showVideoFallback(videoUrl, videoId);
            }.bind(this);
            
            video.onloadstart = function() {
                console.log('🔄 Video loading started for:', video.src);
            };
            
            video.oncanplay = function() {
                console.log('✅ Video can play:', video.src);
                console.log('Video ready state:', video.readyState);
                console.log('Video network state:', video.networkState);
            };
            
            video.onloadeddata = function() {
                console.log('✅ Video data loaded:', video.src);
                console.log('Video dimensions:', video.videoWidth, 'x', video.videoHeight);
                console.log('Video duration:', video.duration);
            };
            
            video.onstalled = function() {
                console.warn('⚠️ Video loading stalled:', video.src);
            };
            
            video.onloadedmetadata = function() {
                console.log('📊 Video metadata loaded');
                console.log('Duration:', video.duration);
                console.log('Dimensions:', video.videoWidth, 'x', video.videoHeight);
            };
            
            // Force clear and reload
            console.log('🔄 Clearing video source...');
            video.removeAttribute('src');
            video.load();
            
            // Set new source with cache busting
            const cacheBustedUrl = `${videoUrl}?t=${Date.now()}`;
            console.log('🎥 Setting video source to:', cacheBustedUrl);
            video.src = cacheBustedUrl;
            
            // Show video, hide placeholder
            console.log('👁️ Making video visible...');
            video.classList.remove('hidden');
            video.style.display = 'block';  // Force display
            
            if (placeholder) {
                placeholder.classList.add('hidden');
                placeholder.style.display = 'none';  // Force hide
            }
            
            // Force load
            video.load();
            
            console.log('📋 Video element current state:');
            console.log('- src:', video.src);
            console.log('- readyState:', video.readyState);
            console.log('- networkState:', video.networkState);
            console.log('- classList:', video.classList.toString());
            console.log('- style.display:', video.style.display);
            console.log('- offsetWidth:', video.offsetWidth);
            console.log('- offsetHeight:', video.offsetHeight);
            
            // Enable download button
            document.getElementById('downloadBtn').disabled = false;
            
            console.log('✅ Video setup complete');
            
        } catch (error) {
            console.error('❌ Error loading processed video:', error);
        }
    }
    
    showVideoFallback(videoUrl, videoId) {
        console.log('🔄 Showing video fallback options');
        
        const video = document.getElementById('analysisVideo');
        const placeholder = document.getElementById('videoPlaceholder');
        
        // Hide broken video element, show placeholder with fallback options
        video.style.display = 'none';
        if (placeholder) {
            placeholder.style.display = 'block';
            placeholder.classList.remove('hidden');
            placeholder.innerHTML = `
                <div style="text-align: center; padding: 20px;">
                    <i class="fas fa-exclamation-triangle" style="color: #f39c12; font-size: 48px; margin-bottom: 15px;"></i>
                    <h4 style="color: #e74c3c; margin-bottom: 15px;">Video Display Issue</h4>
                    <p style="margin-bottom: 20px;">The video element cannot play the analysis video due to browser compatibility issues.</p>
                    <div style="margin-bottom: 15px;">
                        <button id="openVideoTab" class="btn btn-primary" style="margin-right: 10px;">
                            <i class="fas fa-external-link-alt"></i> Open Video in New Tab
                        </button>
                        <button id="retryVideo" class="btn btn-secondary">
                            <i class="fas fa-redo"></i> Retry Loading
                        </button>
                    </div>
                    <small style="color: #7f8c8d;">
                        The video works correctly in a separate tab. This is a known browser limitation with certain video formats.
                    </small>
                </div>
            `;
            
            // Add event listeners for fallback buttons
            document.getElementById('openVideoTab').addEventListener('click', () => {
                console.log('🔗 Opening video in new tab:', videoUrl);
                window.open(videoUrl, '_blank');
            });
            
            document.getElementById('retryVideo').addEventListener('click', () => {
                console.log('🔄 Retrying video load');
                location.reload(); // Simple retry by reloading page
            });
        }
        
        // Still enable download button since that works
        const downloadBtn = document.getElementById('downloadBtn');
        if (downloadBtn) {
            downloadBtn.disabled = false;
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
    
    // 3D Visualization Tab Management
    setup3dVisualization() {
        // Tab switching functionality
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });
        
        // 3D visualization controls
        const prevBtn = document.getElementById('viz3dPrevBtn');
        const nextBtn = document.getElementById('viz3dNextBtn');
        const slider = document.getElementById('viz3dSlider');
        
        if (prevBtn) prevBtn.addEventListener('click', () => this.navigate3dFrame(-1));
        if (nextBtn) nextBtn.addEventListener('click', () => this.navigate3dFrame(1));
        if (slider) slider.addEventListener('input', (e) => this.set3dFrame(parseInt(e.target.value)));
    }
    
    switchTab(tabId) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });
        
        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === tabId);
        });
        
        // Load 3D visualizations if switching to 3D tab
        if (tabId === '3d-visualization' && this.currentVideoId) {
            this.load3dVisualizations();
        }
    }
    
    async load3dVisualizations() {
        if (!this.currentVideoId) return;
        
        try {
            const response = await fetch(`${this.API_BASE}/videos/${this.currentVideoId}/3d-visualizations`);
            const data = await response.json();
            
            if (data.count > 0) {
                this.viz3dFrames = data.frames;
                this.currentViz3dFrame = 0;
                this.show3dGallery();
                this.update3dVisualization();
            } else {
                this.show3dPlaceholder();
            }
        } catch (error) {
            console.error('Error loading 3D visualizations:', error);
            this.show3dPlaceholder();
        }
    }
    
    show3dGallery() {
        const placeholder = document.getElementById('viz3dPlaceholder');
        const gallery = document.getElementById('viz3dGallery');
        
        if (placeholder) placeholder.classList.add('hidden');
        if (gallery) gallery.classList.remove('hidden');
    }
    
    show3dPlaceholder() {
        const placeholder = document.getElementById('viz3dPlaceholder');
        const gallery = document.getElementById('viz3dGallery');
        
        if (placeholder) placeholder.classList.remove('hidden');
        if (gallery) gallery.classList.add('hidden');
    }
    
    navigate3dFrame(direction) {
        if (!this.viz3dFrames || this.viz3dFrames.length === 0) return;
        
        this.currentViz3dFrame += direction;
        this.currentViz3dFrame = Math.max(0, Math.min(this.viz3dFrames.length - 1, this.currentViz3dFrame));
        this.update3dVisualization();
    }
    
    set3dFrame(frameIndex) {
        if (!this.viz3dFrames || this.viz3dFrames.length === 0) return;
        
        this.currentViz3dFrame = Math.max(0, Math.min(this.viz3dFrames.length - 1, frameIndex));
        this.update3dVisualization();
    }
    
    update3dVisualization() {
        if (!this.viz3dFrames || this.viz3dFrames.length === 0) return;
        
        const frame = this.viz3dFrames[this.currentViz3dFrame];
        const image = document.getElementById('viz3dImage');
        const frameInfo = document.getElementById('viz3dFrameInfo');
        const slider = document.getElementById('viz3dSlider');
        
        if (image) {
            image.src = `${this.API_BASE}${frame.url}`;
            image.alt = `3D Pose Visualization - Frame ${frame.frame_number}`;
        }
        
        if (frameInfo) {
            frameInfo.textContent = `Frame ${this.currentViz3dFrame + 1} of ${this.viz3dFrames.length}`;
        }
        
        if (slider) {
            slider.max = this.viz3dFrames.length - 1;
            slider.value = this.currentViz3dFrame;
        }
    }

    // Check if 3D visualization is available and enable the tab
    async check3dVisualizationAvailability(videoId) {
        console.log(`🔍 Checking 3D visualization for video ID: ${videoId}`);
        try {
            const url = `${this.API_BASE}/videos/${videoId}/3d-visualizations`;
            console.log(`🔍 Fetching: ${url}`);
            const response = await fetch(url);
            console.log(`🔍 Response status: ${response.status}`);
            if (response.ok) {
                const data = await response.json();
                console.log(`🔍 Response data:`, data);
                if (data.frames && data.frames.length > 0) {
                    // 3D poses are available - show "Open 3D Visualization" button
                    console.log(`✅ 3D visualization available with ${data.frames.length} frames - showing button`);
                    this.show3dVisualizationButton(videoId, data.frames.length);
                } else {
                    console.log('❌ 3D visualization not available - no frames found');
                }
            } else {
                console.log(`❌ 3D visualization endpoint returned: ${response.status}`);
            }
        } catch (error) {
            console.log('❌ 3D visualization check failed:', error);
        }
    }

    // Show "Open 3D Visualization" button
    show3dVisualizationButton(videoId, frameCount) {
        console.log('🎯 Showing 3D visualization button...');
        
        // Find the video controls area
        const videoControls = document.querySelector('.video-controls');
        if (videoControls) {
            // Remove any existing 3D visualization buttons to prevent duplicates
            const existingButtons = videoControls.querySelectorAll('.viz3d-btn');
            existingButtons.forEach(btn => btn.remove());
            
            // Create the 3D visualization button
            const button3d = document.createElement('button');
            button3d.className = 'control-btn viz3d-btn';
            button3d.innerHTML = `
                <i class="fas fa-cube"></i>
                Open 3D Visualization (${frameCount} frames)
            `;
            button3d.style.marginLeft = '10px';
            
            // Add click handler to open new window
            button3d.addEventListener('click', () => {
                this.open3dVisualizationWindow(videoId);
            });
            
            // Add the button to controls
            videoControls.appendChild(button3d);
            console.log(`✅ 3D visualization button added with ${frameCount} frames`);
        } else {
            console.log('❌ Video controls area not found');
        }
    }

    // Open 3D visualization in new window
    open3dVisualizationWindow(videoId) {
        const url = `${window.location.origin}/3d-viewer/${videoId}`;
        const windowFeatures = 'width=1200,height=800,scrollbars=yes,resizable=yes';
        window.open(url, '_blank', windowFeatures);
        console.log(`🚀 Opened 3D visualization window for video: ${videoId}`);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AICoachApp();
});