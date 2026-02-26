// Configuration
const API_URL = `${window.location.protocol}//${window.location.host}`;
let activeJobId = null;
let statusPolling = null;
let detectionTimeline = [];  // Array of {timestamp_ms, detections, frame_number}
let videoElement = null;
let animationFrame = null;
let lastDrawnIndex = -1;
const MAX_TIMELINE_SIZE = 300; 
let isPlaybackReady = false;
const BUFFER_TARGET_MS = 3000; // Buffer 3 seconds of detections before playing, should be adaptable in production-level.
let detectionFpsInterval = null;
// Debug counters
let debugStats = {
    lastFrameTime: performance.now(),
};


// DOM Elements
const elements = {
    apiStatus: document.getElementById('api-status'),
    imageForm: document.getElementById('image-form'),
    imageInput: document.getElementById('image-input'),
    imageSubmit: document.getElementById('image-submit'),
    videoForm: document.getElementById('video-form'),
    videoInput: document.getElementById('video-input'),
    videoSubmit: document.getElementById('video-submit'),
    overlayCanvas: document.getElementById('overlay-canvas'),
    loading: document.getElementById('loading'),
    fps: document.getElementById('fps'),
    latency: document.getElementById('latency'),
    detectionCount: document.getElementById('detection-count'),
    detectionsList: document.getElementById('detections-list'),
    jobsList: document.getElementById('jobs-list'),
    videoStatus: document.getElementById('video-status'),
    imageMetrics: document.getElementById('image-metrics')
};

// Colors for different classes
const COLORS = {
    'person': '#FF6B6B',
    'car': '#4ECDC4',
    'truck': '#45B7D1',
    'bus': '#96CEB4',
    'motorcycle': '#FFEAA7',
    'bicycle': '#DDA0DD',
    'default': '#FFD93D'
};

// Initialize
async function init() {
    await checkAPIHealth();
    setupEventListeners();
    setupFullscreen();  
    startHealthPolling();
    refreshJobsList();

}


// Check API health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            elements.apiStatus.textContent = '● API Healthy';
            elements.apiStatus.className = 'status-badge healthy';
        } else {
            throw new Error('API unhealthy');
        }
    } catch (error) {
        elements.apiStatus.textContent = '● API Unreachable';
        elements.apiStatus.className = 'status-badge unhealthy';
    }
}

// Health polling every 30 seconds
function startHealthPolling() {
    setInterval(checkAPIHealth, 30000);
}

// Event listeners
function setupEventListeners() {
    elements.imageSubmit.addEventListener('click', handleImageUpload);
    elements.videoSubmit.addEventListener('click', handleVideoUpload);
    document.getElementById('watch-video-btn').addEventListener('click', handleWatchVideo);
    document.getElementById('download-video-btn').addEventListener('click', handleDownloadVideo);
}

function setupFullscreen() {
    const btn = document.getElementById('fullscreen-btn');
    const container = document.getElementById('canvas-container');

    btn.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            container.requestFullscreen().catch(err => {
                console.error('Fullscreen error:', err);
            });
        } else {
            document.exitFullscreen();
        }
    });

    document.addEventListener('fullscreenchange', () => {
        btn.textContent = document.fullscreenElement ? '✕' : '⛶';
    });
}

// Handle image upload
async function handleImageUpload(e) {
    
    const file = elements.imageInput.files[0];
    if (!file) return;

    // Stop any running detection FPS counter from a previous video session
    if (detectionFpsInterval) {
        clearInterval(detectionFpsInterval);
        detectionFpsInterval = null;
    }

    // Reset video player from any previous session
    const videoEl = document.getElementById('video-player');
    videoEl.pause();
    videoEl.removeAttribute('src');
    videoEl.load();

    // Show loading
    elements.loading.classList.add('active');
    elements.imageSubmit.disabled = true;
    elements.imageMetrics.innerHTML = '';

    // Clear previous canvas drawings
    const ctx = elements.overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const startTime = performance.now();
        const response = await fetch(`${API_URL}/v1/detect/image`, {
            method: 'POST',
            body: formData
        });
        const endTime = performance.now();

        // Always parse JSON — API returns structured errors on failure too
        const result = await response.json();

        if (!response.ok) {
            // Extract structured error from our API's detail field
            const detail = result.detail;
            let errorMessage;

            if (typeof detail === 'object' && detail.message) {
                errorMessage = detail.message;
                if (detail.supported_formats) {
                    errorMessage += ` Supported formats: ${detail.supported_formats.join(', ')}`;
                }
            } else {
                errorMessage = typeof detail === 'string' ? detail : `HTTP ${response.status}`;
            }

            const label = response.status === 413 ? 'File too large' : 'Invalid file';
            elements.imageMetrics.innerHTML = 
                `<span style="color: #e53e3e">❌ ${label}: ${errorMessage}</span>`;
            return; // finally block still runs
        }

        // Display original image with detections drawn on canvas
        const imageUrl = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
            drawImageWithDetections(img, result.detections);
            URL.revokeObjectURL(imageUrl);

            if (result.traffic) {
                updateTrafficDisplay(result.traffic);
                updateVehicleCounts(result.traffic.counts);
            }
        };
        img.src = imageUrl;

        // Update detection count — vehicles only, persons excluded from traffic density
        elements.detectionCount.textContent = result.detections.length;
        const detectionUnit = document.getElementById('detection-unit');
        if (detectionUnit) detectionUnit.textContent = '';

        // Update latency
        const latency = endTime - startTime;
        elements.latency.textContent = `${Math.round(latency)}ms`;

        // Update FPS display and unit label
        if (result.timing_ms?.total_ms) {
            const fps = Math.round(1000 / result.timing_ms.total_ms);
            elements.fps.textContent = fps;
            const fpsUnit = document.getElementById('fps-unit');
            if (fpsUnit) fpsUnit.textContent = 'img/s (model)';
        }

        // Show detailed pipeline breakdown
        if (result.timing_ms) {
            const serverTotalMs = result.timing_ms.total_ms;
            const transportMs = Math.max(0, Math.round(latency - serverTotalMs));

            elements.imageMetrics.innerHTML = `
                <strong>Pipeline Breakdown:</strong><br>
                Decode: ${result.timing_ms.decode_ms?.toFixed(1) ?? '0'}ms<br>
                Inference: ${result.timing_ms.inference_ms?.toFixed(1) ?? '0'}ms<br>
                Postprocess: ${result.timing_ms.postprocess_ms?.toFixed(1) ?? '0'}ms<br>
                Serialization: ${result.timing_ms.serialize_ms?.toFixed(1) ?? '0'}ms<br>
                Transport: ~${transportMs}ms<br>
                <strong>Total round-trip: ${Math.round(latency)}ms</strong>
            `;
        } else {
            elements.imageMetrics.innerHTML = `
                <strong>Processing complete</strong><br>
                Total time: ${Math.round(latency)}ms
            `;
        }

        // Show detections list in sidebar
        displayDetectionsList(result.detections);

    } catch (error) {
        // Network-level failure — response never arrived
        console.error('Image upload error:', error);
        elements.imageMetrics.innerHTML = 
            `<span style="color: #e53e3e">❌ Request failed: ${error.message}</span>`;
    } finally {
        elements.loading.classList.remove('active');
        elements.imageSubmit.disabled = false;
    }
}

// Draw image with detection overlays
function drawImageWithDetections(img, detections) {
    // Hide video element so it doesn't bleed through
    const videoEl = document.getElementById('video-player');
    videoEl.style.visibility = 'hidden';

    // Use only overlay canvas
    const canvas = elements.overlayCanvas;
    const ctx = canvas.getContext('2d');
    
    // Set canvas dimensions to match image
    canvas.width = img.width;
    canvas.height = img.height;
    
    // Draw the image
    ctx.drawImage(img, 0, 0, img.width, img.height);
    
    // Draw bounding boxes on top
    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        
        const color = COLORS[det.class_name] || COLORS.default;
        
        // Draw rectangle
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);
        
        // Draw label background
        ctx.fillStyle = color;
        const label = `${det.class_name} ${(det.score * 100).toFixed(1)}%`;
        const metrics = ctx.measureText(label);
        ctx.fillRect(x1, y1 - 25, metrics.width + 10, 20);
        
        // Draw label text
        ctx.fillStyle = '#000000';
        ctx.font = '14px Arial';
        ctx.fillText(label, x1 + 5, y1 - 10);
    });

    // Update stats
}

// Handle video upload
// ui/script.js - Update handleVideoUpload function

async function handleVideoUpload(e) {

    const file = elements.videoInput.files[0];
    if (!file) return;

    if (detectionFpsInterval) {
        clearInterval(detectionFpsInterval);
        detectionFpsInterval = null;
    }

    elements.videoSubmit.disabled = true;
    elements.videoStatus.innerHTML = 'Uploading…';

    activeJobId = null;
    detectionTimeline = [];
    lastDrawnIndex = -1;
    isPlaybackReady = false;

    if (window.detectionWs) { window.detectionWs.close(); window.detectionWs = null; }
    if (animationFrame) { cancelAnimationFrame(animationFrame); animationFrame = null; }
    if (videoElement) { videoElement.pause(); videoElement.src = ''; }

    document.getElementById('video-actions').style.display = 'none';
    document.getElementById('download-progress-section').style.display = 'none';
    stopJobPolling();

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/v1/detect/video`, {
            method: 'POST',
            body: formData
        });

        // Always parse JSON — API returns structured errors on failure too
        const result = await response.json();

        if (!response.ok) {
            const detail = result.detail;
            let errorMessage;

            if (typeof detail === 'object' && detail.message) {
                errorMessage = detail.message;
                if (detail.supported_formats) {
                    errorMessage += ` Supported formats: ${detail.supported_formats.join(', ')}.`;
                }
            } else {
                errorMessage = typeof detail === 'string' ? detail : `HTTP ${response.status}`;
            }

            elements.videoStatus.innerHTML =
                `<span style="color: #e53e3e">❌ Upload failed: ${errorMessage}</span>`;
            return;
        }

        activeJobId = result.job_id;

        elements.videoStatus.innerHTML = `
            <strong>✅ Video uploaded</strong><br>
            Choose an option below to continue.
        `;

        document.getElementById('video-actions').style.display = 'block';
        document.getElementById('watch-video-btn').disabled = false;
        document.getElementById('download-video-btn').disabled = false;

        refreshJobsList();

    } catch (error) {
        console.error('Upload error:', error);
        elements.videoStatus.innerHTML =
            `<span style="color: #e53e3e">❌ Upload failed: ${error.message}</span>`;
    } finally {
        elements.videoSubmit.disabled = false;
    }
}

async function handleWatchVideo() {
    if (!activeJobId) return;

    const btn = document.getElementById('watch-video-btn');
    btn.disabled = true;

    try {
        // Tell the backend to start processing with sample_rate=3
        const resp = await fetch(`${API_URL}/v1/jobs/${activeJobId}/start_watch`, {
            method: 'POST'
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        // Disable download button — the two modes are mutually exclusive
        // once processing starts (different sample rates)
        document.getElementById('download-video-btn').disabled = true;

        // Reset timeline for fresh session
        detectionTimeline = [];
        lastDrawnIndex = -1;
        isPlaybackReady = false;

        // Initialize player (doesn't autoplay — buffer logic controls that)
        initVideoPlayer(activeJobId);

        // Open WebSocket detection stream
        connectDetectionStream(activeJobId);

        // Clear any previous interval
        if (detectionFpsInterval) {
            clearInterval(detectionFpsInterval);
        }

        // Calculate detection FPS once per second
        detectionFpsInterval = setInterval(() => {
            if (detectionTimeline.length >= 2) {
                const newest = detectionTimeline[detectionTimeline.length - 1].timestamp_ms;
                const oldest = detectionTimeline[0].timestamp_ms;
                const spanSeconds = (newest - oldest) / 1000;
                if (spanSeconds > 0) {
                    const fps = (detectionTimeline.length / spanSeconds).toFixed(1);
                    elements.fps.textContent = fps;
                    const fpsUnit = document.getElementById('fps-unit');
                    if (fpsUnit) fpsUnit.textContent = 'det. frames/s';  
                }
            } else {
                elements.fps.textContent = '-';
            }
        }, 1000);

        // Start polling for job status updates in the sidebar
        startJobPolling(activeJobId);

        document.getElementById('loading').innerHTML = '⏳ Buffering detections…';
        document.getElementById('loading').classList.add('active');

        document.querySelector('.visualization-panel').scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error('Watch error:', error);
        elements.videoStatus.innerHTML += `<br><span style="color:red">Error: ${error.message}</span>`;
        btn.disabled = false;
    }
}

async function handleDownloadVideo() {
    if (!activeJobId) return;

    const btn = document.getElementById('download-video-btn');
    const progressSection = document.getElementById('download-progress-section');
    const progressBar = document.getElementById('download-progress-bar');
    const progressPct = document.getElementById('download-progress-pct');

    btn.disabled = true;
    btn.textContent = '⏳ Processing…';

    // Disable watch button — modes are mutually exclusive
    document.getElementById('watch-video-btn').disabled = true;

    progressSection.style.display = 'block';
    progressBar.style.width = '0%';
    progressPct.textContent = '0%';

    elements.videoStatus.innerHTML = `
        <strong>⏳ Processing all frames at full quality…</strong><br>
        This takes longer than watch mode. The download will start automatically when ready.
    `;

    try {
        // Tell backend to start inference + render at sample_rate=1
        const resp = await fetch(`${API_URL}/v1/jobs/${activeJobId}/start_download`, {
            method: 'POST'
        });

        if (!resp.ok) {
            const err = await resp.json();
            const detail = err.detail;
            let errorMessage;

            if (typeof detail === 'object' && detail.message) {
                errorMessage = detail.message;
                if (detail.hint) errorMessage += ` ${detail.hint}`;
            } else {
                errorMessage = typeof detail === 'string' ? detail : `HTTP ${resp.status}`;
            }

            throw new Error(errorMessage);
        }
        const data = await resp.json();

        // If somehow already ready (e.g. watch was done before)
        if (data.status === 'ready') {
            triggerBrowserDownload(activeJobId);
            progressSection.style.display = 'none';
            btn.disabled = false;
            btn.textContent = '⬇ Download Annotated Video';
            return;
        }

        // Poll until the annotated video is rendered
        await pollDownloadReady(activeJobId, progressBar, progressPct);

        elements.videoStatus.innerHTML = `<strong>✅ Ready! Downloading…</strong>`;
        triggerBrowserDownload(activeJobId);

    } catch (error) {
        console.error('Download error:', error);
        elements.videoStatus.innerHTML = `<span style="color:red">Download failed: ${error.message}</span>`;
        // Re-enable so user can retry
        btn.disabled = false;
        document.getElementById('watch-video-btn').disabled = false;
    } finally {
        btn.textContent = '⬇ Download Annotated Video';
        progressSection.style.display = 'none';
    }
}


function triggerBrowserDownload(jobId) {
    // Creates a hidden <a> tag and clicks it — browser shows native Save As dialog
    const a = document.createElement('a');
    a.href = `${API_URL}/v1/jobs/${jobId}/download`;
    a.download = `traffic_analysis_${jobId.slice(0, 8)}.mp4`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    console.log('Browser download triggered for job:', jobId);
}

function pollDownloadReady(jobId, progressBar, progressPct) {
    return new Promise((resolve, reject) => {
        const interval = setInterval(async () => {
            try {
                const resp = await fetch(`${API_URL}/v1/jobs/${jobId}/download_status`);
                if (!resp.ok) return;

                const data = await resp.json();
                const pct = data.progress ?? 0;
                const phase = data.download_status ?? 'processing';

                // Show phase-aware label
                const PHASE_LABELS = {
                    'inference': '🔍 Running detection…',
                    'rendering': '🎬 Rendering video…',
                    'queued':    '⏳ Queued…',
                };
                const phaseLabel = PHASE_LABELS[phase] ?? '⏳ Processing…';

                progressBar.style.width = `${pct}%`;
                progressPct.textContent = `${pct.toFixed(1)}% — ${phaseLabel}`;

                if (data.download_status === 'ready') {
                    clearInterval(interval);
                    progressBar.style.width = '100%';
                    progressPct.textContent = '100% — Done!';
                    resolve();
                } else if (data.download_status === 'failed') {
                    clearInterval(interval);
                    reject(new Error(data.error || 'Generation failed'));
                }
            } catch (err) {
                console.error('Poll error:', err);
                // Don't reject — keep polling through transient network issues
            }
        }, 1500);

        setTimeout(() => {
            clearInterval(interval);
            reject(new Error('Download generation timed out'));
        }, 600_000);
    });
}


// Check job status
async function checkJobStatus(jobId) {
    try {
        const response = await fetch(`${API_URL}/v1/jobs/${jobId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        displayJobStatus(result);
        
        // If completed, stop polling
        if (result.status === 'completed' || result.status === 'failed' || result.status === 'cancelled') {
            stopJobPolling();
        }
        
        return result;
    } catch (error) {
        console.error('Error checking job status:', error);
    }
}

// Start polling for job status
function startJobPolling(jobId) {
    if (statusPolling) {
        clearInterval(statusPolling);
    }
    
    statusPolling = setInterval(async () => {
        const result = await checkJobStatus(jobId);
        if (result && (result.status === 'completed' || result.status === 'failed' || result.status === 'cancelled')) {
            clearInterval(statusPolling);
            statusPolling = null;
        }
    }, 2000);
}

// Stop job polling
function stopJobPolling() {
    if (statusPolling) {
        clearInterval(statusPolling);
        statusPolling = null;
    }
}

// Display job status
function displayJobStatus(job) {
    let statusHtml = '';

    if (job.status === 'completed') {
        statusHtml = `
            <strong>✅ Detection complete</strong><br>
            Frames processed: ${job.frames_processed}<br>
            Avg frame time: ${job.timing_ms?.avg_frame_ms?.toFixed(1) ?? '—'}ms
        `;
        // Enable both action buttons now that the job is done
        document.getElementById('watch-video-btn').disabled = false;
        document.getElementById('download-video-btn').disabled = false;

    } else if (job.status === 'failed') {
        statusHtml = `<strong>❌ Failed:</strong> ${job.error || 'Unknown error'}`;

    } else if (job.status === 'cancelled') {
        statusHtml = `<strong>⏹️ Cancelled</strong>`;

    } else {
        // Still processing
        const progress = job.progress?.toFixed(1) ?? '0.0';
        statusHtml = `
            <strong>⏳ Processing…</strong><br>
            Progress: ${progress}%<br>
            Frames: ${job.frames_processed ?? 0} / ${job.total_frames ?? '?'}
        `;
    }

    elements.videoStatus.innerHTML = statusHtml;
}


// Refresh jobs list
async function refreshJobsList() {
    try {
        const response = await fetch(`${API_URL}/v1/jobs?limit=10`);
        if (!response.ok) return;
        
        const data = await response.json();
        
        if (data.jobs && data.jobs.length > 0) {
            elements.jobsList.innerHTML = data.jobs.map(job => `
                <div class="job-item" onclick="checkJobStatus('${job.job_id}')">
                    <span>${job.filename || 'Unknown'}</span>
                    <span class="job-status status-${job.status}">${job.status}</span>
                </div>
            `).join('');
        } else {
            elements.jobsList.innerHTML = 'No recent jobs';
        }
    } catch (error) {
        console.error('Error refreshing jobs:', error);
        elements.jobsList.innerHTML = 'Error loading jobs';
    }
}

// Display detections list
function displayDetectionsList(detections) {
    if (detections.length === 0) {
        elements.detectionsList.innerHTML = '<i>No detections found</i>';
        return;
    }
    
    elements.detectionsList.innerHTML = detections.map(det => `
        <div class="detection-item">
            <span class="detection-class">${det.class_name}</span>
            <span class="detection-score">${(det.score * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}



function drawDetections(detections) {
    const canvas = document.getElementById('overlay-canvas');
    const ctx = canvas.getContext('2d');
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!detections || detections.length === 0 || !videoElement) {
        return;
    }
    
    // Calculate scale factors once
    const videoWidth = videoElement.videoWidth;
    const videoHeight = videoElement.videoHeight;
    
    if (videoWidth === 0 || videoHeight === 0) {
        return;
    }
    
    const scaleX = canvas.width / videoWidth;
    const scaleY = canvas.height / videoHeight;
    const scale = Math.min(scaleX, scaleY);
    const lineWidth = Math.max(2, 3 * scale);
    
    // Batch drawing operations
    ctx.lineWidth = lineWidth;
    ctx.font = `${Math.max(12, 14 * scale)}px Arial`;
    
    // Draw all boxes first (faster than drawing one by one with labels)
    ctx.beginPath();
    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const scaledX1 = x1 * scaleX;
        const scaledY1 = y1 * scaleY;
        const scaledWidth = (x2 - x1) * scaleX;
        const scaledHeight = (y2 - y1) * scaleY;
        
        const color = COLORS[det.class_name] || COLORS.default;
        ctx.strokeStyle = color;
        ctx.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);
    });
    
    // Then draw labels (smaller batch)
    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;
        const scaledX1 = x1 * scaleX;
        const scaledY1 = y1 * scaleY;
        
        const color = COLORS[det.class_name] || COLORS.default;
        const label = `${det.class_name} ${(det.score * 100).toFixed(1)}%`;
        
        // Draw label background
        ctx.fillStyle = color;
        const metrics = ctx.measureText(label);
        const labelHeight = 20 * scale;
        const labelWidth = metrics.width + 10;
        
        ctx.fillRect(scaledX1, scaledY1 - labelHeight, labelWidth, labelHeight);
        
        // Draw label text
        ctx.fillStyle = '#000000';
        ctx.fillText(label, scaledX1 + 5, scaledY1 - 5);
    });
}

function connectDetectionStream(jobId) {
    // Close existing connection if any
    if (window.detectionWs) {
        window.detectionWs.close();
    }
    
    // Reset timeline
    detectionTimeline = [];
    lastDrawnIndex = -1;
    
    // Create WebSocket connection
    const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;
    const ws = new WebSocket(`${WS_URL}/ws/video/${jobId}`);
    window.detectionWs = ws;

    ws.onopen = () => {
        console.log('Detection stream connected for job', jobId);
        document.getElementById('loading').innerHTML = 'Connecting to detection stream...';
        document.getElementById('loading').classList.add('active');
    };

    ws.onmessage = (event) => {
        const now = performance.now();
        debugStats.lastFrameTime = now;
        
        const data = JSON.parse(event.data);
        
        // Handle control messages
        if (data.status === 'complete') {
            console.log('Job complete, closing stream');
            ws.close();
            return;
        }
        
        if (data.error) {
            console.error('Stream error:', data.error, data.message);
            document.getElementById('loading').innerHTML =
                `<span style="color: #e53e3e">❌ ${data.message || data.error}</span>`;
            document.getElementById('loading').classList.add('active');
            return;
        }
        
        // Regular frame data
        if (data.timestamp_ms !== undefined) {
            // Add to timeline
            detectionTimeline.push({
                timestamp_ms: data.timestamp_ms,
                frame_number: data.frame_number,
                detections: data.detections || [],
                traffic: data.traffic,
                received_at: now
            }); 

            if (detectionTimeline.length > MAX_TIMELINE_SIZE) {
                detectionTimeline = detectionTimeline.slice(-MAX_TIMELINE_SIZE);
            }
            
            // Update progress if available
            if (data.progress) {
                const progress = data.progress.percent || 0;
                document.getElementById('loading').innerHTML = 
                    `Loading detections: ${progress.toFixed(1)}% (${data.progress.processed}/${data.progress.total} frames)`;
                
                if (progress >= 100) {
                    document.getElementById('loading').classList.remove('active');
                }
            }
            
            // 🚨 PROFESSIONAL BUFFER LOGIC 🚨
            // Check if we have enough buffer to start playback
            if (!isPlaybackReady && videoElement) {
                const newestTimestamp = detectionTimeline[detectionTimeline.length - 1].timestamp_ms;
                const oldestTimestamp = detectionTimeline[0].timestamp_ms;
                const bufferedMs = newestTimestamp - oldestTimestamp;

                if (bufferedMs >= BUFFER_TARGET_MS) {
                    isPlaybackReady = true;
                    setTimeout(() => {
                        if (videoElement && videoElement.paused) {
                            videoElement.play()
                                .then(() => {
                                    console.log('Playback started with buffer');
                                    document.getElementById('loading').classList.remove('active');
                                })
                                .catch(e => {
                                    console.log('Autoplay prevented:', e);
                                    // Show a manual play hint if autoplay was blocked
                                    document.getElementById('loading').innerHTML = '▶ Click video to play';
                                    document.getElementById('loading').classList.add('active');
                                });
                        }
                    }, 100);
                } else {
                    const remaining = ((BUFFER_TARGET_MS - bufferedMs) / 1000).toFixed(1);
                    document.getElementById('loading').innerHTML =
                        `⏳ Buffering: ${(bufferedMs / 1000).toFixed(1)}s / ${BUFFER_TARGET_MS / 1000}s (${remaining}s remaining)`;
                    document.getElementById('loading').classList.add('active');
                }
            }
            
            // If video is playing and we just added a frame near current time, trigger redraw
            if (videoElement && !videoElement.paused) {
                const currentTimeMs = videoElement.currentTime * 1000;
                const timeDiff = Math.abs(data.timestamp_ms - currentTimeMs);
                if (timeDiff < 100) {
                    lastDrawnIndex = -1;
                    if (data.traffic) {
                        updateTrafficDisplay(data.traffic);
                        updateVehicleCounts(data.traffic.counts);
                    }
                }
            }
        }
    };



    ws.onerror = (err) => {
        console.error('Detection stream error:', err);
        document.getElementById('loading').innerHTML = 'WebSocket error';
    };

    ws.onclose = () => {
        console.log('Detection stream closed');
        window.detectionWs = null;
        
        // Stop FPS counter
        if (detectionFpsInterval) {
            clearInterval(detectionFpsInterval);
            detectionFpsInterval = null;
        }

        // Only show "stream closed" if video is also done
        // If video is still playing, the canvas animation loop keeps running fine
        if (videoElement && !videoElement.ended && !videoElement.paused) {
            // Video still playing — don't interrupt, just log
            console.log('Stream closed but video still playing — canvas loop continues');
            document.getElementById('loading').classList.remove('active');
        } else {
            document.getElementById('loading').innerHTML = 'Stream closed';
            setTimeout(() => {
                document.getElementById('loading').classList.remove('active');
            }, 2000);
        }
    };
    
    return ws;
}

function initVideoPlayer(jobId) {
    console.log('Initializing video player for job:', jobId);
    videoElement = document.getElementById('video-player');

    // Restore video visibility
    videoElement.style.visibility = 'visible';

    videoElement.controlsList = 'nofullscreen';

    // Construct URL properly
    const videoUrl = `${API_URL}/v1/video/${jobId}`;
    console.log('Video URL:', videoUrl);
    
    videoElement.src = videoUrl;
    videoElement.muted = true; 
    videoElement.load();
    
    // 🚨 IMPORTANT: Don't autoplay immediately - wait for buffer
    // videoElement.play() is removed from here - now controlled by buffer logic
    
    videoElement.onerror = (e) => {
        console.error('Video loading error:', e);
        console.error('Video error code:', videoElement.error ? videoElement.error.code : 'unknown');
    };
    
    // Set up event listeners
    videoElement.addEventListener('loadedmetadata', () => {
        const canvas = document.getElementById('overlay-canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        console.log('Video loaded:', videoElement.videoWidth, 'x', videoElement.videoHeight);
        
        document.getElementById('loading').innerHTML = 'Buffering detections...';
    });
    
    // Reset buffer state for new video
    isPlaybackReady = false;
    
    // Use requestAnimationFrame for smooth updates
    function updateCanvas() {
        if (!videoElement || videoElement.paused || videoElement.ended) {
            // Don't reschedule — play/pause event listeners restart the loop
            animationFrame = null;
            return;
        }
        
        const currentTimeMs = videoElement.currentTime * 1000;
        
        // SUPER OPTIMIZATION: Only update every 3 frames (20fps is still smooth)
        window.frameCounter = (window.frameCounter || 0) + 1;
        if (window.frameCounter % 3 !== 0) {
            animationFrame = requestAnimationFrame(updateCanvas);
            return;
        }
        
        // Find closest detection in timeline (binary search)
        if (detectionTimeline.length > 0) {
            let left = 0;
            let right = detectionTimeline.length - 1;
            let bestIndex = -1;
            
            while (left <= right) {
                const mid = Math.floor((left + right) / 2);
                if (detectionTimeline[mid].timestamp_ms <= currentTimeMs) {
                    bestIndex = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            
            // Only redraw if frame index changed
            if (bestIndex >= 0 && bestIndex !== lastDrawnIndex) {
                const frameData = detectionTimeline[bestIndex];
                
                // 🚨 REMOVED: JSON.stringify comparison (too slow!)
                
                // Draw directly
                drawDetections(frameData.detections);
                
                // Update stats (but not every frame)
                if (window.frameCounter % 15 === 0) {
                    elements.detectionCount.textContent = frameData.detections.length;
                    const detectionUnit = document.getElementById('detection-unit');
                    if (detectionUnit) detectionUnit.textContent = '';


                    if (frameData.traffic) {
                        updateTrafficDisplay(frameData.traffic);
                        updateVehicleCounts(frameData.traffic.counts);
                    }
                    
                    const latency = currentTimeMs - frameData.timestamp_ms;
                    if (latency >= 0 && latency < 10000) {
                        elements.latency.textContent = `${Math.round(latency)}ms`;
                    }
                }
                
                lastDrawnIndex = bestIndex;
            }
        }
        
        animationFrame = requestAnimationFrame(updateCanvas);
    }
    // Start the animation loop
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
    }
    animationFrame = requestAnimationFrame(updateCanvas);
    
    // Add play/pause controls
    videoElement.addEventListener('play', () => {
        animationFrame = requestAnimationFrame(updateCanvas);
    });
    
    videoElement.addEventListener('pause', () => {
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
            animationFrame = null;
        }
    });
    
}

// ===== 🔴 END WEBSOCKET CODE =====

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);


// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    console.log('Page unloading, cleaning up resources...');
    
    // Close WebSocket connection
    if (window.detectionWs) {
        window.detectionWs.close();
        window.detectionWs = null;
    }
    
    // Cancel animation frame
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }
    
    // Stop video playback
    if (videoElement) {
        videoElement.pause();
        videoElement.src = '';
        videoElement.load();
    }
    
    // Clear timeline to free memory
    detectionTimeline = [];
    lastDrawnIndex = -1;
});

// Also add cleanup when navigating away via SPA-style navigation (if using any)
// This is optional but good practice
window.addEventListener('pagehide', () => {
    if (window.detectionWs) {
        window.detectionWs.close();
    }
});


// Add throttling to prevent too many DOM updates
let lastTrafficUpdate = 0;
function updateTrafficDisplay(trafficData) {
    if (!trafficData) return;
    
    const now = Date.now();
    if (now - lastTrafficUpdate < 100) return; // Update max 10x per second
    lastTrafficUpdate = now;
    
    document.getElementById('traffic-density').textContent = `${trafficData.density}%`;
    document.getElementById('traffic-bar').style.width = `${trafficData.density}%`;
    
    const statusEl = document.getElementById('traffic-status');
    statusEl.textContent = `${trafficData.status_icon} ${trafficData.status}`;
    statusEl.className = `traffic-status status-${trafficData.status}`;
}

let lastCountsUpdate = 0;
function updateVehicleCounts(counts) {
    if (!counts) return;
    
    const now = Date.now();
    if (now - lastCountsUpdate < 100) return; // Update max 10x per second
    lastCountsUpdate = now;
    
    document.getElementById('count-cars').textContent = counts.cars || 0;
    document.getElementById('count-motorcycles').textContent = counts.motorcycles || 0;
    document.getElementById('count-trucks').textContent = counts.trucks || 0;
    document.getElementById('count-buses').textContent = counts.buses || 0;
    document.getElementById('count-persons').textContent = counts.persons || 0;
    document.getElementById('count-bicycles').textContent = counts.bicycles || 0;
}
