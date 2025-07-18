document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const startScanBtn = document.getElementById('start-scan-btn');
    const feedbackMessage = document.getElementById('feedback-message');
    const scanLine = document.getElementById('scan-line');
    const eventId = window.PICME_EVENT_ID || 'default_event';

    // Update feedback message
    function updateFeedback(message, type = 'info') {
        const colors = {
            info: 'text-accent',
            warning: 'text-warning',
            success: 'text-success',
            error: 'text-error'
        };
        
        feedbackMessage.innerHTML = `
            <div class="inline-flex items-center space-x-2 px-4 py-2 bg-${type}/10 rounded-full">
                <div class="w-2 h-2 bg-${type} rounded-full animate-pulse"></div>
                <span class="${colors[type]} font-medium">${message}</span>
            </div>
        `;
    }

    // Start camera when scan button is clicked
    startScanBtn.addEventListener('click', async function() {
        try {
            updateFeedback("Initializing camera...", "info");
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'user',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false 
            });
            
            video.srcObject = stream;
            
            // Update UI for scanning state
            startScanBtn.disabled = true;
            startScanBtn.innerHTML = '<div class="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div><span>Scanning...</span>';
            
            updateFeedback("Position your face in the circle", "warning");
            
            // Show scanning animation
            scanLine.classList.remove('opacity-0');
            scanLine.style.top = '0';
            scanLine.style.animation = 'scan-line 2s ease-in-out infinite';
            
            // Wait 3 seconds to capture
            setTimeout(() => {
                captureFace();
                stream.getTracks().forEach(track => track.stop());
            }, 3000);
            
        } catch (err) {
            console.error("Camera access error:", err);
            updateFeedback("Could not access camera. Please ensure permissions are granted.", "error");
            startScanBtn.disabled = false;
            startScanBtn.innerHTML = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg><span>Start Face Scan</span>';
        }
    });

    function captureFace() {
        try {
            updateFeedback("Analyzing facial features...", "info");
            
            // Draw video frame to canvas
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Get image data for API call
            const imageData = canvas.toDataURL('image/jpeg');
            
            // Send to backend API
            recognizeFace(imageData);
        } catch (error) {
            console.error("Capture error:", error);
            updateFeedback("Error capturing image. Please try again.", "error");
            resetScanner();
        }
    }

    async function recognizeFace(imageData) {
        try {
            updateFeedback("Verifying identity...", "info");
            
            const response = await fetch('http://localhost:5000/recognize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    image: imageData.split(',')[1],
                    event_id: eventId
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                updateFeedback("Match found! Redirecting...", "success");
                
                // Store person ID in session storage
                sessionStorage.setItem('picme_person_id', data.person_id);
                sessionStorage.setItem('picme_event_id', data.event_id);
                
                // Redirect to gallery after short delay
                setTimeout(() => {
                    window.location.href = 'personal_photo_gallery.html';
                }, 1500);
            } else {
                updateFeedback(data.error || "No matching photos found", "error");
                resetScanner();
            }
        } catch (error) {
            console.error("Recognition error:", error);
            updateFeedback("Recognition service unavailable. Please try again later.", "error");
            resetScanner();
        }
    }

    function resetScanner() {
        startScanBtn.disabled = false;
        startScanBtn.innerHTML = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg><span>Start Face Scan</span>';
        scanLine.classList.add('opacity-0');
    }
});