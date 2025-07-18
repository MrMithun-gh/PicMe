document.addEventListener('DOMContentLoaded', function() {
    const personId = sessionStorage.getItem('picme_person_id');
    const eventId = sessionStorage.getItem('picme_event_id');
    
    if (!personId || !eventId) {
        window.location.href = 'biometric_authentication_portal.html';
        return;
    }
    
    // Load gallery data
    loadGallery(eventId, personId);
    
    // Set up download buttons
    document.querySelectorAll('.download-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const photoUrl = this.dataset.photoUrl;
            downloadPhoto(eventId, personId, photoUrl);
        });
    });
});

async function loadGallery(eventId, personId) {
    try {
        // Show loading state
        document.getElementById('individual-photos').innerHTML = '<div class="col-span-3 flex justify-center py-12"><div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-accent"></div></div>';
        document.getElementById('group-photos').innerHTML = '<div class="col-span-3 flex justify-center py-12"><div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-accent"></div></div>';
        
        // In a real app, you would fetch from your backend API
        // For demo, we'll simulate the response
        const response = await fetch(`http://localhost:5000/recognize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                event_id: eventId,
                person_id: personId
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            renderPhotos('individual-photos', data.individual_photos, eventId, personId, 'individual');
            renderPhotos('group-photos', data.group_photos, eventId, personId, 'group');
        } else {
            throw new Error(data.error || "Failed to load photos");
        }
    } catch (error) {
        console.error("Gallery load error:", error);
        document.getElementById('individual-photos').innerHTML = `<div class="col-span-3 text-center py-12 text-text-secondary">${error.message}</div>`;
        document.getElementById('group-photos').innerHTML = `<div class="col-span-3 text-center py-12 text-text-secondary">${error.message}</div>`;
    }
}

function renderPhotos(containerId, photos, eventId, personId, photoType) {
    const container = document.getElementById(containerId);
    
    if (!photos || photos.length === 0) {
        container.innerHTML = `<div class="col-span-3 text-center py-12 text-text-secondary">No ${photoType} photos found</div>`;
        return;
    }
    
    let html = '';
    photos.forEach(photo => {
        const isWatermarked = photo.includes('watermarked_');
        const originalPhoto = isWatermarked ? photo.replace('watermarked_', '') : photo;
        
        html += `
            <div class="relative group overflow-hidden rounded-lg shadow-md hover:shadow-lg transition-shadow">
                <img src="http://localhost:5000/photos/${eventId}/${personId}/${photoType}/${photo}" 
                     alt="Event photo" 
                     class="w-full h-64 object-cover transition-transform duration-300 group-hover:scale-105">
                
                <div class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center p-4">
                    <button class="download-btn btn-primary" 
                            data-photo-url="${originalPhoto}"
                            data-photo-type="${photoType}">
                        <svg class="w-5 h-5 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
                        </svg>
                        Download Original
                    </button>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
    
    // Set up download buttons
    document.querySelectorAll('.download-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const photoUrl = this.dataset.photoUrl;
            const photoType = this.dataset.photoType;
            downloadPhoto(eventId, personId, photoType, photoUrl);
        });
    });
}

function downloadPhoto(eventId, personId, photoType, photoUrl) {
    window.open(`http://localhost:5000/download/${eventId}/${personId}/${photoType}/${photoUrl}`, '_blank');
}