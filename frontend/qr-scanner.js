// Using html5-qrcode library (https://github.com/mebjas/html5-qrcode)
document.addEventListener('DOMContentLoaded', function() {
    const qrScanBtn = document.getElementById('qr-scan-btn');
    if (!qrScanBtn) return;
    
    qrScanBtn.addEventListener('click', initQRScanner);
});

function initQRScanner() {
    // Create scanner container if it doesn't exist
    if (!document.getElementById('qr-scanner-container')) {
        const scannerDiv = document.createElement('div');
        scannerDiv.id = 'qr-scanner-container';
        scannerDiv.className = 'fixed inset-0 z-50 bg-black/80 flex items-center justify-center';
        document.body.appendChild(scannerDiv);
    }
    
    const html5QrCode = new Html5Qrcode("qr-scanner-container");
    
    const qrCodeSuccessCallback = (decodedText, decodedResult) => {
        // Stop scanner
        html5QrCode.stop().then(() => {
            // Remove scanner container
            document.getElementById('qr-scanner-container').remove();
            
            // Extract event ID from URL
            const eventId = decodedText.split('/').pop();
            
            // Redirect to biometric portal with event ID
            window.location.href = `biometric_authentication_portal.html?event_id=${eventId}`;
        });
    };
    
    const config = { 
        fps: 10,
        qrbox: { width: 250, height: 250 }
    };
    
    // Start scanner
    html5QrCode.start(
        { facingMode: "environment" },
        config,
        qrCodeSuccessCallback
    ).catch(err => {
        console.error("QR Scanner error:", err);
        alert("Could not start QR scanner. Please ensure camera permissions are granted.");
        document.getElementById('qr-scanner-container').remove();
    });
    
    // Add close button
    const closeBtn = document.createElement('button');
    closeBtn.className = 'absolute top-4 right-4 text-white bg-red-500 hover:bg-red-600 rounded-full p-2';
    closeBtn.innerHTML = '<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>';
    closeBtn.addEventListener('click', () => {
        html5QrCode.stop().then(() => {
            document.getElementById('qr-scanner-container').remove();
        });
    });
    
    document.getElementById('qr-scanner-container').appendChild(closeBtn);
}