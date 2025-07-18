from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import numpy as np
import cv2
import face_recognition
from datetime import datetime, timedelta
import shutil
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import threading
import time

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
WATERMARK_TEXT = "PicMe Preview"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# In-memory storage for face encodings (in production, use a database)
known_encodings = []
known_ids = []

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_watermark(image_path, output_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            draw = ImageDraw.Draw(img)
            
            # Use a default font (you might need to adjust this for your system)
            try:
                font = ImageFont.truetype("arial.ttf", int(min(width, height) / 20))
            except:
                font = ImageFont.load_default()
            
            text_width, text_height = draw.textsize(WATERMARK_TEXT, font)
            x = (width - text_width) / 2
            y = (height - text_height) / 2
            
            # Add semi-transparent white text with black outline
            draw.text((x-1, y-1), WATERMARK_TEXT, font=font, fill=(0,0,0,128))
            draw.text((x+1, y-1), WATERMARK_TEXT, font=font, fill=(0,0,0,128))
            draw.text((x-1, y+1), WATERMARK_TEXT, font=font, fill=(0,0,0,128))
            draw.text((x+1, y+1), WATERMARK_TEXT, font=font, fill=(0,0,0,128))
            draw.text((x, y), WATERMARK_TEXT, font=font, fill=(255,255,255,128))
            
            img.save(output_path)
    except Exception as e:
        print(f"Watermark error: {e}")
        # If watermark fails, just copy the original
        import shutil
        shutil.copy(image_path, output_path)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"success": False, "error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    event_id = request.form.get('event_id', 'default_event')
    
    if not files or len(files) == 0:
        return jsonify({"success": False, "error": "No files selected"}), 400
    
    # Create event directory
    event_dir = os.path.join(app.config['UPLOAD_FOLDER'], event_id)
    os.makedirs(event_dir, exist_ok=True)
    
    # Save all uploaded files
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(event_dir, filename))
    
    # Process images in background
    threading.Thread(target=process_images, args=(event_id,)).start()
    
    return jsonify({
        "success": True,
        "message": f"{len(files)} files uploaded. Processing started.",
        "event_id": event_id
    })

def process_images(event_id):
    try:
        input_dir = os.path.join(app.config['UPLOAD_FOLDER'], event_id)
        output_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id)
        
        # Process each image in the event directory
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)
                
                # Load image and find faces
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                # For each face found
                for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                    # Check if we've seen this face before
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    
                    if True in matches:
                        # Existing person
                        person_id = known_ids[matches.index(True)]
                    else:
                        # New person
                        person_id = f"person_{len(known_encodings)+1:03d}"
                        known_encodings.append(face_encoding)
                        known_ids.append(person_id)
                    
                    # Create person directories if they don't exist
                    person_dir = os.path.join(output_dir, person_id)
                    os.makedirs(person_dir, exist_ok=True)
                    os.makedirs(os.path.join(person_dir, "individual"), exist_ok=True)
                    os.makedirs(os.path.join(person_dir, "group"), exist_ok=True)
                    
                    # Save the face crop (individual photo if only one face)
                    top, right, bottom, left = face_location
                    face_image = image[top:bottom, left:right]
                    face_filename = f"{filename.rsplit('.', 1)[0]}_face_{i}.jpg"
                    
                    if len(face_encodings) == 1:
                        # Single face - save as individual photo
                        output_path = os.path.join(person_dir, "individual", face_filename)
                        cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                    else:
                        # Multiple faces - save as group photo
                        output_path = os.path.join(person_dir, "group", face_filename)
                        cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                
                # Save the full image with all faces (group photo)
                if len(face_encodings) > 0:
                    for person_id in set(known_ids[i] for i, match in enumerate(matches) if match):
                        group_dir = os.path.join(output_dir, person_id, "group")
                        os.makedirs(group_dir, exist_ok=True)
                        output_path = os.path.join(group_dir, filename)
                        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        
                        # Create watermarked version
                        watermarked_path = os.path.join(group_dir, f"watermarked_{filename}")
                        add_watermark(output_path, watermarked_path)
        
        print(f"Finished processing event: {event_id}")
    except Exception as e:
        print(f"Error processing images: {e}")

@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        # Get base64 image from request
        image_data = request.json.get('image')
        event_id = request.json.get('event_id', 'default_event')
        
        if not image_data:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        # Convert to OpenCV format
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get face encoding
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        
        if not face_locations:
            return jsonify({"success": False, "error": "No face detected"}), 400
            
        face_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
        
        # Compare with stored encodings
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        
        if True in matches:
            person_id = known_ids[matches.index(True)]
            event_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id, person_id)
            
            if not os.path.exists(event_dir):
                return jsonify({"success": False, "error": "No photos found for this event"}), 404
            
            # Get list of photos
            individual_photos = []
            group_photos = []
            
            individual_dir = os.path.join(event_dir, "individual")
            if os.path.exists(individual_dir):
                individual_photos = [f for f in os.listdir(individual_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            group_dir = os.path.join(event_dir, "group")
            if os.path.exists(group_dir):
                group_photos = [f for f in os.listdir(group_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith('watermarked_')]
            
            return jsonify({
                "success": True,
                "person_id": person_id,
                "individual_photos": individual_photos,
                "group_photos": group_photos,
                "event_id": event_id
            })
        else:
            return jsonify({"success": False, "error": "No match found"}), 404
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/photos/<event_id>/<person_id>/<photo_type>/<filename>')
def get_photo(event_id, person_id, photo_type, filename):
    photo_path = os.path.join(app.config['PROCESSED_FOLDER'], event_id, person_id, photo_type)
    return send_from_directory(photo_path, filename)

@app.route('/download/<event_id>/<person_id>/<photo_type>/<filename>')
def download_photo(event_id, person_id, photo_type, filename):
    # Remove 'watermarked_' prefix if present
    original_filename = filename.replace('watermarked_', '')
    photo_path = os.path.join(app.config['PROCESSED_FOLDER'], event_id, person_id, photo_type)
    return send_from_directory(photo_path, original_filename, as_attachment=True)

def cleanup_old_events():
    while True:
        try:
            threshold = datetime.now() - timedelta(days=30)
            for event in os.listdir(app.config['UPLOAD_FOLDER']):
                event_path = os.path.join(app.config['UPLOAD_FOLDER'], event)
                event_time = datetime.fromtimestamp(os.path.getmtime(event_path))
                if event_time < threshold:
                    shutil.rmtree(event_path)
            
            for event in os.listdir(app.config['PROCESSED_FOLDER']):
                event_path = os.path.join(app.config['PROCESSED_FOLDER'], event)
                event_time = datetime.fromtimestamp(os.path.getmtime(event_path))
                if event_time < threshold:
                    shutil.rmtree(event_path)
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        time.sleep(24 * 60 * 60)  # Run once per day

if __name__ == '__main__':
    # Start cleanup thread
    threading.Thread(target=cleanup_old_events, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)