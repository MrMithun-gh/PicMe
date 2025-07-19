from flask import Flask, request, jsonify, send_from_directory, render_template
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
import json
import pickle

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/pages')

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

# Dummy data for events (replace with a database in production)
EVENTS_DATA_PATH = 'events_data.json'
KNOWN_FACES_DATA_PATH = 'known_faces.dat'

# Load known faces from file on startup
def load_all_known_faces(encodings_file=KNOWN_FACES_DATA_PATH):
    if os.path.exists(encodings_file):
        try:
            with open(encodings_file, 'rb') as f:
                known_encodings_loaded, known_ids_loaded = pickle.load(f)
            print(f"Loaded {len(known_encodings_loaded)} known faces from {encodings_file}")
            return known_encodings_loaded, known_ids_loaded
        except Exception as e:
            print(f"Error loading known faces from {encodings_file}: {e}. Starting fresh.")
            return [], []
    print(f"No known faces file found at {encodings_file}. Starting fresh.")
    return [], []

# Save known faces to file
def save_all_known_faces(encodings, ids, encodings_file=KNOWN_FACES_DATA_PATH):
    try:
        with open(encodings_file, 'wb') as f:
            pickle.dump((encodings, ids), f)
        print(f"Saved {len(encodings)} known faces to {encodings_file}")
    except Exception as e:
        print(f"Error saving known faces to {encodings_file}: {e}")

# Initialize known faces on app startup
initial_known_encodings, initial_known_ids = load_all_known_faces()
known_encodings.extend(initial_known_encodings)
known_ids.extend(initial_known_ids)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_watermark(image_path, output_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            draw = ImageDraw.Draw(img)

            # Use a default font (you might need to adjust this for your system)
            try:
                # Attempt to load a common font, adjust path if necessary for your OS
                font_path = "arial.ttf" # Or provide full path like "C:/Windows/Fonts/arial.ttf"
                font_size = int(min(width, height) / 20)
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                # Fallback to default PIL font if arial.ttf is not found
                print(f"Warning: Could not load {font_path}. Using default font.")
                font = ImageFont.load_default()

            # Ensure text_width and text_height are calculated correctly for the chosen font
            try:
                # For Pillow 9.0.0 and later
                bbox = draw.textbbox((0, 0), WATERMARK_TEXT, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                # Fallback for older Pillow versions
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
        print(f"Watermark error: {e}. Copying original image instead.")
        shutil.copy(image_path, output_path)

# --- Routes for serving HTML pages ---
@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/signup')
def serve_signup_page():
    return render_template('signup.html')

@app.route('/login')
def serve_login_page():
    return render_template('login.html')

@app.route('/homepage')
def serve_homepage():
    return render_template('homepage.html')

@app.route('/event_discovery')
def serve_event_discovery():
    return render_template('event_discovery.html')

@app.route('/biometric_authentication_portal')
def serve_biometric_authentication_portal():
    return render_template('biometric_authentication_portal.html')

@app.route('/personal_photo_gallery')
def serve_personal_photo_gallery():
    return render_template('personal_photo_gallery.html')

@app.route('/download_center')
def serve_download_center():
    return render_template('download_center.html')

@app.route('/event_organizer_hub')
def serve_event_organizer_hub():
    return render_template('event_organizer_hub.html')

# Generic route for .html files (can be kept as a fallback or removed if all pages have specific routes)
@app.route('/<page_name>.html')
def serve_html_page(page_name):
    template_path = os.path.join(app.template_folder, f'{page_name}.html')
    if os.path.exists(template_path):
        return render_template(f'{page_name}.html')
    else:
        return "Page Not Found", 404

# --- API Endpoints ---
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

    uploaded_filenames = []
    # Save all uploaded files
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(event_dir, filename))
            uploaded_filenames.append(filename)

    # Process images in background
    threading.Thread(target=process_images, args=(event_id,)).start()

    return jsonify({
        "success": True,
        "message": f"{len(files)} files uploaded. Processing started.",
        "event_id": event_id,
        "uploaded_filenames": uploaded_filenames
    })

def process_images(event_id):
    try:
        input_dir = os.path.join(app.config['UPLOAD_FOLDER'], event_id)
        output_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id)

        # Ensure output directory for event exists
        os.makedirs(output_dir, exist_ok=True)

        global known_encodings, known_ids # Declare global to modify the in-memory lists

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)

                try:
                    # Load image and find faces
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image)
                    face_encodings_in_image = face_recognition.face_encodings(image, face_locations)
                except Exception as e:
                    print(f"Error processing image {filename}: {e}")
                    continue # Skip to next image

                # Track which person_ids are found in this specific image
                person_ids_in_current_image = set()

                # For each face found in the current image
                for i, (face_encoding, face_location) in enumerate(zip(face_encodings_in_image, face_locations)):
                    # Check if we've seen this face before
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)

                    person_id = None
                    if True in matches:
                        # Existing person
                        person_id = known_ids[matches.index(True)]
                    else:
                        # New person
                        person_id = f"person_{len(known_encodings)+1:03d}"
                        known_encodings.append(face_encoding)
                        known_ids.append(person_id)

                    person_ids_in_current_image.add(person_id)

                    # Create person directories if they don't exist
                    person_dir = os.path.join(output_dir, person_id)
                    os.makedirs(person_dir, exist_ok=True)
                    os.makedirs(os.path.join(person_dir, "individual"), exist_ok=True)
                    os.makedirs(os.path.join(person_dir, "group"), exist_ok=True)

                    # Save the face crop
                    top, right, bottom, left = face_location
                    face_image_rgb = image[top:bottom, left:right]
                    if face_image_rgb.size == 0: # Check if the cropped image is empty
                         print(f"Warning: Cropped face image is empty for {filename}, face {i}")
                         continue

                    face_filename = f"{filename.rsplit('.', 1)[0]}_face_{i}.jpg"

                    # Decide where to save based on number of faces in the original image
                    if len(face_encodings_in_image) == 1:
                        # Single face - save as individual photo
                        output_path_face_crop = os.path.join(person_dir, "individual", face_filename)
                        cv2.imwrite(output_path_face_crop, cv2.cvtColor(face_image_rgb, cv2.COLOR_RGB2BGR))
                    else:
                        # Multiple faces - save face crop to group for this person
                        output_path_face_crop = os.path.join(person_dir, "group", f"crop_{face_filename}") # Prefix to distinguish
                        cv2.imwrite(output_path_face_crop, cv2.cvtColor(face_image_rgb, cv2.COLOR_RGB2BGR))

                # After processing all faces in an image, save the full image to each relevant person's group folder
                if len(face_encodings_in_image) > 0:
                    for pid in person_ids_in_current_image:
                        group_dir_for_person = os.path.join(output_dir, pid, "group")
                        os.makedirs(group_dir_for_person, exist_ok=True)

                        # Save the original full image to the group folder
                        original_full_image_path = os.path.join(group_dir_for_person, filename)
                        cv2.imwrite(original_full_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                        # Create watermarked version of the full image
                        watermarked_path = os.path.join(group_dir_for_person, f"watermarked_{filename}")
                        add_watermark(original_full_image_path, watermarked_path)

        # Save the updated known faces after processing
        save_all_known_faces(known_encodings, known_ids)
        print(f"Finished processing event: {event_id}. Known faces updated.")
    except Exception as e:
        print(f"Error processing images for event {event_id}: {e}")

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

        if img is None: # Check if imdecode failed
             return jsonify({"success": False, "error": "Could not decode image"}), 400

        # Get face encoding
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)

        if not face_locations:
            return jsonify({"success": False, "error": "No face detected"}), 400

        # For simplicity, we'll only process the first detected face for recognition
        face_encoding_to_recognize = face_recognition.face_encodings(rgb_img, face_locations)[0]

        # Compare with stored encodings
        global known_encodings, known_ids # Access global lists
        matches = face_recognition.compare_faces(known_encodings, face_encoding_to_recognize)

        if True in matches:
            person_id = known_ids[matches.index(True)]
            # Filter photos by the specific event_id if provided
            event_processed_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id)
            person_event_dir = os.path.join(event_processed_dir, person_id)

            if not os.path.exists(person_event_dir):
                return jsonify({"success": False, "error": f"No photos found for this person ({person_id}) in event {event_id}"}), 404

            # Get list of photos
            individual_photos = []
            group_photos = []

            individual_dir = os.path.join(person_event_dir, "individual")
            if os.path.exists(individual_dir):
                individual_photos = [f for f in os.listdir(individual_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            group_dir = os.path.join(person_event_dir, "group")
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
        print(f"Recognition error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/photos/<event_id>/<person_id>/<photo_type>/<filename>')
def get_photo(event_id, person_id, photo_type, filename):
    photo_path = os.path.join(app.config['PROCESSED_FOLDER'], event_id, person_id, photo_type)
    return send_from_directory(photo_path, filename)

@app.route('/download/<event_id>/<person_id>/<photo_type>/<filename>')
def download_photo(event_id, person_id, photo_type, filename):
    # Remove 'watermarked_' prefix if present to get the original file
    original_filename = filename.replace('watermarked_', '')
    photo_path = os.path.join(app.config['PROCESSED_FOLDER'], event_id, person_id, photo_type)
    
    # Ensure the original file exists for download
    full_original_path = os.path.join(photo_path, original_filename)
    if not os.path.exists(full_original_path):
        return jsonify({"success": False, "error": "Original file not found for download"}), 404

    return send_from_directory(photo_path, original_filename, as_attachment=True)

@app.route('/download-all/<event_id>/<person_id>')
def download_all_photos_zip(event_id, person_id):
    event_person_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id, person_id)
    if not os.path.exists(event_person_dir):
        return jsonify({"success": False, "error": "No photos found for this person in this event"}), 404

    all_files_info = []
    # Collect all individual photos (original, not cropped for group)
    individual_dir = os.path.join(event_person_dir, "individual")
    if os.path.exists(individual_dir):
        for f in os.listdir(individual_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files_info.append({
                    "type": "individual",
                    "filename": f,
                    "url": f"/download/{event_id}/{person_id}/individual/{f}"
                })

    # Collect all original full group photos (non-watermarked)
    group_dir = os.path.join(event_person_dir, "group")
    if os.path.exists(group_dir):
        for f in os.listdir(group_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('watermarked_') and not f.startswith('crop_'):
                all_files_info.append({
                    "type": "group",
                    "filename": f,
                    "url": f"/download/{event_id}/{person_id}/group/{f}"
                })

    if not all_files_info:
        return jsonify({"success": False, "error": "No photos to download"}), 404

    return jsonify({
        "success": True,
        "message": f"Found {len(all_files_info)} files for download. Client should initiate downloads.",
        "files_to_download": all_files_info
    })


# API endpoint to get event data for event_discovery.html
@app.route('/events', methods=['GET'])
def get_events():
    # Load dummy events data from a JSON file
    if os.path.exists(EVENTS_DATA_PATH):
        try:
            with open(EVENTS_DATA_PATH, 'r') as f:
                events_data = json.load(f)
            return jsonify(events_data)
        except Exception as e:
            print(f"Error loading events data from {EVENTS_DATA_PATH}: {e}")
            return jsonify({"error": "Failed to load events data"}), 500
    return jsonify([]) # Return empty list if no data file


def cleanup_old_events():
    while True:
        try:
            threshold = datetime.now() - timedelta(days=30)
            for event in os.listdir(app.config['UPLOAD_FOLDER']):
                event_path = os.path.join(app.config['UPLOAD_FOLDER'], event)
                # Check if it's a directory and get its modification time
                if os.path.isdir(event_path):
                    event_time = datetime.fromtimestamp(os.path.getmtime(event_path))
                    if event_time < threshold:
                        print(f"Cleaning up old upload event: {event_path}")
                        shutil.rmtree(event_path)

            for event in os.listdir(app.config['PROCESSED_FOLDER']):
                event_path = os.path.join(app.config['PROCESSED_FOLDER'], event)
                # Check if it's a directory and get its modification time
                if os.path.isdir(event_path):
                    event_time = datetime.fromtimestamp(os.path.getmtime(event_path))
                    if event_time < threshold:
                        print(f"Cleaning up old processed event: {event_path}")
                        shutil.rmtree(event_path)
        except Exception as e:
            print(f"Cleanup error: {e}")

        time.sleep(24 * 60 * 60)  # Run once per day


if __name__ == '__main__':
    # Start cleanup thread
    threading.Thread(target=cleanup_old_events, daemon=True).start()

    # Create a dummy events_data.json if it doesn't exist
    if not os.path.exists(EVENTS_DATA_PATH):
        dummy_events = [
            {
                "id": "event_1",
                "name": "Electric Dreams Festival",
                "location": "San Francisco, CA",
                "date": "July 15-17, 2025",
                "category": "Festival",
                "image": "/static/images/event1.jpg", # Placeholder path, update if needed
                "photos_count": 1428,
                "sample_photos": [
                    "/static/images/sample1.jpg",
                    "/static/images/sample2.jpg",
                    "/static/images/sample3.jpg"
                ]
            },
            {
                "id": "event_2",
                "name": "Global Tech Summit",
                "location": "Bengaluru, India",
                "date": "August 10-12, 2025",
                "category": "Corporate",
                "image": "/static/images/event2.jpg",
                "photos_count": 892,
                "sample_photos": [
                    "/static/images/sample4.jpg",
                    "/static/images/sample5.jpg",
                    "/static/images/sample6.jpg"
                ]
            },
             {
                "id": "event_3",
                "name": "Annual Charity Gala",
                "location": "Mumbai, India",
                "date": "September 1, 2025",
                "category": "Charity",
                "image": "/static/images/event3.jpg",
                "photos_count": 324,
                "sample_photos": [
                    "/static/images/sample7.jpg",
                    "/static/images/sample8.jpg",
                    "/static/images/sample9.jpg"
                ]
            }
        ]
        with open(EVENTS_DATA_PATH, 'w') as f:
            json.dump(dummy_events, f, indent=4)
        print(f"Created dummy events data at {EVENTS_DATA_PATH}")

    # Ensure static images directory exists for the dummy data
    os.makedirs('frontend/static/images', exist_ok=True)
    # Note: You still need to manually place actual image files here if you want them to display.
    # E.g., picme1/frontend/static/images/event1.jpg, picme1/frontend/static/images/sample1.jpg etc.


    app.run(host='0.0.0.0', port=5000)
