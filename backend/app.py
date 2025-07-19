from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for
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
import mysql.connector # Import MySQL connector
from werkzeug.security import generate_password_hash, check_password_hash # For password hashing

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/pages')

# --- Flask Configuration ---
# Set a secret key for session management.
# IMPORTANT: In a real application, use a strong, randomly generated key
# and store it securely (e.g., in an environment variable).
app.secret_key = 'your_super_secret_key_here_replace_me_in_production'

# Database Configuration (for XAMPP MySQL)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root', # Default XAMPP MySQL username
    'password': '', # Default XAMPP MySQL password (usually empty)
    'database': 'picme_db'
}

# --- Application Specific Configuration ---
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

# --- Database Helper Functions ---
def get_db_connection():
    """Establishes a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Database connection error: {err}")
        return None

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

# --- Utility Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_watermark(image_path, output_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            draw = ImageDraw.Draw(img)

            try:
                font_path = "arial.ttf"
                font_size = int(min(width, height) / 20)
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                print(f"Warning: Could not load {font_path}. Using default font.")
                font = ImageFont.load_default()

            try:
                bbox = draw.textbbox((0, 0), WATERMARK_TEXT, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                text_width, text_height = draw.textsize(WATERMARK_TEXT, font)

            x = (width - text_width) / 2
            y = (height - text_height) / 2

            draw.text((x-1, y-1), WATERMARK_TEXT, font=font, fill=(0,0,0,128))
            draw.text((x+1, y-1), WATERMARK_TEXT, font=font, fill=(0,0,0,128))
            draw.text((x-1, y+1), WATERMARK_TEXT, font=font, fill=(0,0,0,128))
            draw.text((x+1, y+1), WATERMARK_TEXT, font=font, fill=(0,0,0,128))
            draw.text((x, y), WATERMARK_TEXT, font=font, fill=(255,255,255,128))

            img.save(output_path)
    except Exception as e:
        print(f"Watermark error: {e}. Copying original image instead.")
        shutil.copy(image_path, output_path)

# --- Authentication Routes ---
@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    full_name = data.get('fullName')
    email = data.get('email')
    password = data.get('password')

    if not all([full_name, email, password]):
        return jsonify({"success": False, "error": "All fields are required"}), 400

    hashed_password = generate_password_hash(password)

    conn = get_db_connection()
    if conn is None:
        return jsonify({"success": False, "error": "Database connection failed"}), 500

    cursor = conn.cursor()
    try:
        # Check if user already exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({"success": False, "error": "Email already registered"}), 409

        cursor.execute(
            "INSERT INTO users (full_name, email, password) VALUES (%s, %s, %s)",
            (full_name, email, hashed_password)
        )
        conn.commit()
        return jsonify({"success": True, "message": "Registration successful!"}), 201
    except mysql.connector.Error as err:
        print(f"Error during registration: {err}")
        conn.rollback()
        return jsonify({"success": False, "error": "Registration failed"}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/login', methods=['POST'])
def login_user():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({"success": False, "error": "Email and password are required"}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({"success": False, "error": "Database connection failed"}), 500

    cursor = conn.cursor(dictionary=True) # Return rows as dictionaries
    try:
        cursor.execute("SELECT id, email, password FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            return jsonify({"success": True, "message": "Login successful!"}), 200
        else:
            return jsonify({"success": False, "error": "Invalid email or password"}), 401
    except mysql.connector.Error as err:
        print(f"Error during login: {err}")
        return jsonify({"success": False, "error": "Login failed"}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/logout')
def logout_user():
    session.pop('logged_in', None)
    session.pop('user_id', None)
    session.pop('user_email', None)
    return redirect(url_for('serve_index')) # Redirect to homepage after logout

# --- Authentication Decorator (Optional but Recommended) ---
# You can use this to protect routes that require a logged-in user
def login_required(f):
    @app.route.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            # You might want to redirect to login page or return an error
            return redirect(url_for('serve_login_page'))
        return f(*args, **kwargs)
    return decorated_function

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
    # Example of protecting a page
    # if not session.get('logged_in'):
    #     return redirect(url_for('serve_login_page'))
    return render_template('homepage.html')

@app.route('/event_discovery')
def serve_event_discovery():
    return render_template('event_discovery.html')

@app.route('/biometric_authentication_portal')
def serve_biometric_authentication_portal():
    return render_template('biometric_authentication_portal.html')

@app.route('/personal_photo_gallery')
def serve_personal_photo_gallery():
    # This page should ideally be protected
    # if not session.get('logged_in'):
    #     return redirect(url_for('serve_login_page'))
    return render_template('personal_photo_gallery.html')

@app.route('/download_center')
def serve_download_center():
    # This page should ideally be protected
    # if not session.get('logged_in'):
    #     return redirect(url_for('serve_login_page'))
    return render_template('download_center.html')

@app.route('/event_organizer_hub')
def serve_event_organizer_hub():
    # This page should ideally be protected
    # if not session.get('logged_in'):
    #     return redirect(url_for('serve_login_page'))
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
    # This endpoint should ideally be protected for authenticated organizers
    # if not session.get('logged_in'):
    #     return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    if 'files' not in request.files:
        return jsonify({"success": False, "error": "No files provided"}), 400

    files = request.files.getlist('files')
    event_id = request.form.get('event_id', 'default_event')

    if not files or len(files) == 0:
        return jsonify({"success": False, "error": "No files selected"}), 400

    event_dir = os.path.join(app.config['UPLOAD_FOLDER'], event_id)
    os.makedirs(event_dir, exist_ok=True)

    uploaded_filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(event_dir, filename))
            uploaded_filenames.append(filename)

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

        os.makedirs(output_dir, exist_ok=True)

        global known_encodings, known_ids

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)

                try:
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image)
                    face_encodings_in_image = face_recognition.face_encodings(image, face_locations)
                except Exception as e:
                    print(f"Error processing image {filename}: {e}")
                    continue

                person_ids_in_current_image = set()

                for i, (face_encoding, face_location) in enumerate(zip(face_encodings_in_image, face_locations)):
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)

                    person_id = None
                    if True in matches:
                        person_id = known_ids[matches.index(True)]
                    else:
                        person_id = f"person_{len(known_encodings)+1:03d}"
                        known_encodings.append(face_encoding)
                        known_ids.append(person_id)

                    person_ids_in_current_image.add(person_id)

                    person_dir = os.path.join(output_dir, person_id)
                    os.makedirs(person_dir, exist_ok=True)
                    os.makedirs(os.path.join(person_dir, "individual"), exist_ok=True)
                    os.makedirs(os.path.join(person_dir, "group"), exist_ok=True)

                    top, right, bottom, left = face_location
                    face_image_rgb = image[top:bottom, left:right]
                    if face_image_rgb.size == 0:
                         print(f"Warning: Cropped face image is empty for {filename}, face {i}")
                         continue

                    face_filename = f"{filename.rsplit('.', 1)[0]}_face_{i}.jpg"

                    if len(face_encodings_in_image) == 1:
                        output_path_face_crop = os.path.join(person_dir, "individual", face_filename)
                        cv2.imwrite(output_path_face_crop, cv2.cvtColor(face_image_rgb, cv2.COLOR_RGB2BGR))
                    else:
                        output_path_face_crop = os.path.join(person_dir, "group", f"crop_{face_filename}")
                        cv2.imwrite(output_path_face_crop, cv2.cvtColor(face_image_rgb, cv2.COLOR_RGB2BGR))

                if len(face_encodings_in_image) > 0:
                    for pid in person_ids_in_current_image:
                        group_dir_for_person = os.path.join(output_dir, pid, "group")
                        os.makedirs(group_dir_for_person, exist_ok=True)

                        original_full_image_path = os.path.join(group_dir_for_person, filename)
                        cv2.imwrite(original_full_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                        watermarked_path = os.path.join(group_dir_for_person, f"watermarked_{filename}")
                        add_watermark(original_full_image_path, watermarked_path)

        save_all_known_faces(known_encodings, known_ids)
        print(f"Finished processing event: {event_id}. Known faces updated.")
    except Exception as e:
        print(f"Error processing images for event {event_id}: {e}")

@app.route('/recognize', methods=['POST'])
def recognize_face():
    # This endpoint might need to be protected or handle user_id from session
    # if not session.get('logged_in'):
    #     return jsonify({"success": False, "error": "Unauthorized"}), 401

    try:
        image_data = request.json.get('image')
        event_id = request.json.get('event_id', 'default_event')

        if not image_data:
            return jsonify({"success": False, "error": "No image provided"}), 400

        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
             return jsonify({"success": False, "error": "Could not decode image"}), 400

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)

        if not face_locations:
            return jsonify({"success": False, "error": "No face detected"}), 400

        face_encoding_to_recognize = face_recognition.face_encodings(rgb_img, face_locations)[0]

        global known_encodings, known_ids
        matches = face_recognition.compare_faces(known_encodings, face_encoding_to_recognize)

        if True in matches:
            person_id = known_ids[matches.index(True)]
            event_processed_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id)
            person_event_dir = os.path.join(event_processed_dir, person_id)

            if not os.path.exists(person_event_dir):
                return jsonify({"success": False, "error": f"No photos found for this person ({person_id}) in event {event_id}"}), 404

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
    # This endpoint should ideally be protected
    # if not session.get('logged_in'):
    #     return redirect(url_for('serve_login_page')) # Or return 401
    photo_path = os.path.join(app.config['PROCESSED_FOLDER'], event_id, person_id, photo_type)
    return send_from_directory(photo_path, filename)

@app.route('/download/<event_id>/<person_id>/<photo_type>/<filename>')
def download_photo(event_id, person_id, photo_type, filename):
    # This endpoint should ideally be protected
    # if not session.get('logged_in'):
    #     return redirect(url_for('serve_login_page')) # Or return 401

    original_filename = filename.replace('watermarked_', '')
    photo_path = os.path.join(app.config['PROCESSED_FOLDER'], event_id, person_id, photo_type)
    
    full_original_path = os.path.join(photo_path, original_filename)
    if not os.path.exists(full_original_path):
        return jsonify({"success": False, "error": "Original file not found for download"}), 404

    return send_from_directory(photo_path, original_filename, as_attachment=True)

@app.route('/download-all/<event_id>/<person_id>')
def download_all_photos_zip(event_id, person_id):
    # This endpoint should ideally be protected
    # if not session.get('logged_in'):
    #     return redirect(url_for('serve_login_page')) # Or return 401

    event_person_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id, person_id)
    if not os.path.exists(event_person_dir):
        return jsonify({"success": False, "error": "No photos found for this person in this event"}), 404

    all_files_info = []
    individual_dir = os.path.join(event_person_dir, "individual")
    if os.path.exists(individual_dir):
        for f in os.listdir(individual_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files_info.append({
                    "type": "individual",
                    "filename": f,
                    "url": f"/download/{event_id}/{person_id}/individual/{f}"
                })

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


@app.route('/events', methods=['GET'])
def get_events():
    if os.path.exists(EVENTS_DATA_PATH):
        try:
            with open(EVENTS_DATA_PATH, 'r') as f:
                events_data = json.load(f)
            return jsonify(events_data)
        except Exception as e:
            print(f"Error loading events data from {EVENTS_DATA_PATH}: {e}")
            return jsonify({"error": "Failed to load events data"}), 500
    return jsonify([])


def cleanup_old_events():
    while True:
        try:
            threshold = datetime.now() - timedelta(days=30)
            for event in os.listdir(app.config['UPLOAD_FOLDER']):
                event_path = os.path.join(app.config['UPLOAD_FOLDER'], event)
                if os.path.isdir(event_path):
                    event_time = datetime.fromtimestamp(os.path.getmtime(event_path))
                    if event_time < threshold:
                        print(f"Cleaning up old upload event: {event_path}")
                        shutil.rmtree(event_path)

            for event in os.listdir(app.config['PROCESSED_FOLDER']):
                event_path = os.path.join(app.config['PROCESSED_FOLDER'], event)
                if os.path.isdir(event_path):
                    event_time = datetime.fromtimestamp(os.path.getmtime(event_path))
                    if event_time < threshold:
                        print(f"Cleaning up old processed event: {event_path}")
                        shutil.rmtree(event_path)
        except Exception as e:
            print(f"Cleanup error: {e}")

        time.sleep(24 * 60 * 60)


if __name__ == '__main__':
    threading.Thread(target=cleanup_old_events, daemon=True).start()

    if not os.path.exists(EVENTS_DATA_PATH):
        dummy_events = [
            {
                "id": "event_1",
                "name": "Electric Dreams Festival",
                "location": "San Francisco, CA",
                "date": "July 15-17, 2025",
                "category": "Festival",
                "image": "/static/images/event1.jpg",
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

    os.makedirs('frontend/static/images', exist_ok=True)

    app.run(host='0.0.0.0', port=5000, debug=True) # Set debug=True for development
