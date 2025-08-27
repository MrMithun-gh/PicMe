from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for
from functools import wraps
import os
import base64
import numpy as np
import cv2
import face_recognition
import shutil
import threading
import json
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import qrcode
from io import BytesIO
import uuid
from datetime import datetime

# NEW: Import the model
from face_model import FaceRecognitionModel

# --- CONFIGURATION ---
app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/pages')
app.secret_key = 'your_super_secret_key_here'
DB_CONFIG = {'host': 'localhost', 'user': 'root', 'password': '', 'database': 'picme_db'}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, '..', 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, '..', 'processed')
EVENTS_DATA_PATH = os.path.join(BASE_DIR, '..', 'events_data.json')
KNOWN_FACES_DATA_PATH = os.path.join(BASE_DIR, 'known_faces.dat')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --- INITIALIZE THE ML MODEL ---
model = FaceRecognitionModel(data_file=KNOWN_FACES_DATA_PATH)

# --- HELPER FUNCTIONS ---
def get_db_connection():
    try: return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err: print(f"DB Error: {err}"); return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'): return redirect(url_for('serve_login_page'))
        return f(*args, **kwargs)
    return decorated_function

def process_images(event_id):
    try:
        input_dir = os.path.join(app.config['UPLOAD_FOLDER'], event_id)
        output_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"--- [PROCESS] Starting for event: {event_id} ---")
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and not filename.endswith('_qr.png'):
                image_path = os.path.join(input_dir, filename)
                print(f"--- [PROCESS] Image: {filename}")
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    print(f"--- [PROCESS] Found {len(face_encodings)} face(s) in {filename}")
                    
                    person_ids_in_image = {model.learn_face(encoding) for encoding in face_encodings}

                    if len(face_encodings) > 0:
                        for pid in person_ids_in_image:
                            person_dir = os.path.join(output_dir, pid)
                            os.makedirs(os.path.join(person_dir, "individual"), exist_ok=True)
                            os.makedirs(os.path.join(person_dir, "group"), exist_ok=True)

                            if len(face_encodings) == 1:
                                shutil.copy(image_path, os.path.join(person_dir, "individual", filename))
                            
                            shutil.copy(image_path, os.path.join(person_dir, "group", f"watermarked_{filename}"))

                except Exception as e:
                    print(f"  -> ERROR processing {filename}: {e}")
        
        model.save_model() # Save any newly learned faces
        print(f"--- [PROCESS] Finished for event: {event_id} ---")
    except Exception as e:
        print(f"  -> FATAL ERROR during processing for event {event_id}: {e}")

# --- ROUTES FOR SERVING PAGES ---
@app.route('/')
def serve_index(): return render_template('index.html')
@app.route('/login')
def serve_login_page(): return render_template('login.html')
@app.route('/signup')
def serve_signup_page(): return render_template('signup.html')
@app.route('/homepage')
@login_required
def serve_homepage(): return render_template('homepage.html')
@app.route('/event_discovery')
@login_required
def serve_event_discovery(): return render_template('event_discovery.html')
@app.route('/event_detail')
@login_required
def serve_event_detail(): return render_template('event_detail.html')
@app.route('/biometric_authentication_portal')
@login_required
def serve_biometric_authentication_portal(): return render_template('biometric_authentication_portal.html')
@app.route('/personal_photo_gallery')
@login_required
def serve_personal_photo_gallery(): return render_template('personal_photo_gallery.html')

# NEW: Event organizer page route
@app.route('/event_organizer')
@login_required
def serve_event_organizer(): return render_template('event_organizer.html')

# --- AUTHENTICATION API ROUTES ---
@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    full_name, email, password = data.get('fullName'), data.get('email'), data.get('password')
    if not all([full_name, email, password]): return jsonify({"success": False, "error": "All fields are required"}), 400
    hashed_password = generate_password_hash(password)
    conn = get_db_connection()
    if conn is None: return jsonify({"success": False, "error": "Database connection failed"}), 500
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone(): return jsonify({"success": False, "error": "Email already registered"}), 409
        cursor.execute("INSERT INTO users (full_name, email, password) VALUES (%s, %s, %s)", (full_name, email, hashed_password))
        conn.commit()
        return jsonify({"success": True, "message": "Registration successful!"}), 201
    except mysql.connector.Error as err:
        conn.rollback(); return jsonify({"success": False, "error": "Registration failed"}), 500
    finally:
        cursor.close(); conn.close()

@app.route('/login', methods=['POST'])
def login_user():
    data = request.get_json()
    email, password = data.get('email'), data.get('password')
    if not all([email, password]): return jsonify({"success": False, "error": "Email and password are required"}), 400
    conn = get_db_connection()
    if conn is None: return jsonify({"success": False, "error": "Database connection failed"}), 500
    cursor = conn.cursor(dictionary=True)
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
        return jsonify({"success": False, "error": "An internal server error occurred during login."}), 500
    finally:
        cursor.close(); conn.close()

@app.route('/logout')
def logout_user():
    session.clear()
    return redirect(url_for('serve_index'))

# --- CORE API & FILE SERVING ROUTES ---
@app.route('/events', methods=['GET'])
def get_events():
    try:
        with open(EVENTS_DATA_PATH, 'r') as f:
            events_data = json.load(f)
        return jsonify(events_data)
    except FileNotFoundError:
        return jsonify([])  # Return empty list if file doesn't exist
    except Exception as e:
        print(f"Error loading events: {e}")
        return jsonify([])

@app.route('/recognize', methods=['POST'])
@login_required
def recognize_face():
    try:
        data = request.get_json()
        image_data = data.get('image')
        event_id = data.get('event_id', 'default_event')
        if not image_data: return jsonify({"success": False, "error": "No image provided"}), 400
        
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img)
        if not face_locations: return jsonify({"success": False, "error": "No face detected in scan."}), 400
        
        scanned_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
        
        # Use the new, accurate model for recognition
        person_id = model.recognize_face(scanned_encoding)
        
        if person_id:
            person_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id, person_id)
            if not os.path.exists(person_dir): return jsonify({"success": False, "error": "Match found, but no photos in this event."}), 404
            
            individual_dir = os.path.join(person_dir, "individual")
            group_dir = os.path.join(person_dir, "group")
            individual_photos = [f for f in os.listdir(individual_dir)] if os.path.exists(individual_dir) else []
            group_photos = [f for f in os.listdir(group_dir) if f.startswith('watermarked_')] if os.path.exists(group_dir) else []
            
            return jsonify({"success": True, "person_id": person_id, "individual_photos": individual_photos, "group_photos": group_photos, "event_id": event_id})
        else:
            return jsonify({"success": False, "error": "No confident match found."}), 404

    except Exception as e:
        print(f"RECOGNIZE ERROR: {e}")
        return jsonify({"success": False, "error": "An internal error occurred."}), 500

# --- EVENT ORGANIZER API ROUTES ---
@app.route('/api/create_event', methods=['POST'])
@login_required
def create_event():
    try:
        data = request.get_json()
        event_name = data.get('eventName')
        event_location = data.get('eventLocation')
        event_date = data.get('eventDate')
        event_category = data.get('eventCategory', 'General')
        
        if not all([event_name, event_location, event_date]):
            return jsonify({"success": False, "error": "All fields are required"}), 400
        
        # Generate unique event ID
        event_id = f"event_{uuid.uuid4().hex[:8]}"
        
        # Create event directory structure
        event_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], event_id)
        event_processed_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id)
        os.makedirs(event_upload_dir, exist_ok=True)
        os.makedirs(event_processed_dir, exist_ok=True)
        
        # Generate QR code for the event
        qr_data = f"http://localhost:5000/event_detail?event_id={event_id}"  # Update with your domain
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_data)
        qr.make(fit=True)
        
        # Save QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_path = os.path.join(event_upload_dir, f"{event_id}_qr.png")
        qr_img.save(qr_path)
        
        # Load existing events data
        if os.path.exists(EVENTS_DATA_PATH):
            with open(EVENTS_DATA_PATH, 'r') as f:
                events_data = json.load(f)
        else:
            events_data = []
        
        # Add new event
        new_event = {
            "id": event_id,
            "name": event_name,
            "location": event_location,
            "date": event_date,
            "category": event_category,
            "image": "/static/images/default_event.jpg",
            "photos_count": 0,
            "qr_code": f"/api/qr_code/{event_id}",
            "created_by": session.get('user_id'),
            "created_at": datetime.now().isoformat(),
            "sample_photos": []
        }
        
        events_data.append(new_event)
        
        # Save updated events data
        with open(EVENTS_DATA_PATH, 'w') as f:
            json.dump(events_data, f, indent=2)
        
        return jsonify({"success": True, "event_id": event_id, "message": "Event created successfully!"}), 201
        
    except Exception as e:
        print(f"Error creating event: {e}")
        return jsonify({"success": False, "error": "Failed to create event"}), 500

@app.route('/api/qr_code/<event_id>')
def get_qr_code(event_id):
    qr_path = os.path.join(app.config['UPLOAD_FOLDER'], event_id, f"{event_id}_qr.png")
    if os.path.exists(qr_path):
        return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], event_id), f"{event_id}_qr.png")
    return "QR Code not found", 404

@app.route('/api/upload_photos/<event_id>', methods=['POST'])
@login_required
def upload_event_photos(event_id):
    try:
        if 'photos' not in request.files:
            return jsonify({"success": False, "error": "No photos uploaded"}), 400
        
        files = request.files.getlist('photos')
        if not files or files[0].filename == '':
            return jsonify({"success": False, "error": "No photos selected"}), 400
        
        event_dir = os.path.join(app.config['UPLOAD_FOLDER'], event_id)
        if not os.path.exists(event_dir):
            return jsonify({"success": False, "error": "Event not found"}), 404
        
        uploaded_files = []
        for file in files:
            if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
                file_path = os.path.join(event_dir, filename)
                file.save(file_path)
                uploaded_files.append(filename)
        
        # Start processing photos in background
        threading.Thread(target=process_images, args=(event_id,)).start()
        
        # Update photo count in events data
        if os.path.exists(EVENTS_DATA_PATH):
            with open(EVENTS_DATA_PATH, 'r') as f:
                events_data = json.load(f)
            
            for event in events_data:
                if event['id'] == event_id:
                    event['photos_count'] += len(uploaded_files)
                    break
            
            with open(EVENTS_DATA_PATH, 'w') as f:
                json.dump(events_data, f, indent=2)
        
        return jsonify({
            "success": True, 
            "message": f"Successfully uploaded {len(uploaded_files)} photos",
            "uploaded_files": uploaded_files
        }), 200
        
    except Exception as e:
        print(f"Error uploading photos: {e}")
        return jsonify({"success": False, "error": "Failed to upload photos"}), 500

@app.route('/api/my_events')
@login_required
def get_my_events():
    try:
        if os.path.exists(EVENTS_DATA_PATH):
            with open(EVENTS_DATA_PATH, 'r') as f:
                all_events = json.load(f)
            
            # Filter events created by current user
            user_events = [event for event in all_events if event.get('created_by') == session.get('user_id')]
            return jsonify({"success": True, "events": user_events})
        
        return jsonify({"success": True, "events": []})
        
    except Exception as e:
        print(f"Error fetching events: {e}")
        return jsonify({"success": False, "error": "Failed to fetch events"}), 500

# --- EXISTING FILE SERVING ROUTES ---
@app.route('/api/events/<event_id>/photos', methods=['GET'])
def get_event_photos(event_id):
    event_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id)
    if not os.path.exists(event_dir):
        return jsonify({"success": False, "error": "No photos found for this event yet."}), 404
    unique_photos = set()
    for person_id in os.listdir(event_dir):
        group_dir = os.path.join(event_dir, person_id, "group")
        if os.path.exists(group_dir):
            for filename in os.listdir(group_dir):
                if filename.startswith('watermarked_'):
                    unique_photos.add(filename)
    photo_urls = [f"/photos/{event_id}/all/{filename}" for filename in sorted(list(unique_photos))]
    return jsonify({"success": True, "photos": photo_urls})

@app.route('/photos/<event_id>/all/<filename>')
def get_public_photo(event_id, filename):
    event_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id)
    for person_id in os.listdir(event_dir):
        photo_path = os.path.join(event_dir, person_id, "group", filename)
        if os.path.exists(photo_path):
            return send_from_directory(os.path.join(event_dir, person_id, "group"), filename)
    return "File Not Found", 404

@app.route('/photos/<event_id>/<person_id>/<photo_type>/<filename>')
@login_required
def get_private_photo(event_id, person_id, photo_type, filename):
    photo_path = os.path.join(app.config['PROCESSED_FOLDER'], event_id, person_id, photo_type)
    return send_from_directory(photo_path, filename)

# --- MAIN EXECUTION BLOCK ---
def process_existing_uploads_on_startup():
    print("--- [LOG] Checking for existing photos on startup... ---")
    if os.path.exists(UPLOAD_FOLDER):
        for event_id in os.listdir(UPLOAD_FOLDER):
            if os.path.isdir(os.path.join(UPLOAD_FOLDER, event_id)):
                threading.Thread(target=process_images, args=(event_id,)).start()

if __name__ == '__main__':
    if not os.path.exists(EVENTS_DATA_PATH): pass
    process_existing_uploads_on_startup()
    app.run(host='0.0.0.0', port=5000, debug=True)