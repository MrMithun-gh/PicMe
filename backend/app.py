from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for
from functools import wraps
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
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

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

known_encodings, known_ids = [], []
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --- HELPER FUNCTIONS ---
def get_db_connection():
    try: return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err: print(f"DB Error: {err}"); return None

def load_all_known_faces(file=KNOWN_FACES_DATA_PATH):
    if os.path.exists(file):
        try:
            with open(file, 'rb') as f: return pickle.load(f)
        except Exception: return [], []
    return [], []

def save_all_known_faces(encodings, ids, file=KNOWN_FACES_DATA_PATH):
    with open(file, 'wb') as f: pickle.dump((encodings, ids), f)

initial_known_encodings, initial_known_ids = load_all_known_faces()
known_encodings.extend(initial_known_encodings)
known_ids.extend(initial_known_ids)

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
        global known_encodings, known_ids
        print(f"--- [LOG] Starting processing for event: {event_id} ---")
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)
                print(f"--- [LOG] Processing image: {filename}")
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings_in_image = face_recognition.face_encodings(image)
                    print(f"--- [LOG] Found {len(face_encodings_in_image)} face(s) in {filename}")
                    
                    person_ids_in_image = set()
                    for face_encoding in face_encodings_in_image:
                        matches = face_recognition.compare_faces(known_encodings, face_encoding)
                        person_id = None
                        if True in matches:
                            person_id = known_ids[matches.index(True)]
                        else:
                            person_id = f"person_{len(known_encodings) + 1:04d}"
                            known_encodings.append(face_encoding)
                            known_ids.append(person_id)
                            print(f"--- [LOG] New person found! ID: {person_id}")
                        person_ids_in_image.add(person_id)

                    if len(face_encodings_in_image) > 0:
                        for pid in person_ids_in_image:
                            person_dir = os.path.join(output_dir, pid)
                            individual_dir = os.path.join(person_dir, "individual")
                            group_dir = os.path.join(person_dir, "group")
                            os.makedirs(individual_dir, exist_ok=True)
                            os.makedirs(group_dir, exist_ok=True)

                            if len(face_encodings_in_image) == 1:
                                shutil.copy(image_path, os.path.join(individual_dir, filename))
                                print(f"--- [LOG] Saved '{filename}' to INDIVIDUAL folder for {pid}")
                            
                            original_dest = os.path.join(group_dir, filename)
                            watermarked_dest = os.path.join(group_dir, f"watermarked_{filename}")
                            shutil.copy(image_path, original_dest)
                            shutil.copy(image_path, watermarked_dest)
                            print(f"--- [LOG] Saved watermarked '{filename}' to GROUP folder for {pid}")

                except Exception as e:
                    print(f"  -> ERROR processing {filename}: {e}")
        save_all_known_faces(known_encodings, known_ids)
        print(f"--- [LOG] Finished processing event: {event_id}. Total known faces: {len(known_encodings)} ---")
    except Exception as e:
        print(f"  -> FATAL ERROR during processing for event {event_id}: {e}")

# --- ROUTES FOR SERVING ALL HTML PAGES ---
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

# --- AUTHENTICATION API ROUTES ---
@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    full_name, email, password = data.get('fullName'), data.get('email'), data.get('password')
    if not all([full_name, email, password]): return jsonify({"success": False, "error": "All fields are required"}), 400
    hashed_password = generate_password_hash(password)
    conn = get_db_connection()
    if conn is None:
        return jsonify({"success": False, "error": "Database connection failed"}), 500
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
    with open(EVENTS_DATA_PATH, 'r') as f: return jsonify(json.load(f))

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
        if not face_locations: return jsonify({"success": False, "error": "No face detected in the scan."}), 400
        scanned_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
        matches = face_recognition.compare_faces(known_encodings, scanned_encoding, tolerance=0.55)
        if True in matches:
            person_id = known_ids[matches.index(True)]
            person_dir = os.path.join(app.config['PROCESSED_FOLDER'], event_id, person_id)
            if not os.path.exists(person_dir): return jsonify({"success": False, "error": "Match found, but no photos in this event."}), 404
            individual_dir = os.path.join(person_dir, "individual")
            group_dir = os.path.join(person_dir, "group")
            individual_photos = [f for f in os.listdir(individual_dir)] if os.path.exists(individual_dir) else []
            group_photos = [f for f in os.listdir(group_dir) if f.startswith('watermarked_')] if os.path.exists(group_dir) else []
            return jsonify({"success": True, "person_id": person_id, "individual_photos": individual_photos, "group_photos": group_photos, "event_id": event_id})
        else:
            return jsonify({"success": False, "error": "No match found in our records."}), 404
    except Exception as e:
        print(f"RECOGNIZE ERROR: {e}")
        return jsonify({"success": False, "error": "An internal error occurred."}), 500

# --- MAIN EXECUTION BLOCK ---
def process_existing_uploads_on_startup():
    print("--- [LOG] Checking for existing photos to process on startup... ---")
    if os.path.exists(UPLOAD_FOLDER):
        for event_id in os.listdir(UPLOAD_FOLDER):
            event_path = os.path.join(UPLOAD_FOLDER, event_id)
            if os.path.isdir(event_path):
                threading.Thread(target=process_images, args=(event_id,)).start()

if __name__ == '__main__':
    if not os.path.exists(EVENTS_DATA_PATH):
        # Your code to create dummy events json file
        pass
    process_existing_uploads_on_startup()
    app.run(host='0.0.0.0', port=5000, debug=True)
