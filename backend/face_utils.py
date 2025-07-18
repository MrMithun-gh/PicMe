import face_recognition
import numpy as np
import os
import pickle

def load_known_faces(encodings_file='known_faces.dat'):
    if os.path.exists(encodings_file):
        with open(encodings_file, 'rb') as f:
            known_encodings, known_ids = pickle.load(f)
        return known_encodings, known_ids
    return [], []

def save_known_faces(encodings, ids, encodings_file='known_faces.dat'):
    with open(encodings_file, 'wb') as f:
        pickle.dump((encodings, ids), f)

def compare_faces(known_encodings, unknown_encoding, tolerance=0.6):
    distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    return list(distances <= tolerance)