import face_recognition
import numpy as np
import os
import pickle

class FaceRecognitionModel:
    def __init__(self, data_file='known_faces.dat'):
        """
        Initializes the model, loading known faces from a file if it exists.
        """
        self.data_file = data_file
        self.known_encodings = []
        self.known_ids = []
        self.load_model()

    def load_model(self):
        """Loads the known faces and IDs from the data file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    self.known_encodings, self.known_ids = pickle.load(f)
                print(f"--- [ML MODEL] Loaded {len(self.known_ids)} known faces. ---")
            except Exception as e:
                print(f"--- [ML MODEL] Error loading model data: {e}. Starting fresh. ---")

    def save_model(self):
        """Saves the current known faces and IDs to the data file."""
        with open(self.data_file, 'wb') as f:
            pickle.dump((self.known_encodings, self.known_ids), f)
        print(f"--- [ML MODEL] Model saved with {len(self.known_ids)} faces. ---")

    def learn_face(self, new_encoding):
        """
        Learns a new face. If the face is already known, it returns the existing ID.
        If the face is new, it assigns a new ID and returns it.
        """
        if not self.known_encodings:
            # This is the first face ever.
            new_id = f"person_{len(self.known_ids) + 1:04d}"
            self.known_encodings.append(new_encoding)
            self.known_ids.append(new_id)
            print(f"--- [ML MODEL] Learned first face. Assigned ID: {new_id} ---")
            return new_id

        # See if this face is already in our known faces
        face_distances = face_recognition.face_distance(self.known_encodings, new_encoding)
        best_match_index = np.argmin(face_distances)

        # A very strict tolerance to decide if this is an existing person
        if face_distances[best_match_index] < 0.5:
            # This is an existing person
            return self.known_ids[best_match_index]
        else:
            # This is a new person
            new_id = f"person_{len(self.known_ids) + 1:04d}"
            self.known_encodings.append(new_encoding)
            self.known_ids.append(new_id)
            print(f"--- [ML MODEL] Learned a new face. Assigned ID: {new_id} ---")
            return new_id

    def recognize_face(self, scanned_encoding):
        """
        Recognizes a face using a strict tolerance. Returns the person's ID on a
        confident match, otherwise returns None.
        """
        if not self.known_encodings:
            return None

        face_distances = face_recognition.face_distance(self.known_encodings, scanned_encoding)
        best_match_index = np.argmin(face_distances)

        # HIGH-ACCURACY THRESHOLD: Only a very close match is accepted.
        STRICT_TOLERANCE = 0.5

        if face_distances[best_match_index] <= STRICT_TOLERANCE:
            person_id = self.known_ids[best_match_index]
            print(f"--- [ML MODEL] Confident match for {person_id} with distance {face_distances[best_match_index]:.2f} ---")
            return person_id
        else:
            print(f"--- [ML MODEL] No confident match. Best distance was {face_distances[best_match_index]:.2f} (Threshold: {STRICT_TOLERANCE}) ---")
            return None
