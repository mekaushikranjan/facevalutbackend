import cv2
import numpy as np
from PIL import Image
import io

def load_image(image_data):
    """Load image from bytes or file path"""
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    else:
        image = Image.open(image_data)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def detect_faces(image):
    """Detect faces in an image using OpenCV"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def get_face_encoding(image, face_location):
    """Get a simple face encoding using OpenCV features"""
    x, y, w, h = face_location
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (128, 128))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return face.flatten()

def compare_faces(known_face_encoding, face_encoding_to_check, tolerance=0.6):
    """Compare two face encodings"""
    return np.linalg.norm(known_face_encoding - face_encoding_to_check) < tolerance

def process_image(image_data):
    """Process an image and return face locations and encodings"""
    image = load_image(image_data)
    faces = detect_faces(image)
    face_encodings = []
    face_locations = []
    
    for face in faces:
        face_encoding = get_face_encoding(image, face)
        face_encodings.append(face_encoding)
        face_locations.append(face)
    
    return face_locations, face_encodings 