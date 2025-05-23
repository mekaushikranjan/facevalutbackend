import dlib
import os
import json
import uuid
from PIL import Image
import numpy as np
import logging
import cv2
import face_recognition_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        self.setup_dirs()
        self.setup_models()
    
    def setup_dirs(self):
        """Create necessary directories for storing face images."""
        try:
            os.makedirs("faces", exist_ok=True)
            logger.info("Created faces directory")
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise
    
    def setup_models(self):
        """Initialize dlib models."""
        try:
            # Initialize face detector
            self.detector = dlib.get_frontal_face_detector()
            
            # Initialize face landmark predictor
            predictor_path = face_recognition_models.pose_predictor_model_location()
            self.predictor = dlib.shape_predictor(predictor_path)
            
            # Initialize face recognition model
            model_path = face_recognition_models.face_recognition_model_location()
            self.face_rec_model = dlib.face_recognition_model_v1(model_path)
            
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def detect_faces(self, image_path):
        """
        Detect faces in an image and return their locations and encodings.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            list: List of dictionaries containing face information
        """
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to RGB (dlib uses RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            logger.info("Detecting faces...")
            face_rects = self.detector(rgb_image)
            logger.info(f"Found {len(face_rects)} faces")
            
            # Process each face
            faces = []
            for i, rect in enumerate(face_rects):
                try:
                    # Get face landmarks
                    shape = self.predictor(rgb_image, rect)
                    
                    # Get face encoding
                    face_descriptor = np.array(self.face_rec_model.compute_face_descriptor(rgb_image, shape))
                    
                    # Create a unique ID for this face
                    face_id = str(uuid.uuid4())
                    
                    # Extract the face image
                    top, left = rect.top(), rect.left()
                    bottom, right = rect.bottom(), rect.right()
                    face_image = image[top:bottom, left:right]
                    
                    # Save the face image
                    face_path = os.path.join("faces", f"{face_id}.jpg")
                    cv2.imwrite(face_path, face_image)
                    
                    # Add face data to results
                    face_data = {
                        "id": face_id,
                        "location": {
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                            "left": left
                        },
                        "path": face_path,
                        "encoding": face_descriptor.tolist()
                    }
                    faces.append(face_data)
                    logger.info(f"Processed face {i+1}/{len(face_rects)}")
                    
                except Exception as e:
                    logger.error(f"Error processing face {i+1}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(faces)} faces")
            return faces
        
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {str(e)}")
            return []
    
    def compare_faces(self, known_face_encodings, face_encoding, tolerance=0.6):
        """
        Compare a face encoding with a list of known face encodings.
        
        Args:
            known_face_encodings (list): List of known face encodings
            face_encoding (list): Face encoding to compare
            tolerance (float): Face distance threshold
            
        Returns:
            list: List of boolean values indicating matches
        """
        try:
            # Convert to numpy arrays
            known_encodings = np.array(known_face_encodings)
            face_encoding = np.array(face_encoding)
            
            # Calculate Euclidean distances
            distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
            
            # Return True for distances below tolerance
            return (distances <= tolerance).tolist()
            
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            return []
