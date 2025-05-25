from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, GetJsonSchemaHandler
from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime, timedelta
import jwt
import bcrypt
import uuid
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import face_recognition
from PIL import Image as PILImage
import io
import json
import logging
import sys
import uvicorn
from database import (
    users_collection,
    images_collection,
    people_collection,
    albums_collection,
    face_encodings_collection,
    create_indexes
)
from bson import ObjectId
from fastapi.responses import FileResponse
from utils.email import send_otp_email, send_reset_password_email
from utils.otp import generate_otp, verify_otp, store_otp, clear_otp
from pydantic.json_schema import JsonSchemaValue
from motor.motor_asyncio import AsyncIOMotorClient
from face_detector import FaceDetector
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Get port from environment variable or use default
PORT = int(os.getenv("PORT", "8080"))
RAILWAY_ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "development")

logger.info(f"Starting application in {RAILWAY_ENVIRONMENT} environment")
logger.info(f"Using port: {PORT}")

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting application lifespan")
    try:
        # Initialize MongoDB connection
        logger.info("Initializing MongoDB connection")
        await initialize_database()
        logger.info("MongoDB connection initialized successfully")
        
        # Create upload directories
        logger.info("Creating upload directories")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        FACES_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Upload directories created successfully")
        
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    yield
    # Shutdown
    logger.info("Shutting down application")

app = FastAPI(
    title="FaceVault API",
    description="API for FaceVault photo gallery with face detection",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB client
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://kaushik2003singh:Fg3yrUlzZPaH9R7y@complaintapp.shaxxqw.mongodb.net/facevault?retryWrites=true&w=majority&appName=ComplaintApp")
DATABASE_NAME = os.getenv("MONGODB_DB_NAME", "facevault")
client = AsyncIOMotorClient(MONGODB_URI)
db = client[DATABASE_NAME]

# Initialize face detector lazily
face_detector = None

def get_face_detector():
    global face_detector
    if face_detector is None:
        from face_detector import FaceDetector
        face_detector = FaceDetector()
    return face_detector

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Create directories if they don't exist
UPLOAD_DIR = Path("uploads")
FACES_DIR = Path("faces")

# Pydantic models
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        return {
            'type': 'str',
            'pattern': r'^[0-9a-fA-F]{24}$',
            'min_length': 24,
            'max_length': 24
        }

class MongoBaseModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)

    class Config:
        json_encoders = {
            ObjectId: str
        }
        populate_by_name = True
        arbitrary_types_allowed = True

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class User(MongoBaseModel):
    username: str
    email: EmailStr
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    storage_used: int = 0  # in bytes
    storage_limit: int = 15 * 1024 * 1024 * 1024  # 15GB in bytes

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class ImageBase(BaseModel):
    caption: Optional[str] = None

class ImageCreate(ImageBase):
    filename: str
    content_type: str
    size: int

class Image(MongoBaseModel):
    filename: str
    content_type: str
    size: int
    caption: Optional[str] = None
    url: str
    pathname: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    width: Optional[int] = None
    height: Optional[int] = None
    album_ids: List[str] = Field(default_factory=list)
    people_ids: List[str] = Field(default_factory=list)
    user_id: str
    faces: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        json_encoders = {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }
        populate_by_name = True
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields in the model

    @classmethod
    def from_mongo(cls, data: dict):
        if not data:
            return None
        # Convert ObjectId to string
        if "_id" in data:
            data["id"] = str(data.pop("_id"))
        return cls(**data)

class FaceDetection(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    person_id: Optional[str] = None

class PersonBase(BaseModel):
    name: str

class PersonCreate(PersonBase):
    pass

class Person(MongoBaseModel):
    name: str
    avatar_url: Optional[str] = None
    image_count: int = 0
    user_id: str

    class Config:
        json_encoders = {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }
        populate_by_name = True
        arbitrary_types_allowed = True

    @classmethod
    def from_mongo(cls, data: dict):
        if not data:
            return None
        # Convert ObjectId to string
        if "_id" in data:
            data["id"] = str(data.pop("_id"))
        return cls(**data)

class AlbumBase(BaseModel):
    name: str
    description: Optional[str] = None

class AlbumCreate(AlbumBase):
    pass

class Album(MongoBaseModel):
    name: str
    description: Optional[str] = None
    cover_image: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    image_count: int = 0
    user_id: str

# New Pydantic models for OTP and password reset
class OTPRequest(BaseModel):
    email: EmailStr

class OTPVerify(BaseModel):
    email: EmailStr
    otp: str

class PasswordReset(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str
    otp: Optional[str] = None

class StorageSettings(BaseModel):
    auto_compression: bool
    keep_original: bool

class NotificationSettings(BaseModel):
    email_notifications: bool
    photo_alerts: bool
    face_detection_updates: bool
    new_album_notifications: bool
    storage_alerts: bool

class AppearanceSettings(BaseModel):
    dark_mode: bool = False
    compact_view: bool = False
    grid_density: str = "medium"

# Helper functions
def get_password_hash(password: str) -> str:
    """Hash a password for storing."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a stored password against a provided password."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get the current user from the JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = await users_collection.find_one({"username": token_data.username})
    if user is None:
        raise credentials_exception
    
    # Ensure storage fields exist
    if "storage_used" not in user:
        user["storage_used"] = 0
    if "storage_limit" not in user:
        user["storage_limit"] = 15 * 1024 * 1024 * 1024  # 15GB in bytes
    
    # Convert ObjectId to string
    user["_id"] = str(user["_id"])
    return user

# Face detection functions
def detect_faces(image_path: str) -> List[Dict[str, Any]]:
    """Detect faces in an image and return their locations and encodings."""
    detector = get_face_detector()
    # Load the image
    image = face_recognition.load_image_file(image_path)
    
    # Find all face locations and encodings
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # Process each face
    faces = []
    for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
        # Extract face coordinates
        top, right, bottom, left = face_location
        
        # Create a unique ID for this face
        face_id = str(uuid.uuid4())
        
        # Extract the face image
        face_image = image[top:bottom, left:right]
        pil_image = PILImage.fromarray(face_image)
        
        # Save the face image
        face_path = os.path.join(FACES_DIR, f"{face_id}.jpg")
        pil_image.save(face_path)
        
        # Add face data to results
        faces.append({
            "id": face_id,
            "location": {
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left
            },
            "path": str(face_path),
            "encoding": face_encoding.tolist(),
            "confidence": 1.0  # Placeholder, face_recognition doesn't provide confidence scores
        })
        
        # Store face encoding in MongoDB
        face_encodings_collection.insert_one({
            "face_id": face_id,
            "encoding": face_encoding.tolist()
        })
    
    return faces

async def match_face_with_person(face_encoding) -> Optional[str]:
    """Match a face encoding with existing people."""
    # Get all face encodings from MongoDB
    face_encodings = await face_encodings_collection.find().to_list(length=None)
    
    if not face_encodings:
        return None
    
    # Convert face encoding to numpy array
    face_encoding = np.array(face_encoding)
    
    # Check against known face encodings
    for face_data in face_encodings:
        stored_encoding = np.array(face_data["encoding"])
        # Compare face encodings
        matches = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.6)
        if matches[0]:
            return face_data["person_id"]
    
    return None

# API endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Try to find user by username first
    user = await users_collection.find_one({"username": form_data.username})
    
    # If not found by username, try email
    if not user:
        user = await users_collection.find_one({"email": form_data.username})
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users", response_model=User)
async def create_user(user: UserCreate, background_tasks: BackgroundTasks):
    try:
        # Check if username or email already exists
        existing_user = await users_collection.find_one({
            "$or": [
                {"username": user.username},
                {"email": user.email}
            ]
        })
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered"
            )
    
        # Generate OTP
        otp = generate_otp()
        
        # Create new user with verification status
        hashed_password = get_password_hash(user.password)
        user_dict = {
            "username": user.username,
            "email": user.email,
            "hashed_password": hashed_password,
            "created_at": datetime.utcnow(),
            "is_verified": False,
            "storage_used": 0,  # Initialize storage used to 0
            "storage_limit": 15 * 1024 * 1024 * 1024  # 15GB in bytes
        }
        
        # Insert user first
        result = await users_collection.insert_one(user_dict)
        user_dict["_id"] = result.inserted_id
        
        # Store OTP after user is created
        await store_otp(user.email, otp)
        
        # Send verification email in background
        background_tasks.add_task(send_otp_email, user.email, otp)
        
        # Remove hashed_password from response
        user_dict.pop("hashed_password")
        
        # Convert ObjectId to string for response
        user_dict["_id"] = str(user_dict["_id"])
    
        return User(**user_dict)
    
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.post("/users/verify")
async def verify_user(request: OTPVerify):
    """Verify user's email with OTP."""
    if not await verify_otp(request.email, request.otp):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )
    
    # Update user verification status
    await users_collection.update_one(
        {"email": request.email},
        {"$set": {"is_verified": True}}
    )
    
    # Clear OTP after successful verification
    await clear_otp(request.email)
    
    return {"message": "Email verified successfully"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    # Remove hashed_password from response
    if "hashed_password" in current_user:
        current_user.pop("hashed_password")
    return current_user

@app.post("/images", response_model=Image)
async def upload_image(
    file: UploadFile = File(...),
    caption: Optional[str] = Form(None),
    detect_faces_param: str = Form("false"),  # Renamed parameter to avoid conflict
    current_user: dict = Depends(get_current_user)
):
    file_path = None
    try:
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
        logger.info(f"Caption: {caption}")
        logger.info(f"Detect faces: {detect_faces_param}")
        
        # Convert detect_faces string to boolean
        should_detect_faces = detect_faces_param.lower() == "true"
        
        # Check storage limit
        file_size = 0
        # Read file size from content
        content = await file.read()
        file_size = len(content)
        await file.seek(0)  # Reset file pointer

        logger.info(f"File size: {file_size} bytes")

        # Check if user has enough storage space
        if current_user["storage_used"] + file_size > current_user["storage_limit"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Storage limit exceeded. Please upgrade your plan or delete some images."
            )

        # Ensure upload directory exists
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    
        # Get image dimensions
        with PILImage.open(file_path) as img:
            width, height = img.size
    
        # Initialize faces list
        faces = []
        people_ids = set()

        # Only detect faces if requested
        if should_detect_faces:
            faces = detect_faces(file_path)
            
            # Match faces with existing people
            for face in faces:
                face_encoding = np.array(face["encoding"])
                # Get all face encodings from MongoDB
                face_encodings = await face_encodings_collection.find().to_list(length=None)
                
                # Track if we found a match
                found_match = False
                
                for stored_face in face_encodings:
                    stored_encoding = np.array(stored_face["encoding"])
                    # Compare face encodings with a slightly higher threshold for matching
                    matches = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.5)
                    if matches[0] and stored_face.get("person_id"):
                        people_ids.add(stored_face["person_id"])
                        face["person_id"] = stored_face["person_id"]
                        found_match = True
                        break
                
                # If no match found, create a new person
                if not found_match:
                    # Create new person
                    person_dict = {
                        "name": f"Person {len(people_ids) + 1}",
                        "user_id": str(current_user["_id"]),
                        "image_count": 1,
                        "created_at": datetime.utcnow()
                    }
                    result = await people_collection.insert_one(person_dict)
                    new_person_id = str(result.inserted_id)
                    
                    # Add to people_ids and update face
                    people_ids.add(new_person_id)
                    face["person_id"] = new_person_id
                    
                    # Store face encoding with person_id
                    await face_encodings_collection.insert_one({
                        "face_id": face["id"],
                        "encoding": face["encoding"],
                        "person_id": new_person_id,
                        "user_id": str(current_user["_id"])
                    })
    
        # Create image document
        image_dict = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file_size,
            "caption": caption or "",
            "url": f"{os.getenv('API_URL', 'http://localhost:8000')}/images/{file.filename}/content",
            "pathname": str(file_path),
            "created_at": datetime.utcnow(),
            "width": width,
            "height": height,
            "album_ids": [],
            "people_ids": list(people_ids),
            "user_id": str(current_user["_id"]),
            "faces": faces
        }
    
        logger.info(f"Created image document: {image_dict}")
        
        # Insert image without transaction first
        result = await images_collection.insert_one(image_dict)
        image_dict["_id"] = result.inserted_id
    
        # Update user's storage usage
        await users_collection.update_one(
            {"_id": ObjectId(current_user["_id"])},
            {"$inc": {"storage_used": file_size}}
        )
        
        # Update person image counts
        for person_id in people_ids:
            await people_collection.update_one(
                {"_id": ObjectId(person_id)},
                {"$inc": {"image_count": 1}}
            )
        
        # Convert to Image model
        image = Image.from_mongo(image_dict)
        logger.info(f"Returning image: {image}")
        return image
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        # Clean up the uploaded file if it exists
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up file: {str(cleanup_error)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/images", response_model=List[Image])
async def list_images(current_user: dict = Depends(get_current_user)):
    try:
        images = await images_collection.find({"user_id": str(current_user["_id"])}).to_list(length=None)
        # Convert MongoDB documents to Image models
        return [Image.from_mongo(image) for image in images]
    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/images/{image_id}", response_model=Image)
async def get_image(image_id: str, current_user: dict = Depends(get_current_user)):
    image = await images_collection.find_one({"_id": ObjectId(image_id)})
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    if image["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to access this image")
    return Image(**image)

@app.get("/images/{image_id}/content")
async def get_image_content(image_id: str):
    image = await images_collection.find_one({"_id": ObjectId(image_id)})
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image["pathname"])

@app.post("/people", response_model=Person)
async def create_person(person: PersonCreate, current_user: dict = Depends(get_current_user)):
    person_dict = person.dict()
    person_dict["user_id"] = str(current_user["_id"])
    person_dict["image_count"] = 0
    
    result = await people_collection.insert_one(person_dict)
    person_dict["_id"] = result.inserted_id
    
    return Person(**person_dict)

@app.get("/people", response_model=List[Person])
async def list_people(current_user: dict = Depends(get_current_user)):
    people = await people_collection.find({"user_id": str(current_user["_id"])}).to_list(length=None)
    # Convert ObjectId to string before creating Person model
    return [Person(**{**person, "_id": str(person["_id"])}) for person in people]

@app.get("/people/{person_id}", response_model=Person)
async def get_person(person_id: str, current_user: dict = Depends(get_current_user)):
    person = await people_collection.find_one({"_id": ObjectId(person_id)})
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    if person["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to access this person")
    # Convert ObjectId to string before creating Person model
    person["_id"] = str(person["_id"])
    return Person(**person)

@app.put("/people/{person_id}", response_model=Person)
async def update_person(
    person_id: str,
    person_update: PersonCreate,
    current_user: dict = Depends(get_current_user)
):
    person = await people_collection.find_one({"_id": ObjectId(person_id)})
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    if person["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to update this person")
    
    update_data = person_update.dict()
    await people_collection.update_one(
        {"_id": ObjectId(person_id)},
        {"$set": update_data}
    )
    
    updated_person = await people_collection.find_one({"_id": ObjectId(person_id)})
    # Convert ObjectId to string before creating Person model
    updated_person["_id"] = str(updated_person["_id"])
    return Person(**updated_person)

@app.post("/albums", response_model=Album)
async def create_album(album: AlbumCreate, current_user: dict = Depends(get_current_user)):
    album_dict = album.dict()
    album_dict["user_id"] = str(current_user["_id"])
    album_dict["image_count"] = 0
    album_dict["created_at"] = datetime.utcnow()
    
    result = await albums_collection.insert_one(album_dict)
    album_dict["_id"] = str(result.inserted_id)  # Convert ObjectId to string
    
    return Album(**album_dict)

@app.get("/albums", response_model=List[Album])
async def list_albums(current_user: dict = Depends(get_current_user)):
    albums = await albums_collection.find({"user_id": str(current_user["_id"])}).to_list(length=None)
    # Convert ObjectId to string before creating Album model
    return [Album(**{**album, "_id": str(album["_id"])}) for album in albums]

@app.get("/albums/{album_id}", response_model=Album)
async def get_album(album_id: str, current_user: dict = Depends(get_current_user)):
    album = await albums_collection.find_one({"_id": ObjectId(album_id)})
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")
    if album["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to access this album")
    
    # If no cover image is set, try to get the latest image
    if not album.get("cover_image"):
        latest_image = await images_collection.find_one(
            {"album_ids": album_id},
            sort=[("created_at", -1)]
        )
        if latest_image:
            album["cover_image"] = f"/images/{latest_image['_id']}/content"
            # Update the album with the cover image
            await albums_collection.update_one(
                {"_id": ObjectId(album_id)},
                {"$set": {"cover_image": album["cover_image"]}}
            )
    
    # Convert ObjectId to string before creating Album model
    album["_id"] = str(album["_id"])
    return Album(**album)

@app.put("/albums/{album_id}")
async def update_album(
    album_id: str,
    album_update: AlbumCreate,
    current_user: dict = Depends(get_current_user)
):
    # Verify album exists and belongs to user
    album = await albums_collection.find_one({"_id": ObjectId(album_id)})
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")
    if album["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to update this album")
    
    try:
        # Update album
        update_data = {
            "name": album_update.name,
            "description": album_update.description,
            "updated_at": datetime.utcnow()
        }
        
        # Preserve existing cover_image if it exists
        if "cover_image" in album:
            update_data["cover_image"] = album["cover_image"]
        
        await albums_collection.update_one(
            {"_id": ObjectId(album_id)},
            {"$set": update_data}
        )
        
        # Get updated album
        updated_album = await albums_collection.find_one({"_id": ObjectId(album_id)})
        if not updated_album:
            raise HTTPException(status_code=404, detail="Album not found after update")
        
        # Convert ObjectId to string
        updated_album["_id"] = str(updated_album["_id"])
        
        return updated_album
    except Exception as e:
        logger.error(f"Error updating album: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update album"
        )

@app.post("/albums/{album_id}/images")
async def add_images_to_album(
    album_id: str,
    image_ids: List[str] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    album = await albums_collection.find_one({"_id": ObjectId(album_id)})
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")
    if album["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to modify this album")
    
    # Update images with album_id
    for image_id in image_ids:
        await images_collection.update_one(
            {"_id": ObjectId(image_id)},
            {"$addToSet": {"album_ids": album_id}}
        )
    
    # Update album image count
    image_count = await images_collection.count_documents({"album_ids": album_id})
    
    # Get the latest image to use as cover
    latest_image = await images_collection.find_one(
        {"album_ids": album_id},
        sort=[("created_at", -1)]
    )
    
    # Update album with new count and cover image
    update_data = {"image_count": image_count}
    if latest_image:
        update_data["cover_image"] = f"/images/{latest_image['_id']}/content"
    
    await albums_collection.update_one(
        {"_id": ObjectId(album_id)},
        {"$set": update_data}
    )
    
    return {"message": "Images added to album successfully"}

@app.post("/people/merge")
async def merge_people(
    person_ids: List[str] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    if len(person_ids) < 2:
        raise HTTPException(status_code=400, detail="At least two people must be selected for merging")
    
    # Get all people to merge
    people = await people_collection.find({"_id": {"$in": [ObjectId(pid) for pid in person_ids]}}).to_list(length=None)
    if len(people) != len(person_ids):
        raise HTTPException(status_code=404, detail="One or more people not found")
    
    # Verify ownership
    for person in people:
        if person["user_id"] != str(current_user["_id"]):
            raise HTTPException(status_code=403, detail="Not authorized to merge these people")
    
    # Keep the first person and merge others into it
    target_person = people[0]
    other_people = people[1:]
    
    # Update all images and face encodings to point to the target person
    for person in other_people:
        await images_collection.update_many(
            {"people_ids": str(person["_id"])},
            {"$set": {"people_ids.$": str(target_person["_id"])}}
        )
        
        await face_encodings_collection.update_many(
            {"person_id": str(person["_id"])},
            {"$set": {"person_id": str(target_person["_id"])}}
        )
    
    # Delete other people
    await people_collection.delete_many({"_id": {"$in": [ObjectId(p["_id"]) for p in other_people]}})
    
    # Update target person's image count
    image_count = await images_collection.count_documents({"people_ids": str(target_person["_id"])})
    await people_collection.update_one(
        {"_id": target_person["_id"]},
        {"$set": {"image_count": image_count}}
    )
    
    return {"message": "People merged successfully"}

@app.get("/faces/{face_id}/content")
async def get_face_content(face_id: str):
    face_encoding = await face_encodings_collection.find_one({"face_id": face_id})
    if not face_encoding:
        raise HTTPException(status_code=404, detail="Face not found")
    
    face_path = os.path.join(FACES_DIR, f"{face_id}.jpg")
    if not os.path.exists(face_path):
        raise HTTPException(status_code=404, detail="Face image not found")
    
    return FileResponse(face_path)

# New endpoints for OTP and password reset
@app.post("/otp/request")
async def request_otp(request: OTPRequest, background_tasks: BackgroundTasks):
    """Request an OTP for email verification or password reset."""
    user = await users_collection.find_one({"email": request.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Generate and store OTP
    otp = generate_otp()
    await store_otp(request.email, otp)
    
    # Send OTP email in background
    background_tasks.add_task(send_otp_email, request.email, otp)
    
    return {"message": "OTP sent successfully"}

@app.post("/otp/verify")
async def verify_otp_endpoint(request: OTPVerify):
    """Verify OTP for email verification or password reset."""
    if not verify_otp(request.email, request.otp):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )
    
    # Clear OTP after successful verification
    clear_otp(request.email)
    
    return {"message": "OTP verified successfully"}

@app.post("/password/reset/request")
async def request_password_reset(request: PasswordReset, background_tasks: BackgroundTasks):
    """Request a password reset."""
    user = await users_collection.find_one({"email": request.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Generate and store OTP
    otp = generate_otp()
    await store_otp(request.email, otp)
    
    # Create reset URL with OTP
    reset_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/reset-password?email={request.email}&otp={otp}"
    
    # Send reset password email in background
    background_tasks.add_task(send_reset_password_email, request.email, reset_url)
    
    return {"message": "Password reset instructions sent to your email"}

@app.post("/password/reset/confirm")
async def confirm_password_reset(request: PasswordResetConfirm):
    """Confirm password reset with OTP."""
    if not await verify_otp(request.email, request.otp):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )
    
    # Update password
    hashed_password = get_password_hash(request.new_password)
    await users_collection.update_one(
        {"email": request.email},
        {"$set": {"hashed_password": hashed_password}}
    )
    
    # Clear OTP
    await clear_otp(request.email)
    
    return {"message": "Password reset successful"}

@app.delete("/images/{image_id}")
async def delete_image(image_id: str, current_user: dict = Depends(get_current_user)):
    # Get image details
    image = await images_collection.find_one({"_id": ObjectId(image_id)})
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    if image["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to delete this image")
    
    try:
        # Delete image file
        try:
            os.remove(image["pathname"])
        except OSError:
            pass  # Ignore if file doesn't exist
        
        # Delete image from database
        await images_collection.delete_one({"_id": ObjectId(image_id)})
        
        # Update user's storage usage
        await users_collection.update_one(
            {"_id": ObjectId(current_user["_id"])},
            {"$inc": {"storage_used": -image["size"]}}
        )
        
        # Delete associated face images and encodings
        for face in image.get("faces", []):
            try:
                os.remove(face["path"])
            except OSError:
                pass  # Ignore if file doesn't exist
        
        # Update person image counts
        for person_id in image.get("people_ids", []):
            await people_collection.update_one(
                {"_id": ObjectId(person_id)},
                {"$inc": {"image_count": -1}}
            )
        
        return {"message": "Image deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/users/storage")
async def get_storage_info(current_user: dict = Depends(get_current_user)):
    """Get user's storage usage information."""
    user = await users_collection.find_one({"_id": ObjectId(current_user["_id"])})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get storage values with defaults
    storage_used = user.get("storage_used", 0)
    storage_limit = user.get("storage_limit", 15 * 1024 * 1024 * 1024)  # Default 15GB
    
    # Calculate percentage
    used_percentage = (storage_used / storage_limit) * 100 if storage_limit > 0 else 0
    
    return {
        "used": storage_used,
        "limit": storage_limit,
        "used_percentage": used_percentage
    }

@app.put("/users/profile")
async def update_user_profile(
    profile_data: dict,
    current_user: User = Depends(get_current_user)
):
    try:
        # Update only allowed fields
        update_data = {}
        if "full_name" in profile_data:
            update_data["full_name"] = profile_data["full_name"]
        if "bio" in profile_data:
            update_data["bio"] = profile_data["bio"]

        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields to update")

        # Update user in database
        result = await db.users.update_one(
            {"_id": current_user["_id"]},
            {"$set": update_data}
        )

        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="Failed to update profile")

        # Get updated user data
        updated_user = await db.users.find_one({"_id": current_user["_id"]})
        return {
            "message": "Profile updated successfully",
            "user": {
                "id": str(updated_user["_id"]),
                "username": updated_user["username"],
                "email": updated_user["email"],
                "full_name": updated_user.get("full_name", ""),
                "bio": updated_user.get("bio", ""),
                "profile_image": updated_user.get("profile_image", "")
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users/password/update")
async def update_password(
    password_data: PasswordUpdate,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Verify current password
        if not verify_password(password_data.current_password, current_user["hashed_password"]):
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        # If OTP is not provided, send OTP
        if not password_data.otp:
            # Generate and store OTP
            otp = generate_otp()
            await store_otp(current_user["email"], otp)
            
            # Send OTP email
            await send_otp_email(current_user["email"], otp)
            
            return {"message": "OTP sent successfully"}

        # Verify OTP
        if not await verify_otp(current_user["email"], password_data.otp):
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")

        # Update password
        hashed_password = get_password_hash(password_data.new_password)
        await users_collection.update_one(
            {"_id": ObjectId(current_user["_id"])},
            {"$set": {"hashed_password": hashed_password}}
        )

        # Clear OTP
        await clear_otp(current_user["email"])

        return {"message": "Password updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/storage/stats")
async def get_storage_stats(current_user: dict = Depends(get_current_user)):
    """Get detailed storage statistics for the user."""
    try:
        # Get total storage used
        total_storage = current_user.get("storage_used", 0)
        
        # Get image count and sizes
        images = await images_collection.find({"user_id": str(current_user["_id"])}).to_list(length=None)
        image_count = len(images)
        
        # Get album count
        album_count = await albums_collection.count_documents({"user_id": str(current_user["_id"])})
        
        # Get people count
        people_count = await people_collection.count_documents({"user_id": str(current_user["_id"])})
        
        # Calculate storage by type
        storage_by_type = {
            "images": sum(img.get("size", 0) for img in images),
            "faces": sum(len(img.get("faces", [])) * 1024 * 10 for img in images),  # Approximate face storage
            "metadata": len(images) * 1024  # Approximate metadata storage
        }
        
        return {
            "total_storage": total_storage,
            "storage_limit": current_user.get("storage_limit", 15 * 1024 * 1024 * 1024),  # 15GB default
            "image_count": image_count,
            "album_count": album_count,
            "people_count": people_count,
            "storage_by_type": storage_by_type,
            "settings": {
                "auto_compression": current_user.get("auto_compression", True),
                "keep_original": current_user.get("keep_original", False)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users/storage/clear")
async def clear_storage(
    clear_type: str = Body(..., embed=True),  # "all", "faces", "cache"
    current_user: dict = Depends(get_current_user)
):
    """Clear storage based on type."""
    try:
        if clear_type not in ["all", "faces", "cache"]:
            raise HTTPException(status_code=400, detail="Invalid clear type")

        # Start a session for transaction
        async with await client.start_session() as session:
            async with session.start_transaction():
                if clear_type == "all":
                    # Delete all images and their files
                    images = await images_collection.find({"user_id": str(current_user["_id"])}).to_list(length=None)
                    for image in images:
                        try:
                            os.remove(image["pathname"])
                        except OSError:
                            pass
                    
                    # Delete all face images
                    for image in images:
                        for face in image.get("faces", []):
                            try:
                                os.remove(face["path"])
                            except OSError:
                                pass
                    
                    # Clear all collections
                    await images_collection.delete_many({"user_id": str(current_user["_id"])}, session=session)
                    await albums_collection.delete_many({"user_id": str(current_user["_id"])}, session=session)
                    await people_collection.delete_many({"user_id": str(current_user["_id"])}, session=session)
                    await face_encodings_collection.delete_many({"user_id": str(current_user["_id"])}, session=session)
                    
                    # Reset storage usage
                    await users_collection.update_one(
                        {"_id": ObjectId(current_user["_id"])},
                        {"$set": {"storage_used": 0}},
                        session=session
                    )
                
                elif clear_type == "faces":
                    # Delete only face images and encodings
                    images = await images_collection.find({"user_id": str(current_user["_id"])}).to_list(length=None)
                    for image in images:
                        for face in image.get("faces", []):
                            try:
                                os.remove(face["path"])
                            except OSError:
                                pass
                    
                    # Clear face encodings
                    await face_encodings_collection.delete_many({"user_id": str(current_user["_id"])}, session=session)
                    
                    # Update images to remove face data
                    await images_collection.update_many(
                        {"user_id": str(current_user["_id"])},
                        {"$set": {"faces": []}},
                        session=session
                    )
                
                elif clear_type == "cache":
                    # Clear temporary files and cache
                    cache_dir = Path("cache")
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                    cache_dir.mkdir(exist_ok=True)

        return {"message": f"Storage cleared successfully: {clear_type}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/users/storage/settings")
async def update_storage_settings(
    settings: StorageSettings,
    current_user: dict = Depends(get_current_user)
):
    """Update storage settings."""
    try:
        await users_collection.update_one(
            {"_id": ObjectId(current_user["_id"])},
            {"$set": {
                "auto_compression": settings.auto_compression,
                "keep_original": settings.keep_original
            }}
        )
        return {"message": "Storage settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_notification_settings(user_id: str) -> NotificationSettings:
    """Get user's notification settings from database."""
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Return default settings if none exist
    return NotificationSettings(
        email_notifications=user.get("email_notifications", True),
        photo_alerts=user.get("photo_alerts", True),
        face_detection_updates=user.get("face_detection_updates", True),
        new_album_notifications=user.get("new_album_notifications", True),
        storage_alerts=user.get("storage_alerts", True)
    )

@app.get("/users/notifications/settings")
async def get_user_notification_settings(current_user: dict = Depends(get_current_user)):
    """Get user's notification settings."""
    try:
        settings = await get_notification_settings(str(current_user["_id"]))
        return settings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/users/notifications/settings")
async def update_notification_settings(
    settings: NotificationSettings,
    current_user: dict = Depends(get_current_user)
):
    """Update user's notification settings."""
    try:
        # Update settings in database
        await users_collection.update_one(
            {"_id": ObjectId(current_user["_id"])},
            {"$set": {
                "email_notifications": settings.email_notifications,
                "photo_alerts": settings.photo_alerts,
                "face_detection_updates": settings.face_detection_updates,
                "new_album_notifications": settings.new_album_notifications,
                "storage_alerts": settings.storage_alerts
            }}
        )
        
        return {"message": "Notification settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_appearance_settings(user_id: str) -> AppearanceSettings:
    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user or "appearance_settings" not in user:
        return AppearanceSettings()
    return AppearanceSettings(**user["appearance_settings"])

@app.get("/users/appearance/settings", response_model=AppearanceSettings)
async def get_user_appearance_settings(current_user: dict = Depends(get_current_user)):
    try:
        settings = await get_appearance_settings(str(current_user["_id"]))
        return settings
    except Exception as e:
        logger.error(f"Error getting appearance settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get appearance settings")

@app.put("/users/appearance/settings", response_model=AppearanceSettings)
async def update_user_appearance_settings(
    settings: AppearanceSettings,
    current_user: dict = Depends(get_current_user)
):
    try:
        result = await db.users.update_one(
            {"_id": ObjectId(current_user["_id"])},
            {"$set": {"appearance_settings": settings.dict()}}
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        return settings
    except Exception as e:
        logger.error(f"Error updating appearance settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update appearance settings")

# Add new endpoint for face detection
@app.post("/images/{image_id}/detect-faces")
async def detect_faces_in_image(
    image_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Get image details
        image = await images_collection.find_one({"_id": ObjectId(image_id)})
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        if image["user_id"] != str(current_user["_id"]):
            raise HTTPException(status_code=403, detail="Not authorized to detect faces in this image")

        # Get existing people_ids from the image
        existing_people_ids = set(image.get("people_ids", []))

        # Detect faces in the image
        faces = await detect_faces(image["pathname"])
        if not faces:
            return {
                "message": "No faces detected",
                "faces": [],
                "skipped": True
            }

        # Set to store unique person IDs
        people_ids = set()

        # Match faces with existing people
        for face in faces:
            face_encoding = np.array(face["encoding"])
            # Get all face encodings from MongoDB
            face_encodings = await face_encodings_collection.find().to_list(length=None)
            
            # Track if we found a match
            found_match = False
            
            for stored_face in face_encodings:
                stored_encoding = np.array(stored_face["encoding"])
                # Compare face encodings
                matches = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.6)
                if matches[0] and stored_face.get("person_id"):
                    people_ids.add(stored_face["person_id"])
                    face["person_id"] = stored_face["person_id"]
                    found_match = True
                    break
            
            # If no match found, create a new person
            if not found_match:
                # Create new person
                person_dict = {
                    "name": f"Person {len(people_ids) + 1}",
                    "user_id": str(current_user["_id"]),
                    "image_count": 1,
                    "created_at": datetime.utcnow()
                }
                result = await people_collection.insert_one(person_dict)
                new_person_id = str(result.inserted_id)
                
                # Add to people_ids and update face
                people_ids.add(new_person_id)
                face["person_id"] = new_person_id
                
                # Store face encoding with person_id
                await face_encodings_collection.insert_one({
                    "face_id": face["id"],
                    "encoding": face["encoding"],
                    "person_id": new_person_id,
                    "user_id": str(current_user["_id"])
                })

        # Update image with detected faces
        await images_collection.update_one(
            {"_id": ObjectId(image_id)},
            {
                "$set": {
                    "faces": faces,
                    "people_ids": list(people_ids)
                }
            }
        )

        # Update image counts for all people
        # First, get all people that need their counts updated
        all_affected_people = existing_people_ids.union(people_ids)
        
        # For each affected person, count their actual images and update their count
        for person_id in all_affected_people:
            # Count actual number of images this person appears in
            image_count = await images_collection.count_documents({
                "people_ids": person_id,
                "user_id": str(current_user["_id"])
            })
            # Update the person's image count
            await people_collection.update_one(
                {"_id": ObjectId(person_id)},
                {"$set": {"image_count": image_count}}
            )

        return {
            "message": "Face detection completed",
            "faces": faces,
            "skipped": False
        }
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Add endpoint to update person name
@app.put("/people/{person_id}/name")
async def update_person_name(
    person_id: str,
    name: str = Body(..., embed=True),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Check if person exists and belongs to user
        person = await people_collection.find_one({"_id": ObjectId(person_id)})
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        if person["user_id"] != str(current_user["_id"]):
            raise HTTPException(status_code=403, detail="Not authorized to update this person")

        # Update person name
        await people_collection.update_one(
            {"_id": ObjectId(person_id)},
            {"$set": {"name": name}}
        )

        return {"message": "Person name updated successfully"}
    except Exception as e:
        logger.error(f"Error updating person name: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/albums/{album_id}/images", response_model=List[Image])
async def get_album_images(album_id: str, current_user: dict = Depends(get_current_user)):
    # Verify album exists and belongs to user
    album = await albums_collection.find_one({"_id": ObjectId(album_id)})
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")
    if album["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to access this album")
    
    # Get all images in the album
    images = await images_collection.find({"album_ids": album_id}).to_list(length=None)
    # Convert ObjectId to string for each image
    for image in images:
        image["_id"] = str(image["_id"])
    return [Image.from_mongo(image) for image in images]

@app.delete("/albums/{album_id}")
async def delete_album(album_id: str, current_user: dict = Depends(get_current_user)):
    # Verify album exists and belongs to user
    album = await albums_collection.find_one({"_id": ObjectId(album_id)})
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")
    if album["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to delete this album")
    
    try:
        # Remove album_id from all images
        await images_collection.update_many(
            {"album_ids": album_id},
            {"$pull": {"album_ids": album_id}}
        )
        
        # Delete the album
        await albums_collection.delete_one({"_id": ObjectId(album_id)})
        
        return {"message": "Album deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting album: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete album"
        )

@app.get("/people/{person_id}/images", response_model=List[Image])
async def get_person_images(person_id: str, current_user: dict = Depends(get_current_user)):
    # Verify person exists and belongs to user
    person = await people_collection.find_one({"_id": ObjectId(person_id)})
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    if person["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to access this person's images")
    
    # Get all images that have this person's ID in their people_ids
    images = await images_collection.find({"people_ids": person_id}).to_list(length=None)
    
    # Convert ObjectId to string for each image
    for image in images:
        image["_id"] = str(image["_id"])
    
    return [Image.from_mongo(image) for image in images]

# Add root endpoint for welcome message
@app.get("/")
async def root():
    return {
        "message": "Welcome to FaceVault API!",
        "status": "Server is running successfully",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    }

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
