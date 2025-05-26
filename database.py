from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB configuration
logger.info("Attempting to load MongoDB URI from environment variables")
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    logger.error("MONGODB_URI environment variable is not set")
    raise ValueError("MONGODB_URI environment variable is not set. Please set it to your MongoDB connection string.")
else:
    logger.info("MONGODB_URI found in environment variables")

DATABASE_NAME = os.getenv("MONGODB_DB_NAME", "facevault")
logger.info(f"Using database name: {DATABASE_NAME}")

# Async client for FastAPI
logger.info("Initializing MongoDB async client")
async_client = AsyncIOMotorClient(MONGODB_URI)
db = async_client[DATABASE_NAME]

# Sync client for background tasks
sync_client = MongoClient(MONGODB_URI)
sync_db = sync_client[DATABASE_NAME]

# Collections
users_collection = db.users
images_collection = db.images
people_collection = db.people
albums_collection = db.albums
face_encodings_collection = db.face_encodings
subscriptions_collection = db.subscriptions
cold_storage_collection = db.cold_storage

async def initialize_database():
    try:
        # Users collection indexes
        await users_collection.create_index("username", unique=True)
        await users_collection.create_index("email", unique=True)
        
        # Images collection indexes
        await images_collection.create_index("user_id")
        await images_collection.create_index("created_at")
        await images_collection.create_index("is_in_cold_storage")
        
        # People collection indexes
        await people_collection.create_index("user_id")
        await people_collection.create_index("name")
        
        # Albums collection indexes
        await albums_collection.create_index("user_id")
        await albums_collection.create_index("created_at")
        
        # Face encodings collection indexes
        await face_encodings_collection.create_index("person_id")
        await face_encodings_collection.create_index("image_id")
        
        # Subscriptions collection indexes
        await subscriptions_collection.create_index("user_id", unique=True)
        await subscriptions_collection.create_index("status")
        await subscriptions_collection.create_index("end_date")
        await subscriptions_collection.create_index([("user_id", 1), ("status", 1)])
        
        # Cold storage collection indexes
        await cold_storage_collection.create_index("user_id")
        await cold_storage_collection.create_index("image_id", unique=True)
        await cold_storage_collection.create_index("moved_at")
        
        logger.info("All database indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating indexes: {str(e)}")
        raise 