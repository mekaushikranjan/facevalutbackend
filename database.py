from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging
import asyncio
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    logger.error("MONGODB_URI environment variable is not set!")
    # For debugging, print all environment variables (excluding sensitive ones)
    logger.debug("Available environment variables:")
    for key, value in os.environ.items():
        if 'SECRET' not in key and 'KEY' not in key and 'PASSWORD' not in key:
            logger.debug(f"{key}: {value}")
    raise ValueError("MONGODB_URI environment variable is not set. Please set it to your MongoDB connection string.")

# Log the MongoDB URI (without password) for debugging
uri_parts = MONGODB_URI.split('@')
if len(uri_parts) > 1:
    logger.info(f"Using MongoDB URI: {uri_parts[0]}@***")
else:
    logger.info(f"Using MongoDB URI: {MONGODB_URI}")

DATABASE_NAME = os.getenv("MONGODB_DB_NAME", "facevault")

# Connection retry settings
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

async def connect_with_retry():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            logger.info(f"Attempting to connect to MongoDB (attempt {retries + 1}/{MAX_RETRIES})")
            # Add connection timeout and server selection timeout
            client = AsyncIOMotorClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            # Test the connection
            await client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            return client
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            retries += 1
            if retries == MAX_RETRIES:
                logger.error(f"Failed to connect to MongoDB after {MAX_RETRIES} attempts: {str(e)}")
                raise
            logger.warning(f"Connection attempt {retries} failed: {str(e)}. Retrying in {RETRY_DELAY} seconds...")
            await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {str(e)}")
            raise

# Initialize clients
async_client = None
sync_client = None
db = None
sync_db = None

# Collections
users_collection = None
images_collection = None
people_collection = None
albums_collection = None
face_encodings_collection = None
subscriptions_collection = None
cold_storage_collection = None

async def initialize_database():
    global async_client, sync_client, db, sync_db
    global users_collection, images_collection, people_collection, albums_collection
    global face_encodings_collection, subscriptions_collection, cold_storage_collection

    try:
        # Initialize async client
        async_client = await connect_with_retry()
        db = async_client[DATABASE_NAME]

        # Initialize sync client with same settings
        sync_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000
        )
        sync_db = sync_client[DATABASE_NAME]

        # Initialize collections
        users_collection = db.users
        images_collection = db.images
        people_collection = db.people
        albums_collection = db.albums
        face_encodings_collection = db.face_encodings
        subscriptions_collection = db.subscriptions
        cold_storage_collection = db.cold_storage

        # Create indexes
        await create_indexes()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

# Create indexes
async def create_indexes():
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