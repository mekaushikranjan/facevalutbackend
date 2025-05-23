import pyotp
import time
import random
from datetime import datetime, timedelta
from typing import Optional
from database import users_collection

# OTP configuration
OTP_EXPIRY_MINUTES = 10
OTP_LENGTH = 6

def generate_otp() -> str:
    """Generate a 6-digit numeric OTP."""
    return ''.join([str(random.randint(0, 9)) for _ in range(OTP_LENGTH)])

async def verify_otp(email: str, otp: str) -> bool:
    """
    Verify the OTP for a given email.
    
    Args:
        email: User's email address
        otp: OTP to verify
    
    Returns:
        bool: True if OTP is valid, False otherwise
    """
    user = await users_collection.find_one({"email": email})
    if not user or "otp" not in user or "otp_expiry" not in user:
        return False
    
    # Check if OTP has expired
    if datetime.utcnow() > user["otp_expiry"]:
        return False
    
    # Verify OTP
    return user["otp"] == otp

async def store_otp(email: str, otp: str) -> None:
    """
    Store OTP in the database with expiry time.
    
    Args:
        email: User's email address
        otp: OTP to store
    """
    expiry = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
    await users_collection.update_one(
        {"email": email},
        {
            "$set": {
                "otp": otp,
                "otp_expiry": expiry
            }
        },
        upsert=True
    )

async def clear_otp(email: str) -> None:
    """
    Clear OTP from the database.
    
    Args:
        email: User's email address
    """
    await users_collection.update_one(
        {"email": email},
        {
            "$unset": {
                "otp": "",
                "otp_expiry": ""
            }
        }
    ) 