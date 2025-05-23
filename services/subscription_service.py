from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from motor.motor_asyncio import AsyncIOMotorClient
from ..models.subscription import Subscription, ColdStorage
from ..config import settings
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class SubscriptionService:
    def __init__(self, db: AsyncIOMotorClient):
        self.db = db
        self.subscriptions = db.subscriptions
        self.cold_storage = db.cold_storage
        self.images = db.images
        self.users = db.users

    async def create_subscription(self, user_id: str, plan: str, duration_months: int) -> Dict[str, Any]:
        """Create a new subscription for a user."""
        try:
            # Validate plan type
            valid_plans = ["free", "pro", "enterprise"]
            if plan not in valid_plans:
                raise HTTPException(status_code=400, detail=f"Invalid plan type. Must be one of: {', '.join(valid_plans)}")

            # Validate duration for paid plans
            if plan != "free" and duration_months not in [1, 6, 12]:
                raise HTTPException(status_code=400, detail="Invalid duration. Must be 1, 6, or 12 months")

            # Check if user already has an active subscription
            existing_sub = await self.subscriptions.find_one({"user_id": user_id, "status": "active"})
            if existing_sub and plan != "free":
                raise HTTPException(status_code=400, detail="User already has an active subscription")

            # Calculate dates
            start_date = datetime.utcnow()
            end_date = start_date + timedelta(days=30 * duration_months) if plan != "free" else None

            # Set storage limits based on plan
            storage_limits = {
                "free": 10 * 1024 * 1024 * 1024,  # 10GB
                "pro": 100 * 1024 * 1024 * 1024,  # 100GB
                "enterprise": 1024 * 1024 * 1024 * 1024  # 1TB
            }

            subscription = {
                "user_id": user_id,
                "plan": plan,
                "status": "active",
                "start_date": start_date,
                "end_date": end_date,
                "storage_limit": storage_limits[plan],
                "storage_used": 0,
                "is_in_cold_storage": False,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            # For free plan, update existing subscription or create new one
            if plan == "free":
                await self.subscriptions.update_one(
                    {"user_id": user_id},
                    {"$set": subscription},
                    upsert=True
                )
            else:
                # For paid plans, create new subscription
                await self.subscriptions.insert_one(subscription)

            return subscription

        except Exception as e:
            logger.error(f"Error creating subscription: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create subscription")

    async def get_subscription(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's current subscription."""
        try:
            subscription = await self.subscriptions.find_one({"user_id": user_id, "status": "active"})
            if not subscription:
                # Create free subscription if none exists
                return await self.create_subscription(user_id, "free", 0)
            return subscription
        except Exception as e:
            logger.error(f"Error getting subscription: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get subscription")

    async def update_storage_usage(self, user_id: str, bytes_to_add: int) -> bool:
        """Update user's storage usage and check if they've exceeded their limit."""
        try:
            subscription = await self.get_subscription(user_id)
            if not subscription:
                return False

            new_usage = subscription["storage_used"] + bytes_to_add
            if new_usage > subscription["storage_limit"]:
                logger.warning(f"User {user_id} exceeded storage limit")
                return False

            await self.subscriptions.update_one(
                {"user_id": user_id},
                {"$set": {"storage_used": new_usage, "updated_at": datetime.utcnow()}}
            )
            return True
        except Exception as e:
            logger.error(f"Error updating storage usage: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to update storage usage")

    async def check_subscription_status(self, user_id: str) -> Dict[str, Any]:
        """Check if a user's subscription is active and handle expired subscriptions."""
        try:
            subscription = await self.get_subscription(user_id)
            if not subscription:
                return {
                    "plan": "free",
                    "status": "active",
                    "storage_limit": 10 * 1024 * 1024 * 1024,  # 10GB
                    "storage_used": 0,
                    "is_active": True,
                    "days_remaining": None
                }

            # For free plan, always return active
            if subscription["plan"] == "free":
                return {
                    "plan": "free",
                    "status": "active",
                    "storage_limit": subscription["storage_limit"],
                    "storage_used": subscription["storage_used"],
                    "is_active": True,
                    "days_remaining": None
                }

            # Check if subscription is expired
            if subscription["end_date"] and subscription["end_date"] < datetime.utcnow():
                # Move to cold storage if not already
                if not subscription["is_in_cold_storage"]:
                    await self.move_to_cold_storage(user_id)
                
                # Update subscription status
                await self.subscriptions.update_one(
                    {"_id": subscription["_id"]},
                    {"$set": {"status": "expired", "updated_at": datetime.utcnow()}}
                )
                
                return {
                    "plan": subscription["plan"],
                    "status": "expired",
                    "storage_limit": subscription["storage_limit"],
                    "storage_used": subscription["storage_used"],
                    "is_active": False,
                    "days_remaining": 0
                }

            # Calculate days remaining
            days_remaining = (subscription["end_date"] - datetime.utcnow()).days

            return {
                "plan": subscription["plan"],
                "status": subscription["status"],
                "storage_limit": subscription["storage_limit"],
                "storage_used": subscription["storage_used"],
                "is_active": True,
                "days_remaining": days_remaining
            }

        except Exception as e:
            logger.error(f"Error checking subscription status: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to check subscription status")

    async def move_to_cold_storage(self, user_id: str) -> None:
        """Move a user's images to cold storage."""
        try:
            # Get all user's images
            cursor = self.images.find({"user_id": user_id})
            images = await cursor.to_list(length=None)

            for image in images:
                # Create cold storage record
                cold_storage_record = {
                    "user_id": user_id,
                    "image_id": image["_id"],
                    "original_path": image["path"],
                    "cold_storage_path": f"cold_storage/{user_id}/{image['_id']}",
                    "moved_at": datetime.utcnow()
                }
                await self.cold_storage.insert_one(cold_storage_record)

                # Update image record
                await self.images.update_one(
                    {"_id": image["_id"]},
                    {"$set": {"is_in_cold_storage": True}}
                )

            # Update subscription
            await self.subscriptions.update_one(
                {"user_id": user_id, "status": "active"},
                {"$set": {"is_in_cold_storage": True}}
            )

        except Exception as e:
            logger.error(f"Error moving to cold storage: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to move to cold storage")

    async def restore_from_cold_storage(self, user_id: str) -> None:
        """Restore a user's images from cold storage."""
        try:
            # Get all cold storage records for user
            cursor = self.cold_storage.find({"user_id": user_id})
            cold_storage_records = await cursor.to_list(length=None)

            for record in cold_storage_records:
                # Update image record
                await self.images.update_one(
                    {"_id": record["image_id"]},
                    {"$set": {"is_in_cold_storage": False}}
                )

                # Delete cold storage record
                await self.cold_storage.delete_one({"_id": record["_id"]})

            # Update subscription
            await self.subscriptions.update_one(
                {"user_id": user_id, "status": "active"},
                {"$set": {"is_in_cold_storage": False}}
            )

        except Exception as e:
            logger.error(f"Error restoring from cold storage: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to restore from cold storage")

    async def get_subscription_history(self, user_id: str) -> list:
        """Get a user's subscription history."""
        try:
            cursor = self.subscriptions.find(
                {"user_id": user_id}
            ).sort("created_at", -1)
            
            history = await cursor.to_list(length=None)
            return history

        except Exception as e:
            logger.error(f"Error getting subscription history: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to get subscription history")

    async def cancel_subscription(self, user_id: str) -> bool:
        """Cancel a user's subscription."""
        try:
            subscription = await self.get_subscription(user_id)
            if not subscription or subscription["plan"] == "free":
                raise HTTPException(status_code=400, detail="No active subscription to cancel")

            # Update subscription status
            result = await self.subscriptions.update_one(
                {"user_id": user_id, "status": "active"},
                {
                    "$set": {
                        "status": "cancelled",
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            return result.modified_count > 0

        except Exception as e:
            logger.error(f"Error cancelling subscription: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to cancel subscription") 