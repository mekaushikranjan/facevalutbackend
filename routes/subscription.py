from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
from ..services.subscription_service import SubscriptionService
from ..database import get_database
from ..auth import get_current_user
from ..models.user import User
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])

class SubscriptionCreate(BaseModel):
    plan: str
    duration_months: int

class SubscriptionResponse(BaseModel):
    id: str
    user_id: str
    plan: str
    status: str
    start_date: datetime
    end_date: datetime
    storage_limit: int
    storage_used: int
    is_in_cold_storage: bool

@router.post("/", response_model=SubscriptionResponse)
async def create_subscription(
    subscription: SubscriptionCreate,
    current_user: User = Depends(get_current_user),
    db = Depends(get_database)
):
    """Create a new subscription for the current user."""
    subscription_service = SubscriptionService(db)
    return await subscription_service.create_subscription(
        current_user.id,
        subscription.plan,
        subscription.duration_months
    )

@router.get("/status", response_model=Dict[str, Any])
async def get_subscription_status(
    current_user: User = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get the current subscription status for the user."""
    subscription_service = SubscriptionService(db)
    return await subscription_service.check_subscription_status(current_user.id)

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_subscription_history(
    current_user: User = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get subscription history for the user."""
    subscription_service = SubscriptionService(db)
    return await subscription_service.get_subscription_history(current_user.id)

@router.post("/cancel")
async def cancel_subscription(
    current_user: User = Depends(get_current_user),
    db = Depends(get_database)
):
    """Cancel the current subscription."""
    subscription_service = SubscriptionService(db)
    success = await subscription_service.cancel_subscription(current_user.id)
    if success:
        return {"message": "Subscription cancelled successfully"}
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Failed to cancel subscription"
    )

@router.get("/storage", response_model=Dict[str, Any])
async def get_storage_info(
    current_user: User = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get storage usage information for the user."""
    subscription_service = SubscriptionService(db)
    status = await subscription_service.check_subscription_status(current_user.id)
    return {
        "storage_limit": status["storage_limit"],
        "storage_used": status["storage_used"],
        "storage_remaining": status["storage_limit"] - status["storage_used"],
        "usage_percentage": (status["storage_used"] / status["storage_limit"]) * 100
    } 