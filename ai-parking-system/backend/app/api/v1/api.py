"""
Main API router for the AI Parking Management System
"""

from fastapi import APIRouter
from app.api.v1.endpoints import vehicles, slots, reservations, analytics, payments

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(vehicles.router, prefix="/vehicles", tags=["vehicles"])
api_router.include_router(slots.router, prefix="/slots", tags=["slots"])
api_router.include_router(reservations.router, prefix="/reservations", tags=["reservations"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(payments.router, prefix="/payments", tags=["payments"])
