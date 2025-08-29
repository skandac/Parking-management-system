"""
AI Parking Management System - FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.v1.api import api_router
from app.core.events import create_start_app_handler, create_stop_app_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await create_start_app_handler(app)()
    yield
    # Shutdown
    await create_stop_app_handler(app)()


def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="AI Parking Management System",
        description="Advanced parking management with AI-powered vehicle detection and analytics",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=settings.ALLOWED_HOSTS
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {
            "message": "AI Parking Management System API",
            "version": "1.0.0",
            "status": "operational"
        }

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

    return app


app = create_application()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
