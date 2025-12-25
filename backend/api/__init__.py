"""API router package for LLM Council."""

from fastapi import APIRouter
from .config import router as config_router
from .export import router as export_router

# Create main API router
api_router = APIRouter(prefix="/api")

# Include sub-routers
api_router.include_router(config_router)
api_router.include_router(export_router)
