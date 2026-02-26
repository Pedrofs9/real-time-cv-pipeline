"""
Development entrypoint for running the FastAPI application directly with uvicorn.
Not used in Docker — the docker-compose command calls uvicorn directly.
Use this for local development outside of containers.
"""
import uvicorn
from core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,  # Use True in development, False in production
        log_level=settings.log_level.lower(),  # Match configured log level
        workers=settings.workers  # Number of worker processes
    )
