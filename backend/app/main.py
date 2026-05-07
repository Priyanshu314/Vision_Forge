from fastapi import FastAPI

from ..core.config import load_config
from ..routes import upload, sampling, annotation


def create_app() -> FastAPI:
    app = FastAPI(title="Industrial Defect Detection API")

    # Load configuration once at startup
    load_config("backend/config.yaml")

    # Include routers
    app.include_router(upload.router, tags=["Upload"])
    app.include_router(sampling.router, tags=["Sampling"])
    app.include_router(annotation.router, tags=["Annotation"])

    return app


app = create_app()
