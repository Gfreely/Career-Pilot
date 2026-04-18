from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routers import chat_router, interview_router, profile_router


def create_app() -> FastAPI:
    """创建 FastAPI 应用。"""
    app = FastAPI(title="XinghuoLLM API", version="0.1.0")
    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"

    @app.get("/healthz", tags=["system"])
    def healthz():
        return {"status": "ok"}

    app.include_router(chat_router)
    app.include_router(interview_router)
    app.include_router(profile_router)

    if frontend_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dir)), name="assets")

        @app.get("/", include_in_schema=False)
        def frontend_index():
            return FileResponse(frontend_dir / "index.html")

    return app


app = create_app()
