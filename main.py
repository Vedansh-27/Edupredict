"""
EduPredict Cloud — Root ASGI Entrypoint
=======================================
Enables both `uvicorn main:app` and `uvicorn app.main:app` to work
seamlessly regardless of how the start command is configured in Render.
"""

from app.main import app

__all__ = ["app"]
