"""Compatibility re-exports of the HTTP API Pydantic contracts."""

from api.schema import AskRequest, AskResponse, HealthResponse

__all__ = ["AskRequest", "AskResponse", "HealthResponse"]
