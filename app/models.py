from __future__ import annotations

from pydantic import BaseModel, Field


class TokenInfo(BaseModel):
    access_token: str = Field(..., description="Spotify access token")
    refresh_token: str | None = Field(None, description="Spotify refresh token")
    expires_at: int | None = Field(None, description="Epoch seconds when the token expires")


class UserProfile(BaseModel):
    id: str
    display_name: str | None = None
    email: str | None = None
    country: str | None = None
    product: str | None = None


