"""schemas.py — Pydantic request/response models for the FastAPI layer."""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator


# --- Auth ---

class SignUpRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Repos ---

_GITHUB_REPO_URL = re.compile(
    r"^https?://(?:www\.)?github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/?$"
)


class IngestRepoRequest(BaseModel):
    github_url: str = Field(min_length=1, max_length=500)

    @field_validator("github_url")
    @classmethod
    def must_be_a_github_repo_url(cls, value: str) -> str:
        """
        Rejects anything that isn't a plain github.com/<owner>/<repo> URL.

        Without this, any string reached the ingestion background task and
        became a Repo row that could only ever fail — "localhost:5173" and
        "github.com/<user>?tab=repositories" (a profile page, not a repo)
        both got saved that way, cluttering the sidebar with dead entries.
        Validating here rejects them with a 422 before any row is written.
        """
        url = value.strip().rstrip("/")
        if url.startswith("github.com/"):
            url = f"https://{url}"
        if not _GITHUB_REPO_URL.match(url):
            raise ValueError(
                "Must be a GitHub repository URL, e.g. "
                "https://github.com/owner/repo"
            )
        return url


class RepoResponse(BaseModel):
    id: str
    github_url: str
    display_name: str
    total_chunks: int
    status: str
    error_message: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Conversations & messages ---

class CreateConversationRequest(BaseModel):
    repo_id: str


class ConversationResponse(BaseModel):
    id: str
    repo_id: str
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    refused: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    conversation_id: str
    question: str = Field(min_length=1, max_length=2000)
