"""schemas.py — Pydantic request/response models for the FastAPI layer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


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

class IngestRepoRequest(BaseModel):
    github_url: str = Field(min_length=1, max_length=500)


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
