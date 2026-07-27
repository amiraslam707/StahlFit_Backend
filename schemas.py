from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ProfileUpdateRequest(BaseModel):
    username: Optional[str] = None


class UserOut(BaseModel):
    id: int
    email: EmailStr
    username: str
    avatar_url: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True  # on Pydantic v1, use `orm_mode = True` instead — check with: pip show pydantic


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut