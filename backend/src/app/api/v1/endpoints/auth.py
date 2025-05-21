from datetime import timedelta
from typing import Any 

from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.core.config import settings
from app.core.security import create_access_token, get_current_user
from app.crud.user import authenticate_user, create_user, get_user_by_email
from app.models.user import User
from app.schemas.token import Token
from app.schemas.user import User as UserSchema, UserCreate

router = APIRouter()

@router.post("/login", response_model=Token)
async def login_access_token(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """ OAuth2 compatible token login, get an access token for future requests """
    user = authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject = user.email, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }

@router.post("/register", response_model=UserSchema)
async def register_user(
    *,
    db: Session = Depends(get_db),
    user_in: UserCreate,
) -> Any:
    """ Create a new user """
    user = get_user_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The user with this email already exists in the system",
        )
    user = create_user(db, obj_in=user_in)
    return user

@router.get("/me", response_model=UserSchema)
async def read_users_me(
    current_user: User = Depends(get_current_user),
) -> Any:
    """ Get details of currently authenticated user """
    return current_user


        
