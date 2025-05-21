from typing import Any, List

from fastapi import APIRouter, Body, Depends, HTTPException, status, Path
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.core.permissions import allow_admin, allow_all_authenticated
from app.core.security import get_current_user
from app.crud.user import get_user, get_users, update_user, delete_user
from app.models.user import User
from app.schemas.user import User as UserSchema, UserUpdate

router = APIRouter()

@router.get("/", response_model=List[UserSchema])
def read_users(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(allow_all_authenticated),
) -> Any:
    """Retrieve users. Only accessible to admins and managers."""
    users = get_users(db, skip=skip, limit=limit)
    return users

@router.get("/{user_id}", response_model=UserSchema)
def read_user_by_id(
    user_id: int = Path(..., ge=1),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Get a specific user by id. Users can only see themselves, admins can see anyone."""
    user = get_user(db, user_id=user_id)
    if user == current_user:
        return user
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges",
        )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user

@router.put("/{user_id}", response_model=UserSchema)
def update_user_by_id(
    *,
    user_id: int = Path(..., ge=1),
    user_in: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Update a user. Users can only update themselves, admins can update anyone."""
    user = get_user(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    if user.id == current_user.id:
        # Users can update their own information but not their role
        if user_in.role and user_in.role != current_user.role and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to modify role",
            )
        return update_user(db, db_obj=user, obj_in=user_in)
    
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges",
        )
    
    return update_user(db, db_obj=user, obj_in=user_in)

@router.patch("/{user_id}/role", response_model=UserSchema)
def update_user_role(
    *,
    user_id: int = Path(..., ge=1),
    role: str = Body(..., embed=True),
    current_user: User = Depends(allow_admin),
    db: Session = Depends(get_db),
) -> Any:
    """Update a user's role. Only accessible to admins."""
    user = get_user(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    if role not in ["user", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role",
        )
    
    user_in = UserUpdate(role=role)
    return update_user(db, db_obj=user, obj_in=user_in)

@router.delete("/{user_id}", response_model=UserSchema)
def delete_user_by_id(
    *,
    user_id: int = Path(..., ge=1),
    current_user: User = Depends(allow_admin),
    db: Session = Depends(get_db),
) -> Any:
    """Delete a user. Only accessible to admins."""
    user = get_user(db, user_id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    deleted_user = delete_user(db, user_id=user_id)
    return deleted_user