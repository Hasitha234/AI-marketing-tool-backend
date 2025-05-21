from typing import List, Optional
from fastapi import HTTPException, status, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.models.user import User

class RoleChecker:
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_user)):
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {current_user.role} not authorized to access this resource",
            )
        return current_user

allow_admin = RoleChecker(["admin"])
allow_all_authenticated = RoleChecker(["admin", "user"])






