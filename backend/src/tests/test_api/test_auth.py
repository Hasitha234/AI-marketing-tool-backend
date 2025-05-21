import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.main import app
from app.schemas.user import UserCreate
from app.crud.user import create_user

client = TestClient(app)

@pytest.fixture
def test_user(db: Session) -> dict:
    """Create a test user for other tests."""
    user_in = UserCreate(
        email="test@example.com",
        username="testuser",
        password="testpassword123",
        is_active=True,
    )
    user = create_user(db, user_in)
    return {
        "email": user.email,
        "username": user.username,
        "password": "testpassword123",
        "id": user.id
    }

def test_register_user_success(db: Session) -> None:
    """Test successful user registration."""
    data = {
        "email": "new@example.com",
        "username": "newuser",
        "password": "newpassword123",
    }
    
    response = client.post(f"{settings.API_V1_STR}/auth/register", json=data)
    
    assert response.status_code == 200
    content = response.json()
    assert content["email"] == data["email"]
    assert content["username"] == data["username"]
    assert content["role"] == "user"
    assert "id" in content
    assert "hashed_password" not in content

def test_register_user_duplicate_email(db: Session, test_user: dict) -> None:
    """Test registration with duplicate email."""
    data = {
        "email": test_user["email"],
        "username": "differentuser",
        "password": "password123",
    }
    
    response = client.post(f"{settings.API_V1_STR}/auth/register", json=data)
    assert response.status_code == 400
    assert "email already registered" in response.json()["detail"].lower()

def test_register_user_duplicate_username(db: Session, test_user: dict) -> None:
    """Test registration with duplicate username."""
    data = {
        "email": "different@example.com",
        "username": test_user["username"],
        "password": "password123",
    }
    
    response = client.post(f"{settings.API_V1_STR}/auth/register", json=data)
    assert response.status_code == 400
    assert "username already registered" in response.json()["detail"].lower()

def test_register_user_invalid_password(db: Session) -> None:
    """Test registration with invalid password."""
    data = {
        "email": "valid@example.com",
        "username": "validuser",
        "password": "short",  # Too short
    }
    
    response = client.post(f"{settings.API_V1_STR}/auth/register", json=data)
    assert response.status_code == 422
    assert "password" in response.json()["detail"][0]["loc"]

def test_login_success(db: Session, test_user: dict) -> None:
    """Test successful login."""
    login_data = {
        "username": test_user["email"],
        "password": test_user["password"],
    }
    
    response = client.post(
        f"{settings.API_V1_STR}/auth/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    assert response.status_code == 200
    content = response.json()
    assert content["token_type"] == "bearer"
    assert "access_token" in content
    return content["access_token"]

def test_login_wrong_password(db: Session, test_user: dict) -> None:
    """Test login with wrong password."""
    login_data = {
        "username": test_user["email"],
        "password": "wrongpassword",
    }
    
    response = client.post(
        f"{settings.API_V1_STR}/auth/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    assert response.status_code == 401
    assert "incorrect" in response.json()["detail"].lower()

def test_login_nonexistent_user(db: Session) -> None:
    """Test login with non-existent user."""
    login_data = {
        "username": "nonexistent@example.com",
        "password": "password123",
    }
    
    response = client.post(
        f"{settings.API_V1_STR}/auth/login",
        data=login_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    assert response.status_code == 401
    assert "incorrect" in response.json()["detail"].lower()

def test_get_current_user_success(db: Session, test_user: dict) -> None:
    """Test getting current user with valid token."""
    # First login to get token
    token = test_login_success(db, test_user)
    
    # Get current user
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get(f"{settings.API_V1_STR}/auth/me", headers=headers)
    
    assert response.status_code == 200
    content = response.json()
    assert content["email"] == test_user["email"]
    assert content["username"] == test_user["username"]
    assert "hashed_password" not in content

def test_get_current_user_invalid_token(db: Session) -> None:
    """Test getting current user with invalid token."""
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.get(f"{settings.API_V1_STR}/auth/me", headers=headers)
    
    assert response.status_code == 401
    assert "invalid" in response.json()["detail"].lower()

def test_get_current_user_no_token(db: Session) -> None:
    """Test getting current user without token."""
    response = client.get(f"{settings.API_V1_STR}/auth/me")
    
    assert response.status_code == 401
    assert "not provided" in response.json()["detail"].lower()