# backend/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import get_db
import models
from utils import verify_password, create_access_token

router = APIRouter()

# Frontend'den gelecek giriş verisi modeli
class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/auth/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    # 1. Kullanıcıyı bul
    user = db.query(models.User).filter(models.User.email == request.email).first()
    
    # 2. Kullanıcı yoksa veya şifre yanlışsa hata ver
    if not user or not user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-posta veya şifre hatalı",
        )
    
    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-posta veya şifre hatalı",
        )

    # 3. Her şey doğruysa Token oluştur
    access_token = create_access_token(data={"sub": user.email, "user_id": str(user.user_id)})
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user_name": user.display_name,
        "is_admin": user.role == "admin"
    }