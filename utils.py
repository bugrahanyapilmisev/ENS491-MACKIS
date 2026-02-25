# backend/utils.py
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# GÜVENLİK AYARLARI
# Bunu normalde .env dosyasına koymalısın ama şimdilik burada kalsın
SECRET_KEY = "_fiwJ.UNX-QfhuvVt7vWqv6nXo3Yz7gqZkz-8Qz5U" 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 saat geçerli token



def verify_password(plain_password, hashed_password):
    """Kullanıcının girdiği şifreyi, DB'deki hash ile karşılaştırır."""
    # Bcrypt byte formatında çalışır, encode/decode işlemleri şarttır
    password_byte = plain_password.encode('utf-8')
    
    # Veritabanından gelen hash bazen str bazen bytes olabilir, kontrol edelim
    if isinstance(hashed_password, str):
        hashed_byte = hashed_password.encode('utf-8')
    else:
        hashed_byte = hashed_password

    return bcrypt.checkpw(password_byte, hashed_byte)

def get_password_hash(password):
    """Şifreyi veritabanına kaydetmeden önce hashler."""
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(pwd_bytes, salt)
    return hashed_bytes.decode('utf-8') # Veritabanına string olarak kaydetmek için

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """JWT Token oluşturur."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    """JWT token'dan user_id'yi çıkarır. Auth gerektiren her endpoint bunu kullanır."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Geçersiz veya süresi dolmuş token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
        return user_id
    except JWTError:
        raise credentials_exception
