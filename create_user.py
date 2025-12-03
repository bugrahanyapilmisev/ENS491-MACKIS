# backend/create_user.py
from database import SessionLocal
import models
from utils import get_password_hash

db = SessionLocal()

email = "ahmet@sabanciuniv.edu"
sifre = "654321"  # <-- Buraya istediğin şifreyi yaz

print(f"Kullanıcı oluşturuluyor: {email}")

# Hashlenmiş şifreyi oluştur
hashed_pw = get_password_hash(sifre)

# Kullanıcıyı DB'ye ekle
new_user = models.User(
    email=email,
    display_name="Student User",
    role="undergrad",
    status="active",
    password_hash=hashed_pw 
)

try:
    db.add(new_user)
    db.commit()
    print("✅ Kullanıcı başarıyla oluşturuldu!")
except Exception as e:
    print(f"❌ Hata: {e}")
finally:
    db.close()