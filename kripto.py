from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64

# AES anahtarı oluşturma
def aes_anahtar_olustur():
    anahtar = get_random_bytes(16)  # AES için 128 bit anahtar (16 byte)
    return anahtar

# AES ile şifreleme
def aes_sifrele(metin, anahtar):
    cipher = AES.new(anahtar, AES.MODE_CBC)  # CBC modunda şifreleme
    # Metni blok boyutuna uygun şekilde dolduruyoruz
    metin_padded = pad(metin.encode(), AES.block_size)
    sifreli_metin = cipher.encrypt(metin_padded)
    # Şifreli metni ve IV'yi (Initialization Vector) base64 formatında döndürüyoruz
    return base64.b64encode(cipher.iv + sifreli_metin).decode()

# AES ile şifre çözme
def aes_sifreyi_coz(sifreli_metin, anahtar):
    # Base64'ten çözme
    sifreli_metin = base64.b64decode(sifreli_metin)
    iv = sifreli_metin[:16]  # IV ilk 16 bayttır
    sifreli_metin = sifreli_metin[16:]  # Geriye kalan kısmı şifreli metin
    cipher = AES.new(anahtar, AES.MODE_CBC, iv)
    orijinal_metin = unpad(cipher.decrypt(sifreli_metin), AES.block_size)
    return orijinal_metin.decode()

# Kullanıcıdan metin al
metin = input("Şifrelemek istediğiniz metni girin: ")

# AES anahtarı oluştur
anahtar = aes_anahtar_olustur()

# AES ile şifrele
sifreli_metin = aes_sifrele(metin, anahtar)
print("Şifreli Metin:", sifreli_metin)

# AES ile çöz
cozulmus_metin = aes_sifreyi_coz(sifreli_metin, anahtar)
print("Çözülmüş Metin:", cozulmus_metin)
