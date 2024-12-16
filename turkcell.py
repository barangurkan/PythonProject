from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64

def aes_sifrele(metin, anahtar):
    cipher = AES.new(anahtar, AES.MODE_ECB)
    metin_padded = pad(metin.encode(), AES.block_size)
    sifreli_metin = cipher.encrypt(metin_padded)
    return base64.b64encode(sifreli_metin).decode()

def aes_sifre_coz(sifreli_metin, anahtar):
    sifreli_metin = base64.b64decode(sifreli_metin)
    cipher = AES.new(anahtar, AES.MODE_ECB)
    orijinal_metin = unpad(cipher.decrypt(sifreli_metin), AES.block_size)
    return orijinal_metin.decode()

# Test
anahtar = base64.b64decode("4JkjWy8llbgGL76LLOHm7Q==")  # Anahtarı decode et
sifreli_veri = "MjhJZCtPSXlYVGViZFUweVl1Z2NPUT09"  # Şifreli veriyi kullan

try:
    cozulmus_metin = aes_sifre_coz(sifreli_veri, anahtar)
    print("Çözülen Metin:", cozulmus_metin)
except Exception as e:
    print("Şifre çözme başarısız:", str(e))
