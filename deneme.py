from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64

# Sabit anahtar
anahtar_base64 = "V9VxCInkG4OBTbQ51whI+A=="
anahtar = base64.b64decode(anahtar_base64)  # Sabit anahtarın Base64'ten çözülmesi


# Şifreleme fonksiyonu (IV'siz)
def aes_sifrele(metin):
    # AES şifreleyicisini ECB modunda oluşturuyoruz (IV yok)
    cipher = AES.new(anahtar, AES.MODE_ECB)
    print("paddingden önce",metin)
    # Veriyi padding'le 16 byte bloklarına hizalıyoruz
    metin_padded = pad(metin.encode(), AES.block_size)
    print("paddingden sonra",metin_padded)
    # Şifreli metni elde ediyoruz
    sifreli_metin = cipher.encrypt(metin_padded)
    print("en son şifreli hali",sifreli_metin)

    # Şifreli metni Base64'e kodluyoruz
    sifreli_metin_base64 = base64.b64encode(sifreli_metin).decode()

    return sifreli_metin_base64


# Şifre çözme fonksiyonu (IV'siz)
def aes_sifre_coz(sifreli_metin_base64):
    try:
        # Base64'ten çözme
        sifreli_metin = base64.b64decode(sifreli_metin_base64)

        # AES şifre çözücüsünü ECB modunda oluşturuyoruz (IV yok)
        cipher = AES.new(anahtar, AES.MODE_ECB)

        # Şifreyi çözme ve padding'i kaldırma
        orijinal_metin = unpad(cipher.decrypt(sifreli_metin), AES.block_size)
        cozulmus_metin = orijinal_metin.decode()  # Çözülmüş metni string olarak alıyoruz
        return cozulmus_metin
    except (ValueError, TypeError) as e:
        return f"Decryption failed: {str(e)}"


# Örnek kullanım
metin = "1999gurkan"
sifreli_metin = aes_sifrele(metin)  # Şifreleme yap

# Şifreli metni ve anahtarı veritabanına kaydedebilirsiniz
print(f"Şifreli Metin: {sifreli_metin}")

# Şifre çözme
cozulmus_metin = aes_sifre_coz(sifreli_metin)
print(f"Çözülmüş Metin: {cozulmus_metin}")
print(aes_sifre_coz("IZ+0zZ4QtRY2BwEq1TXFX/0YNyDuvT4x10vBpwPSmYY="))
