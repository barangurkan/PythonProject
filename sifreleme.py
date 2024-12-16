from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64


def generate_key():
    """Rastgele bir 256-bit (32 byte) anahtar oluşturur."""
    return get_random_bytes(32)


def encrypt(plain_text, key):
    """Verilen düz metni AES-GCM kullanarak şifreler.

    Args:
        plain_text (str): Şifrelenecek metin.
        key (bytes): AES anahtarı (32 byte).

    Returns:
        dict: Şifrelenmiş veriyi ve şifreleme için kullanılan nonce'u içerir.
    """
    cipher = AES.new(key, AES.MODE_GCM)
    nonce = cipher.nonce
    cipher_text, tag = cipher.encrypt_and_digest(plain_text.encode("utf-8"))

    return {
        "cipher_text": base64.b64encode(cipher_text).decode("utf-8"),
        "nonce": base64.b64encode(nonce).decode("utf-8"),
        "tag": base64.b64encode(tag).decode("utf-8")
    }


def decrypt(encrypted_data, key):
    """AES-GCM ile şifrelenmiş metni çözer.

    Args:
        encrypted_data (dict): Şifrelenmiş metni, nonce'u ve tag'i içerir.
        key (bytes): AES anahtarı (32 byte).

    Returns:
        str: Çözülen düz metin.
    """
    cipher_text = base64.b64decode(encrypted_data["cipher_text"])
    nonce = base64.b64decode(encrypted_data["nonce"])
    tag = base64.b64decode(encrypted_data["tag"])

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    plain_text = cipher.decrypt_and_verify(cipher_text, tag)

    return plain_text.decode("utf-8")


# Örnek Kullanım
if __name__ == "__main__":
    # Rastgele bir anahtar oluştur
    key = generate_key()
    print(f"Anahtar: {base64.b64encode(key).decode('utf-8')}")

    # Şifrelenecek metin
    plain_text = "Bu bir test mesajıdır."
    print(f"Orijinal Metin: {plain_text}")

    # Metni şifrele
    encrypted = encrypt(plain_text, key)
    print(f"Şifrelenmiş Veri: {encrypted}")

    # Metni çöz
    decrypted = decrypt(encrypted, key)
    print(f"Çözülmüş Metin: {decrypted}")