from flask import Flask, render_template, request, redirect, url_for
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import sqlite3

# SQLite veritabanı dosyasını oluştur veya bağlan
db_path = "aes_project.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

# Tablo oluşturma (eğer tablo yoksa)
cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_text TEXT,
    encrypted_text TEXT,
    decrypted_text TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()


app = Flask(__name__)

# AES ile şifreleme
def aes_sifrele(metin, anahtar):
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(anahtar, AES.MODE_CBC, iv)
    metin_padded = pad(metin.encode(), AES.block_size)
    sifreli_metin = cipher.encrypt(metin_padded)
    return iv + sifreli_metin

# AES ile şifre çözme
def aes_sifreyi_coz(sifreli_metin, anahtar):
    iv = sifreli_metin[:AES.block_size]
    sifreli_metin = sifreli_metin[AES.block_size:]
    cipher = AES.new(anahtar, AES.MODE_CBC, iv)
    orijinal_metin = unpad(cipher.decrypt(sifreli_metin), AES.block_size)
    return orijinal_metin.decode()
@app.route('/messages')
def messages():
    query = "SELECT * FROM messages"
    cursor.execute(query)
    rows = cursor.fetchall()
    return render_template('messages.html', rows=rows)

# Anahtar oluşturma (AES 256-bit anahtar)
anahtar = get_random_bytes(32)  # 256-bit anahtar (32 byte)

# Anasayfa
@app.route('/')
def index():
    return render_template('index.html')

# Şifreleme işlemi
@app.route('/encrypt', methods=['POST'])
def encrypt():
    if request.method == 'POST':
        metin = request.form['metin']
        sifreli_metin = aes_sifrele(metin, anahtar)

        # Şifreli metni SQLite veritabanına kaydet
        query = "INSERT INTO messages (original_text, encrypted_text) VALUES (?, ?)"
        cursor.execute(query, (metin, sifreli_metin.hex()))
        conn.commit()

        return render_template('index.html', sifreli_metin=sifreli_metin.hex())


# Şifre çözme işlemi
@app.route('/decrypt', methods=['POST'])
def decrypt():
    if request.method == 'POST':
        sifreli_metin_hex = request.form['sifreli_metin']
        sifreli_metin = bytes.fromhex(sifreli_metin_hex)
        cozulmus_metin = aes_sifreyi_coz(sifreli_metin, anahtar)

        # Çözülmüş metni SQLite veritabanına güncelle
        query = "UPDATE messages SET decrypted_text = ? WHERE encrypted_text = ?"
        cursor.execute(query, (cozulmus_metin, sifreli_metin_hex))
        conn.commit()

        return render_template('index.html', cozulmus_metin=cozulmus_metin)


if __name__ == '__main__':
    app.run(debug=True)
