import os
import telebot
from flask import Flask, request
import tensorflow as tf
import numpy as np
import cv2

TOKEN = "MASUKKAN_BOT_TOKEN_TELEGRAM_KAMU"  
bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

# Ganti 'nama_file_model.h5' dengan nama file model aslimu dari Edge Impulse
model = tf.keras.models.load_model('nama_file_model.h5')
CLASSES = ['masker', 'obat', 'uang_50rb']  # Sesuaikan urutan kelas saat training

@app.route(f"/{TOKEN}", methods=["POST"])
def receive_message():
    json_string = request.get_data().decode("utf-8")
    update = telebot.types.Update.de_json(json_string)
    bot.process_new_updates([update])
    return "!", 200

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    image_path = "downloaded_image.jpg"
    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (96, 96)) 
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  

    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions[0])
    hasil_klasifikasi = CLASSES[predicted_class_index]
    akurasi = float(np.max(predictions[0]) * 100)

    bot.reply_to(message, f"🤖 *Hasil Klasifikasi AI Cloud*\n\n"
                          f"📌 Objek: *{hasil_klasifikasi}*\n"
                          f"📊 Akurasi: *{akurasi:.2f}%*")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))