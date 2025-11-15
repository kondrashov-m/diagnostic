from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr
from werkzeug.utils import secure_filename
import io
import base64
from pathlib import Path
import traceback

app = Flask(__name__)
CORS(app)

# Конфигурация
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'aac'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def reduce_noise(audio, sr, noise_reduction_level=0.8):
    """Шумоподавление"""
    if len(audio) > int(0.5 * sr):
        noise_sample = audio[:int(0.5 * sr)]
    else:
        noise_sample = audio
    
    reduced_noise = nr.reduce_noise(
        y=audio, 
        sr=sr, 
        y_noise=noise_sample,
        prop_decrease=noise_reduction_level,
        stationary=False
    )
    
    return reduced_noise

def create_spectrogram_base64(audio, sr, title):
    """Создание спектрограммы и возврат как base64"""
    try:
        plt.figure(figsize=(10, 4))
        
        # Простая спектрограмма
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        # Сохраняем в buffer как base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        # Конвертируем в base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        print(f"❌ Ошибка создания спектрограммы: {e}")
        plt.close()
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("🔄 Начало обработки файла...")
    
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла!'}), 400
    
    file = request.files['file']
    noise_reduction_level = float(request.form.get('noise_level', 0.8))
    
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран!'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Сохраняем файл
            filename = secure_filename(file.filename)
            original_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(original_path)
            print(f"📁 Файл сохранен: {original_path}")

            # Загружаем аудио
            audio, sr = librosa.load(original_path, sr=22050)
            print(f"🎵 Аудио загружено: {len(audio)} samples")

            # Сохраняем оригинал
            original_filename = f"original_{Path(filename).stem}.wav"
            original_audio_path = os.path.join(PROCESSED_FOLDER, original_filename)
            sf.write(original_audio_path, audio, sr)

            # Шумоподавление
            denoised_audio = reduce_noise(audio, sr, noise_reduction_level)
            denoised_filename = f"denoised_{Path(filename).stem}.wav"
            denoised_path = os.path.join(PROCESSED_FOLDER, denoised_filename)
            sf.write(denoised_path, denoised_audio, sr)

            # Остаточный шум
            residual_noise = audio - denoised_audio
            residual_filename = f"residual_{Path(filename).stem}.wav"
            residual_path = os.path.join(PROCESSED_FOLDER, residual_filename)
            sf.write(residual_path, residual_noise, sr)

            # Создаем спектрограммы как base64
            print("🖼️ Создаем спектрограммы...")
            
            original_spectrogram = create_spectrogram_base64(audio, sr, 'Оригинальный звук')
            denoised_spectrogram = create_spectrogram_base64(denoised_audio, sr, 'Очищенный звук')
            residual_spectrogram = create_spectrogram_base64(residual_noise, sr, 'Удаленный шум')

            # Base64 для аудио превью
            def audio_to_base64(audio_data, sr):
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, sr, format='WAV')
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode('utf-8')

            response = {
                'message': 'Файл обработан!',
                'noise_reduction_level': noise_reduction_level,
                'files': {
                    'original_audio': f'/download/{original_filename}',
                    'denoised_audio': f'/download/{denoised_filename}', 
                    'residual_noise': f'/download/{residual_filename}'
                },
                'spectrograms': {
                    'original': original_spectrogram,
                    'denoised': denoised_spectrogram,
                    'residual': residual_spectrogram
                },
                'preview_audio': {
                    'original': audio_to_base64(audio, sr),
                    'denoised': audio_to_base64(denoised_audio, sr),
                    'residual': audio_to_base64(residual_noise, sr)
                }
            }

            print("✅ Обработка завершена!")
            return jsonify(response)

        except Exception as e:
            error_msg = f'Ошибка: {str(e)}'
            print(f"💥 {error_msg}")
            traceback.print_exc()
            return jsonify({'error': error_msg}), 500
    
    return jsonify({'error': 'Неподдерживаемый формат файла!'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    """Скачивание аудиофайлов"""
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'Файл не найден!'}), 404

if __name__ == '__main__':
    print("🚀 Запуск системы...")
    print(f"📁 Upload: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"📁 Processed: {os.path.abspath(PROCESSED_FOLDER)}")
    print("🌐 Сервер: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)