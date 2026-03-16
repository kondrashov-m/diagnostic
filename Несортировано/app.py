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
import uuid
import gc
import zipfile
import tempfile

app = Flask(__name__)
CORS(app)

# Конфигурация
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'aac'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB максимум

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(file_path, output_path):
    """Конвертирует любой аудиофайл в WAV формат"""
    try:
        audio, sr = librosa.load(file_path, sr=22050, mono=True)
        sf.write(output_path, audio, sr, format='WAV')
        return audio, sr
    except Exception as e:
        raise Exception(f"Ошибка конвертации в WAV: {str(e)}")

def reduce_noise(audio, sr, noise_reduction_level=0.8):
    """Шумоподавление"""
    try:
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
    except Exception as e:
        print(f"❌ Ошибка шумоподавления: {e}")
        return audio

def create_simple_spectrogram(audio, sr, title):
    """Простое создание спектрограммы"""
    try:
        plt.figure(figsize=(8, 3))
        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"❌ Ошибка создания спектрограммы: {e}")
        plt.close()
        return None

def save_spectrogram_to_file(audio, sr, title, filepath):
    """Сохраняет спектрограмму в файл"""
    try:
        plt.figure(figsize=(8, 3))
        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        plt.savefig(filepath, format='png', dpi=80, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"❌ Ошибка сохранения спектрограммы: {e}")
        plt.close()
        return False

def process_single_file(file, session_id, noise_reduction_level=0.8, save_spectrograms=False):
    """Обработка одного файла"""
    try:
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())[:8]
        
        print(f"🔄 Обработка файла: {filename}")

        # Создаем временный файл
        temp_dir = os.path.join(UPLOAD_FOLDER, session_id, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        original_temp_path = os.path.join(temp_dir, f"temp_{file_id}_{filename}")
        file.save(original_temp_path)

        # Конвертируем в WAV
        wav_filename = f"{Path(filename).stem}_{file_id}.wav"
        wav_path = os.path.join(UPLOAD_FOLDER, session_id, wav_filename)
        audio, sr = convert_to_wav(original_temp_path, wav_path)
        duration = len(audio) / sr

        # Создаем директории для результатов
        base_name = Path(filename).stem
        processed_dir = os.path.join(PROCESSED_FOLDER, session_id)
        os.makedirs(processed_dir, exist_ok=True)

        # Сохраняем оригинал
        original_filename = f"original_{base_name}_{file_id}.wav"
        original_audio_path = os.path.join(processed_dir, original_filename)
        sf.write(original_audio_path, audio, sr)

        # Шумоподавление
        denoised_audio = reduce_noise(audio, sr, noise_reduction_level)
        denoised_filename = f"denoised_{base_name}_{file_id}.wav"
        denoised_path = os.path.join(processed_dir, denoised_filename)
        sf.write(denoised_path, denoised_audio, sr)

        # Остаточный шум
        residual_noise = audio - denoised_audio
        residual_filename = f"residual_{base_name}_{file_id}.wav"
        residual_path = os.path.join(processed_dir, residual_filename)
        sf.write(residual_path, residual_noise, sr)

        # СОЗДАЕМ СПЕКТРОГРАММЫ ДЛЯ ВСЕХ ФАЙЛОВ
        print(f"📊 Создание спектрограмм для {filename}...")
        original_spectrogram = create_simple_spectrogram(audio, sr, f'Оригинальный: {base_name}')
        denoised_spectrogram = create_simple_spectrogram(denoised_audio, sr, f'Очищенный: {base_name}')
        residual_spectrogram = create_simple_spectrogram(residual_noise, sr, f'Шум: {base_name}')
        
        # Сохраняем спектрограммы в файлы если нужно
        if save_spectrograms:
            spectrograms_dir = os.path.join(processed_dir, 'spectrograms')
            os.makedirs(spectrograms_dir, exist_ok=True)
            
            save_spectrogram_to_file(audio, sr, f'Оригинальный: {base_name}', 
                                   os.path.join(spectrograms_dir, f'original_{base_name}_{file_id}.png'))
            save_spectrogram_to_file(denoised_audio, sr, f'Очищенный: {base_name}', 
                                   os.path.join(spectrograms_dir, f'denoised_{base_name}_{file_id}.png'))
            save_spectrogram_to_file(residual_noise, sr, f'Шум: {base_name}', 
                                   os.path.join(spectrograms_dir, f'residual_{base_name}_{file_id}.png'))
        
        print(f"✅ Спектрограммы созданы для {filename}")

        # Base64 для аудио превью
        def audio_to_base64(audio_data, sr):
            max_preview = min(10 * sr, len(audio_data))
            preview_data = audio_data[:max_preview]
            
            buffer = io.BytesIO()
            sf.write(buffer, preview_data, sr, format='WAV')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')

        result = {
            'filename': filename,
            'file_id': file_id,
            'base_name': base_name,
            'duration': duration,
            'files': {
                'original_audio': f'/download/{session_id}/{original_filename}',
                'denoised_audio': f'/download/{session_id}/{denoised_filename}', 
                'residual_noise': f'/download/{session_id}/{residual_filename}'
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
            },
            'status': 'success'
        }

        # Очистка
        try:
            os.remove(original_temp_path)
        except:
            pass
            
        gc.collect()
        return result

    except Exception as e:
        print(f"💥 Ошибка обработки файла {file.filename}: {e}")
        traceback.print_exc()
        return {
            'filename': file.filename,
            'status': 'error',
            'error': str(e)
        }

# Глобальное хранилище для сессий
upload_sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_upload', methods=['POST'])
def start_upload():
    """Начинает новую сессию загрузки"""
    try:
        session_id = str(uuid.uuid4())[:8]
        file_count = int(request.json.get('file_count', 0))
        noise_level = float(request.json.get('noise_level', 0.8))
        
        upload_sessions[session_id] = {
            'total_files': file_count,
            'processed_files': 0,
            'noise_level': noise_level,
            'results': [],
            'status': 'active'
        }
        
        print(f"🚀 Начата сессия {session_id} для {file_count} файлов")
        return jsonify({'session_id': session_id, 'status': 'started'})
        
    except Exception as e:
        return jsonify({'error': f'Ошибка начала сессии: {str(e)}'}), 500

@app.route('/upload_chunk/<session_id>', methods=['POST'])
def upload_chunk(session_id):
    """Загружает и обрабатывает небольшую порцию файлов"""
    try:
        if session_id not in upload_sessions:
            return jsonify({'error': 'Сессия не найдена'}), 404
        
        if 'files' not in request.files:
            return jsonify({'error': 'Нет файлов'}), 400
        
        files = request.files.getlist('files')
        valid_files = [f for f in files if f and f.filename and allowed_file(f.filename)]
        
        if not valid_files:
            return jsonify({'error': 'Нет валидных файлов'}), 400
        
        print(f"📦 Обработка чанка из {len(valid_files)} файлов для сессии {session_id}")
        
        noise_level = upload_sessions[session_id]['noise_level']
        results = []
        
        for file in valid_files:
            result = process_single_file(file, session_id, noise_level, save_spectrograms=True)
            results.append(result)
            upload_sessions[session_id]['processed_files'] += 1
            upload_sessions[session_id]['results'].append(result)
        
        successful = [r for r in results if r.get('status') == 'success']
        
        return jsonify({
            'message': f'Обработано {len(successful)} из {len(valid_files)} файлов в чанке',
            'processed_in_chunk': len(valid_files),
            'successful_in_chunk': len(successful),
            'current_progress': upload_sessions[session_id]['processed_files'],
            'total_files': upload_sessions[session_id]['total_files']
        })
        
    except Exception as e:
        return jsonify({'error': f'Ошибка обработки чанка: {str(e)}'}), 500

@app.route('/get_progress/<session_id>')
def get_progress(session_id):
    """Возвращает прогресс обработки"""
    try:
        if session_id not in upload_sessions:
            return jsonify({'error': 'Сессия не найдена'}), 404
        
        session = upload_sessions[session_id]
        progress = (session['processed_files'] / session['total_files']) * 100 if session['total_files'] > 0 else 0
        
        return jsonify({
            'processed': session['processed_files'],
            'total': session['total_files'],
            'progress': progress,
            'status': session['status']
        })
        
    except Exception as e:
        return jsonify({'error': f'Ошибка получения прогресса: {str(e)}'}), 500

@app.route('/get_results/<session_id>')
def get_results(session_id):
    """Возвращает финальные результаты"""
    try:
        if session_id not in upload_sessions:
            return jsonify({'error': 'Сессия не найдена'}), 404
        
        session = upload_sessions[session_id]
        successful = [r for r in session['results'] if r.get('status') == 'success']
        failed = [r for r in session['results'] if r.get('status') == 'error']
        
        response = {
            'message': f'Обработано {len(successful)} из {session["total_files"]} файлов',
            'session_id': session_id,
            'noise_reduction_level': session['noise_level'],
            'processed_files': successful,
            'failed_files': failed,
            'summary': {
                'total': session['total_files'],
                'successful': len(successful),
                'failed': len(failed)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Ошибка получения результатов: {str(e)}'}), 500

@app.route('/download_spectrograms/<session_id>/<spectrogram_type>')
def download_spectrograms(session_id, spectrogram_type):
    """Скачивание архивов со спектрограммами"""
    try:
        if session_id not in upload_sessions:
            return jsonify({'error': 'Сессия не найдена'}), 404
        
        session = upload_sessions[session_id]
        successful_files = [r for r in session['results'] if r.get('status') == 'success']
        
        if not successful_files:
            return jsonify({'error': 'Нет обработанных файлов для скачивания'}), 400
        
        # Создаем временный zip-архив
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_result in successful_files:
                base_name = file_result.get('base_name', Path(file_result['filename']).stem)
                file_id = file_result.get('file_id', '')
                
                spectrogram_filename = f"{spectrogram_type}_{base_name}_{file_id}.png"
                spectrogram_path = os.path.join(PROCESSED_FOLDER, session_id, 'spectrograms', spectrogram_filename)
                
                if os.path.exists(spectrogram_path):
                    # Добавляем файл в архив с понятным именем
                    archive_name = f"{base_name}_{spectrogram_type}.png"
                    zipf.write(spectrogram_path, archive_name)
        
        # Определяем имя файла для скачивания
        if spectrogram_type == 'original':
            download_filename = f'исходные_спектрограммы_{session_id}.zip'
        elif spectrogram_type == 'denoised':
            download_filename = f'обработанные_спектрограммы_{session_id}.zip'
        elif spectrogram_type == 'residual':
            download_filename = f'спектрограммы_шумов_{session_id}.zip'
        else:
            download_filename = f'спектрограммы_{session_id}.zip'
        
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name=download_filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        return jsonify({'error': f'Ошибка создания архива: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    """Старый метод для небольших загрузок (макс 5 файлов)"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'Нет файлов!'}), 400
        
        files = request.files.getlist('files')
        noise_reduction_level = float(request.form.get('noise_level', 0.8))
        
        valid_files = [f for f in files if f and f.filename and allowed_file(f.filename)]
        
        if not valid_files:
            return jsonify({'error': 'Нет подходящих файлов!'}), 400
        
        if len(valid_files) > 5:
            return jsonify({'error': 'Для больших загрузок используйте потоковый метод. Максимум 5 файлов для этого метода.'}), 400
        
        session_id = str(uuid.uuid4())[:8]
        results = []
        
        for file in valid_files:
            result = process_single_file(file, session_id, noise_reduction_level, save_spectrograms=True)
            results.append(result)
        
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'error']
        
        response = {
            'message': f'Обработано {len(successful)} из {len(valid_files)} файлов',
            'session_id': session_id,
            'noise_reduction_level': noise_reduction_level,
            'processed_files': successful,
            'failed_files': failed,
            'summary': {
                'total': len(valid_files),
                'successful': len(successful),
                'failed': len(failed)
            }
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Ошибка сервера: {str(e)}'}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """Скачивание файлов"""
    try:
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
            file_path = os.path.join(folder, filename)
            if os.path.exists(file_path):
                return send_file(file_path, as_attachment=True)
        return jsonify({'error': 'Файл не найден!'}), 404
    except Exception as e:
        return jsonify({'error': f'Ошибка скачивания: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Слишком большой размер запроса! Используйте потоковую загрузку.'}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"💥 ОШИБКА: {e}")
    return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Запуск системы с возможностью скачивания спектрограмм!")
    print("📁 Загружайте ЛЮБОЕ количество файлов!")
    print("📊 Спектрограммы создаются и сохраняются для скачивания!")
    print("🗂️  Доступны архивы со спектрограммами!")
    print("🌐 Сервер: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)