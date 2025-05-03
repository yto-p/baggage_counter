import os
import uuid
import subprocess
import json
from datetime import datetime
from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import imageio_ffmpeg
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = os.path.join(UPLOAD_FOLDER, 'results')
HISTORY_FILE = 'history.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO('yolov5s.pt')
baggage_ids = [24, 26, 28]  # backpack, handbag, suitcase

# Загружаем историю
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        history = json.load(f)
else:
    history = []


def save_history():
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


@app.route('/')
def index():
    return render_template('index.html', result=None, video_path=None, history=history)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return render_template('index.html', result="Файл не выбран", history=history)

    file = request.files['video']
    if file.filename == '':
        return render_template('index.html', result="Имя файла пустое", history=history)

    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    save_dir = os.path.join(RESULT_FOLDER, str(uuid.uuid4()))
    os.makedirs(save_dir, exist_ok=True)

    model.track(source=filepath, persist=True, save=True, project=save_dir, name='', exist_ok=True)

    tracked_video = None
    for root, _, files in os.walk(save_dir):
        for f in files:
            if f.endswith('.avi'):
                tracked_video = os.path.join(root, f)
                break

    if not tracked_video:
        return render_template('index.html', result="Обработка не удалась: .avi файл не найден.", history=history)

    mp4_video = tracked_video.replace('.avi', '.mp4')
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    subprocess.run([
        ffmpeg_path, '-y', '-i', tracked_video,
        '-vcodec', 'libx264', '-acodec', 'aac', mp4_video
    ], check=True)

    os.remove(tracked_video)

    unique_ids = set()
    for result in model.track(source=filepath, persist=True, stream=True):
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for obj_id, cls_id in zip(boxes.id, boxes.cls):
                if int(cls_id) in baggage_ids:
                    unique_ids.add(int(obj_id.item()))

    baggage_count = len(unique_ids)
    video_path = '/' + mp4_video.replace('\\', '/').replace(os.sep, '/')

    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'filename': os.path.basename(filepath),
        'result_video': video_path,
        'count': baggage_count
    }
    history.insert(0, record)
    save_history()

    return render_template('index.html', result=f'Найдено уникальных объектов багажа: {baggage_count}',
                           video_path=video_path, history=history)


@app.route('/download-history')
def download_history():
    if not history:
        return "История пуста"
    df = pd.DataFrame(history)
    file_path = 'history.xlsx'
    df.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
