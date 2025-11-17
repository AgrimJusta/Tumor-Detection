import os
import time
import threading
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
from model_utils import train_from_folder, infer_image, get_model, load_image_for_infer
from datetime import datetime

BASE = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE, 'uploads')
MODEL_FOLDER = os.path.join(BASE, 'models')
REPORT_FOLDER = os.path.join(BASE, 'reports')
ALLOWED_EXT = { 'png','jpg','jpeg','dcm','zip' }

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'change-me-local-demo'

# global simple status dictionary (shared with background thread)
status = { 'phase':'idle', 'progress':0, 'last_message':'idle' }

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/')
def index():
    # render UI, pass status (it's read by ajax but show initial)
    return render_template('index.html', status=status)

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f or f.filename=='':
        return "No file", 400
    if not allowed_file(f.filename):
        return "File type not allowed", 400
    fname = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(path)
    return redirect(url_for('index'))

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    # serve uploaded files (heatmaps, images)
    safe = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(safe):
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/models/<path:filename>')
def serve_model(filename):
    # serve model files (download)
    if not os.path.exists(os.path.join(MODEL_FOLDER, filename)):
        abort(404)
    return send_from_directory(MODEL_FOLDER, filename, as_attachment=True)

@app.route('/start-training', methods=['POST'])
def start_training():
    params = request.get_json() or {}
    epochs = int(params.get('epochs', 3))
    lr = float(params.get('lr', 1e-3))
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset')  # user guideline

    # run training in background thread
    def _train():
        status.update({'phase':'preparing','progress':0,'last_message':'Preparing dataset...'})
        try:
            model_name = f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            model_path = os.path.join(MODEL_FOLDER, model_name)
            # train_from_folder will call the status_callback periodically
            train_from_folder(dataset_path, model_path, epochs=epochs, lr=lr, status_callback=lambda s: status.update(s))
            status.update({
                "phase": "finished",
                "progress": 100,
                "last_message": "Training completed",
                "model": model_name
            })
        except Exception as e:
            status.update({'phase':'error','last_message':str(e)})
    t = threading.Thread(target=_train, daemon=True)
    t.start()
    return jsonify({'started':True})

@app.route('/status')
def get_status():
    # return current status
    return jsonify(status)

@app.route('/infer', methods=['POST'])
def infer():
    f = request.files.get('file')
    if not f or f.filename=='':
        return jsonify({'error':'no file'}), 400
    if not allowed_file(f.filename):
        return jsonify({'error':'file type not allowed'}), 400

    fname = secure_filename(f.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(in_path)

    # pick latest model if present
    model_files = sorted(os.listdir(MODEL_FOLDER), reverse=True)
    model_path = os.path.join(MODEL_FOLDER, model_files[0]) if model_files else None

    # prepare / load model (get_model returns a torch model if torch installed, else dummy)
    model = get_model(num_classes=2, pretrained=False)

    # infer_image returns (probs, heatmap_path) where probs is numpy-like [p0,p1]
    try:
        probs, heatmap_path = infer_image(model, in_path, model_path=model_path, return_heatmap=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    heatmap_url = None
    if heatmap_path and os.path.exists(heatmap_path):
        heatmap_url = url_for('serve_upload', filename=os.path.basename(heatmap_path))

    pred = 'tumor' if float(probs[1]) > float(probs[0]) else 'no_tumor'
    return jsonify({'input': in_path, 'prediction': pred, 'probs': [float(p) for p in probs], 'heatmap': heatmap_url})

if __name__ == '__main__':
    # dev server
    app.run(debug=True, host='0.0.0.0', port=5001)
