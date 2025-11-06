import os, time, threading, json
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'change-me-local-demo'

status = { 'phase':'idle', 'progress':0, 'last_message':'idle' }

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/')
def index():
    return render_template('index.html', status=status)

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f or f.filename=='':
        return "No file", 400
    fname = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(path)
    return redirect(url_for('index'))

@app.route('/start-training', methods=['POST'])
def start_training():
    params = request.get_json() or {}
    epochs = int(params.get('epochs', 3))
    lr = float(params.get('lr', 1e-3))
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset')
    # run training in thread
    def _train():
        status.update({'phase':'preparing','progress':0,'last_message':'Preparing dataset...'})
        try:
            model_path = os.path.join(MODEL_FOLDER, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            # train_from_folder will update status by itself via callback
            train_from_folder(dataset_path, model_path, epochs=epochs, lr=lr, status_callback=lambda s: status.update(s))
            model_name = os.path.basename(model_path)  # extract only the filename
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
    return jsonify(status)

@app.route('/infer', methods=['POST'])
def infer():
    f = request.files.get('file')
    if not f or f.filename=='':
        return jsonify({'error':'no file'}), 400
    fname = secure_filename(f.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    f.save(in_path)
    # load model if exists else a fresh small model (random weights)
    model_files = sorted(os.listdir(MODEL_FOLDER), reverse=True)
    model_path = os.path.join(MODEL_FOLDER, model_files[0]) if model_files else None
    model = get_model(num_classes=2, pretrained=False)
    probs = infer_image(model, in_path, model_path=model_path)
    return jsonify({'input': in_path, 'prediction': 'tumor' if probs[1]>probs[0] else 'no_tumor', 'probs': [float(p) for p in probs]})

@app.route('/models/<path:filename>')
def serve_model(filename):
    return send_from_directory(MODEL_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
