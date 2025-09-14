import os
import uuid
import threading
import queue
import shutil
import zipfile
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file, url_for, Response, send_from_directory

from analysis_logic import setup_cfg, run_inference_on_folder, DETECTRON2_AVAILABLE, SKIMAGE_AVAILABLE, SCIPY_AVAILABLE


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['SECRET_KEY'] = 'super-secret-key-change-me'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

PREDICTOR = None
DATASET_METADATA = None
analysis_jobs = {}

#load model here
def initialize_model():
    global PREDICTOR, DATASET_METADATA
    if not DETECTRON2_AVAILABLE:
        print("CRITICAL ERROR: Detectron2 is not available.")
        return

    model_weights_path = os.path.join("wound_model", "model_final.pth")
    if not os.path.exists(model_weights_path):
        print(f"CRITICAL ERROR: Model weights not found at '{model_weights_path}'")
        return

    print("Initializing Detectron2 model... (This may take a moment)")
    try:
        from detectron2.engine import DefaultPredictor
        from detectron2.data import MetadataCatalog

        cfg = setup_cfg(model_weights_path, score_thresh=0.5)
        PREDICTOR = DefaultPredictor(cfg)

        metadata_name = "wound_dataset_metadata_flask"
        DATASET_METADATA = MetadataCatalog.get(metadata_name)
        DATASET_METADATA.set(thing_classes=["wound"])

        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not initialize Detectron2 predictor: {e}")
        PREDICTOR = None

#must ensure all models are loaded before init call
with app.app_context():
    initialize_model()

#the main run
def analysis_worker(job_id, image_folder, output_folder, pixels_per_cm, score_thresh):
    log_queue = queue.Queue()
    progress_queue = queue.Queue()
    analysis_jobs[job_id]['log_queue'] = log_queue
    analysis_jobs[job_id]['progress_queue'] = progress_queue

    try:
        from detectron2.engine import DefaultPredictor
        model_weights_path = os.path.join("wound_model", "model_final.pth")
        cfg = setup_cfg(model_weights_path, score_thresh, log_queue=log_queue)
        job_predictor = DefaultPredictor(cfg)

        run_inference_on_folder(
            image_folder=image_folder, output_folder=output_folder,
            predictor=job_predictor, dataset_metadata=DATASET_METADATA,
            pixels_per_cm=pixels_per_cm, log_queue=log_queue, progress_queue=progress_queue
        )
        analysis_jobs[job_id]['status'] = 'completed'
        log_queue.put('---JOB-COMPLETE---')
    except Exception as e:
        analysis_jobs[job_id]['status'] = 'failed'
        log_queue.put(f"ERROR: Analysis failed: {e}")
        log_queue.put('---JOB-FAILED---')

#standard flask routes
@app.route('/results_data/<path:filename>')
def results_data(filename):
    """Serves generated result files from the 'results' directory."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/')
def index():
    context = {
        'detectron2_ok': DETECTRON2_AVAILABLE and PREDICTOR is not None,
        'scipy_ok': SCIPY_AVAILABLE,
        'skimage_ok': SKIMAGE_AVAILABLE
    }
    return render_template('index.html', **context)

@app.route('/analyze', methods=['POST'])
def analyze():
    if not DETECTRON2_AVAILABLE or PREDICTOR is None:
        return jsonify({'error': 'Model is not available.'}), 500

    job_id = str(uuid.uuid4())
    job_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    job_output_folder = os.path.join(app.config['RESULTS_FOLDER'], job_id)
    os.makedirs(job_upload_folder, exist_ok=True)
    os.makedirs(job_output_folder, exist_ok=True)

    uploaded_files = request.files.getlist('images')
    if not uploaded_files or uploaded_files[0].filename == '':
        return jsonify({'error': 'No files selected.'}), 400

    for file in uploaded_files:
        file.save(os.path.join(job_upload_folder, file.filename))

    score_thresh = float(request.form.get('score_thresh', 0.5))
    pixels_per_cm = float(px_str) if (px_str := request.form.get('pixels_per_cm')) else None

    analysis_jobs[job_id] = {'status': 'running'}
    thread = threading.Thread(target=analysis_worker, args=(job_id, job_upload_folder, job_output_folder, pixels_per_cm, score_thresh))
    thread.daemon = True
    thread.start()
    return jsonify({'job_id': job_id})

@app.route('/stream/<job_id>')
def stream(job_id):
    def event_stream():
        log_q = analysis_jobs.get(job_id, {}).get('log_queue')
        prog_q = analysis_jobs.get(job_id, {}).get('progress_queue')
        if not log_q or not prog_q:
            yield f"data: ERROR: Job ID {job_id} not found on server.\n\n"
            return
        while True:
            try:
                log_message = log_q.get(timeout=0.1)
                if '---JOB-COMPLETE---' in log_message or '---JOB-FAILED---' in log_message:
                    yield f"event: status\ndata: {'completed' if 'COMPLETE' in log_message else 'failed'}\n\n"
                    break
                yield f"event: log\ndata: {log_message}\n\n"
            except queue.Empty:
                pass
            try:
                progress_message = prog_q.get_nowait()
                yield f"event: progress\ndata: {progress_message}\n\n"
            except queue.Empty:
                pass
    return Response(event_stream(), content_type='text/event-stream')

@app.route('/results/<job_id>')
def results(job_id):
    output_folder = os.path.join(app.config['RESULTS_FOLDER'], job_id)
    if not os.path.exists(output_folder): return "Results not found.", 404

    results_data = {}

    for root, _, files in os.walk(output_folder):
        category_path = os.path.relpath(root, output_folder)

        category = category_path if category_path != '.' else 'main'
        
        if category not in results_data: results_data[category] = []
            
        for name in files:

            file_path_in_results = os.path.join(category, name) if category != 'main' else name

            file_url = url_for('results_data', filename=f'{job_id}/{file_path_in_results}')
            results_data[category].append({'name': name, 'url': file_url})


    if 'main' in results_data:

        csv_file_data = next((file for file in results_data['main'] if file['name'] == 'wound_metrics.csv'), None)
        
        if csv_file_data:

            results_data['wound_metrics.csv'] = [csv_file_data]

            results_data['main'] = [file for file in results_data['main'] if file['name'] != 'wound_metrics.csv']


    for category in results_data:
        results_data[category].sort(key=lambda x: x['name'])

    return render_template('results.html', job_id=job_id, results=results_data)

@app.route('/download/<job_id>')
def download_results(job_id):
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
    if not os.path.isdir(result_dir): return "Job not found.", 404
    
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(result_dir):
            for file in files:
                arcname = os.path.relpath(os.path.join(root, file), result_dir)
                zf.write(os.path.join(root, file), arcname)
    memory_file.seek(0)
    
    #cleanup server file
    shutil.rmtree(result_dir)
    shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], job_id))
    if job_id in analysis_jobs: del analysis_jobs[job_id]

    return send_file(memory_file, download_name=f'wound_analysis_{job_id}.zip', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
