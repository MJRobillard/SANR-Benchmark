import os
import sys
import json
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add project root to path to import main and src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

# Now we can import from the project
import main
from models import registry

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "version": "1.0.0"})

@app.route('/models', methods=['GET'])
def list_models():
    """List available models in the registry."""
    # Currently registry doesn't expose a list directly if it's a dict, 
    # but we can inspect MODEL_REGISTRY
    try:
        models = list(registry.MODEL_REGISTRY.keys())
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/benchmark/run', methods=['POST'])
def main_endpoint():
    """
    Trigger a benchmark task.
    Body: {
        "task": "A" | "B",
        "model": "model_name"
    }
    """
    data = request.json
    task = data.get("task")
    model_name = data.get("model")
    
    if not task or not model_name:
        return jsonify({"error": "Missing task or model parameter"}), 400
        
    if model_name not in registry.MODEL_REGISTRY:
         return jsonify({"error": f"Model {model_name} not found"}), 404

    try:
        # Reload data to ensure freshness
        train_df, test_df = main.load_data(DATA_DIR)
        
        result = {}
        if task == "A":
            f1 = main.run_task_a(model_name, train_df, test_df, RESULTS_DIR)
            result = {"task": "A", "f1_macro": f1}
        elif task == "B":
            # Task B returns None but writes to file, we should capture the result from file or modify main
            # For now, we'll run it and read the file
            main.run_task_b(model_name, train_df, test_df, RESULTS_DIR)
            result_file = os.path.join(RESULTS_DIR, "task_b_results.json")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    result = json.load(f)
            else:
                result = {"status": "completed", "message": "Check output files"}
        else:
             return jsonify({"error": "Invalid task. Use A or B"}), 400
             
        return jsonify({"status": "success", "result": result})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/results', methods=['GET'])
def list_results():
    """List available result files."""
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
    return jsonify({"files": files})

@app.route('/results/<filename>', methods=['GET'])
def get_result(filename):
    """Get content of a specific result file."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
        
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/data/preview', methods=['GET'])
def preview_data():
    """Get top N rows of the test set."""
    try:
        limit = int(request.args.get('limit', 10))
        test_path = os.path.join(DATA_DIR, "test.csv")
        if not os.path.exists(test_path):
             return jsonify({"error": "Data not found"}), 404
             
        df = pd.read_csv(test_path)
        # Convert to dict records
        data = df.head(limit).fillna("").to_dict(orient='records')
        return jsonify({"count": len(df), "preview": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    """Serve images from the data/images directory."""
    return send_from_directory(IMAGES_DIR, filename)

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)






