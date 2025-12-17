import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import logging
import os

# --- Configuration ---
WORD_MODEL_PATH = "Finalbdsl11_words.h5"  # Your existing word model
ALPHA_MODEL_PATH = "my_model_name.h5"        # Your alphabet model (rename if needed)
PORT = 5000

app = Flask(__name__)

# Suppress Logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# --- Load Models ---
models = {}

def load_models():
    # Load Word Model
    if os.path.exists(WORD_MODEL_PATH):
        print(f"Loading Word Model: {WORD_MODEL_PATH}...")
        models['word'] = tf.keras.models.load_model(WORD_MODEL_PATH)
        print("✅ Word Model Loaded.")
    else:
        print(f"❌ Error: {WORD_MODEL_PATH} not found.")

    # Load Alphabet Model
    if os.path.exists(ALPHA_MODEL_PATH):
        print(f"Loading Alphabet Model: {ALPHA_MODEL_PATH}...")
        models['alpha'] = tf.keras.models.load_model(ALPHA_MODEL_PATH)
        print("✅ Alphabet Model Loaded.")
    else:
        print(f"❌ Error: {ALPHA_MODEL_PATH} not found.")

load_models()

@app.route('/predict/<mode>', methods=['POST'])
def predict(mode):
    """
    mode: 'word' or 'alpha'
    """
    try:
        # Check if model exists
        if mode not in models:
            return jsonify({'error': f'Model mode "{mode}" not loaded'}), 400

        data = request.json
        features_list = data.get('features')

        if not features_list or len(features_list) != 63:
            return jsonify({'error': 'Expected 63 features'}), 400

        # Prepare data
        x = np.array(features_list, dtype=np.float32).reshape(1, 63)

        # Predict using the selected model
        probs = models[mode].predict(x, verbose=0)[0]
        return jsonify({'probabilities': probs.tolist()})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Starting Dual-Model Server on port {PORT}...")
    app.run(host='127.0.0.1', port=PORT, debug=False)