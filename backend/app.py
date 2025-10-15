"""
Flask server for Industrial Symbiosis Predictor
Loads saved models and provides /predict endpoint
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
from collections import defaultdict, namedtuple
import time
import traceback

app = Flask(__name__)
CORS(app)

# models directory (relative to this file)
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
REG_PATH = os.path.join(MODELS_DIR, "symbiosis_regressor.joblib")
CLF_PATH = os.path.join(MODELS_DIR, "symbiosis_classifier.joblib")
META_PATH = os.path.join(MODELS_DIR, "symbiosis_preproc_meta.joblib")

# Provide the same PreprocessingMeta namedtuple used when the meta was saved.
# This allows pickle/joblib to find the symbol during unpickling when the
# training script was run as __main__.
PreprocessingMeta = namedtuple("PreprocessingMeta",
                              ["num_medians", "cat_modes", "cat_values_map", "selected_features", "categorical_features"])

reg_model = clf_model = preproc_meta = None
load_error = None

def load_models():
    global reg_model, clf_model, preproc_meta, load_error
    load_error = None
    try:
        if os.path.exists(REG_PATH) and os.path.exists(CLF_PATH) and os.path.exists(META_PATH):
            reg_model = joblib.load(REG_PATH)
            clf_model = joblib.load(CLF_PATH)
            # load preproc_meta (may be a namedtuple or plain dict)
            loaded_meta = joblib.load(META_PATH)
            # If joblib returned a dict (or other), try to convert to the expected namedtuple shape
            if isinstance(loaded_meta, dict):
                try:
                    preproc_meta = PreprocessingMeta(
                        num_medians=loaded_meta.get("num_medians", {}),
                        cat_modes=loaded_meta.get("cat_modes", {}),
                        cat_values_map=loaded_meta.get("cat_values_map", {}),
                        selected_features=loaded_meta.get("selected_features", []),
                        categorical_features=loaded_meta.get("categorical_features", [])
                    )
                except Exception:
                    preproc_meta = loaded_meta
            else:
                preproc_meta = loaded_meta
            print(f"[Server] Models loaded from {MODELS_DIR}")
        else:
            missing = [p for p in (REG_PATH, CLF_PATH, META_PATH) if not os.path.exists(p)]
            load_error = f"Model files missing: {missing}"
            print(f"[Server] Model files missing: {missing}")
            reg_model = clf_model = preproc_meta = None
    except Exception as e:
        load_error = f"{e}\n{traceback.format_exc()}"
        print(f"[Server] Error loading models: {e}")
        print(traceback.format_exc())
        reg_model = clf_model = preproc_meta = None

load_models()

# Rate limiting
request_counts = defaultdict(list)
RATE_LIMIT = 100
RATE_WINDOW = 3600

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    request_counts[ip] = [t for t in request_counts[ip] if now - t < RATE_WINDOW]
    if len(request_counts[ip]) >= RATE_LIMIT:
        return False
    request_counts[ip].append(now)
    return True

def preprocess_input_rows(rows: pd.DataFrame, meta) -> np.ndarray:
    df = rows.copy()
    # ensure selected_features exist
    cols = [c for c in meta.selected_features if c in df.columns]
    # create missing columns with NaN
    for c in meta.selected_features:
        if c not in df.columns:
            df[c] = np.nan
    df = df[meta.selected_features].copy()

    for c, med in getattr(meta, "num_medians", {}).items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(med)

    for c, mode in getattr(meta, "cat_modes", {}).items():
        if c in df.columns:
            df[c] = df[c].fillna(mode).astype(str)

    if getattr(meta, "categorical_features", None):
        try:
            from sklearn.preprocessing import OrdinalEncoder
            encoder = OrdinalEncoder(
                categories=[meta.cat_values_map[c] for c in meta.categorical_features],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            df[meta.categorical_features] = encoder.fit_transform(df[meta.categorical_features].astype(str))
        except Exception:
            # if sklearn not available, leave as-is (best-effort)
            pass

    # final numeric conversion (fill remaining NaN to 0 to avoid errors)
    out = df.fillna(0).values.astype(float)
    return out

def predict_from_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    if reg_model is None or clf_model is None or preproc_meta is None:
        raise RuntimeError("Models not loaded")

    row = {}
    for f in preproc_meta.selected_features:
        if f in input_dict and input_dict[f] is not None:
            row[f] = input_dict[f]
        else:
            if f in getattr(preproc_meta, "num_medians", {}):
                row[f] = preproc_meta.num_medians[f]
            elif f in getattr(preproc_meta, "cat_modes", {}):
                row[f] = preproc_meta.cat_modes[f]
            else:
                row[f] = 0

    df_row = pd.DataFrame([row])
    X_ready = preprocess_input_rows(df_row, preproc_meta)

    reg_pred = float(reg_model.predict(X_ready)[0])
    clf_pred = clf_model.predict(X_ready)[0]

    if hasattr(clf_model, "predict_proba"):
        probs = dict(zip(clf_model.classes_.tolist(), clf_model.predict_proba(X_ready)[0].tolist()))
    else:
        probs = {str(clf_pred): 1.0}

    response = {
        "predicted_percentage": reg_pred,
        "predicted_label": str(clf_pred),
        "label_probabilities": probs,
        "used_feature_values": row,
        "model_feature_importances": {
            "regressor_importances_top": sorted(
                list(zip(preproc_meta.selected_features, getattr(reg_model, "feature_importances_", [0]*len(preproc_meta.selected_features)))),
                key=lambda x: x[1],
                reverse=True
            )[:8],
            "classifier_importances_top": sorted(
                list(zip(preproc_meta.selected_features, getattr(clf_model, "feature_importances_", [0]*len(preproc_meta.selected_features)))),
                key=lambda x: x[1],
                reverse=True
            )[:8]
        }
    }
    return response

@app.route('/predict', methods=['POST'])
def predict():
    client_ip = request.remote_addr or 'local'
    if not check_rate_limit(client_ip):
        return jsonify({"error":"Rate limit exceeded"}), 429

    # return clear JSON when models are not ready
    if reg_model is None or clf_model is None or preproc_meta is None:
        return jsonify({"error": "Models not loaded", "load_error": load_error}), 503

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error":"Invalid JSON"}), 400

    # validate minimal fields to avoid easy mistakes
    required = ['recycled_percentage','sustainability_certified','eco_industrial_park_member','recycling_rate']
    for f in required:
        if f not in data:
            return jsonify({"error":f"Missing required field: {f}"}), 400

    try:
        # convert types where useful
        data['recycled_percentage'] = float(data['recycled_percentage'])
        data['recycling_rate'] = float(data['recycling_rate'])
        data['sustainability_certified'] = int(data['sustainability_certified'])
        data['eco_industrial_park_member'] = int(data['eco_industrial_park_member'])
    except Exception as e:
        return jsonify({"error":"Invalid field types"}), 400

    # basic ranges
    if not (0 <= data['recycled_percentage'] <= 100):
        return jsonify({"error":"recycled_percentage must be 0..100"}), 400
    if data['recycling_rate'] < 0:
        return jsonify({"error":"recycling_rate must be non-negative"}), 400

    try:
        res = predict_from_input(data)
        return jsonify(res), 200
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print("[Server] Prediction error:", e)
        return jsonify({"error":"Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health():
    status = {
        "status": "healthy" if reg_model is not None else "unhealthy",
        "models_loaded": reg_model is not None and clf_model is not None
    }
    return jsonify(status), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "service":"Industrial Symbiosis Predictor API",
        "endpoints":{"/predict":"POST","/health":"GET"},
        "version":"1.0"
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)