"""
Flask server for Industrial Symbiosis Predictor
Loads saved models and provides /predict + /dashboard_data endpoints
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
import shap   # SHAP added
import json    # Needed for dashboard artifacts

app = Flask(__name__)
CORS(app)

# models directory (relative to this file)
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
REG_PATH = os.path.join(MODELS_DIR, "symbiosis_regressor.joblib")
CLF_PATH = os.path.join(MODELS_DIR, "symbiosis_classifier.joblib")
META_PATH = os.path.join(MODELS_DIR, "symbiosis_preproc_meta.joblib")

# same namedtuple used in training
PreprocessingMeta = namedtuple(
    "PreprocessingMeta",
    ["num_medians", "cat_modes", "cat_values_map", "selected_features", "categorical_features"]
)

reg_model = clf_model = preproc_meta = None
load_error = None
explainer = None

# >>> DASHBOARD ADDITIONS START
global_importances = {}
feature_histograms = {}
shap_global_abs = {}
# >>> DASHBOARD ADDITIONS END


def load_models():
    global reg_model, clf_model, preproc_meta, load_error, explainer
    global global_importances, feature_histograms, shap_global_abs

    load_error = None
    try:
        if os.path.exists(REG_PATH) and os.path.exists(CLF_PATH) and os.path.exists(META_PATH):
            reg_model = joblib.load(REG_PATH)
            clf_model = joblib.load(CLF_PATH)

            # load metadata
            loaded_meta = joblib.load(META_PATH)
            if isinstance(loaded_meta, dict):
                preproc_meta = PreprocessingMeta(
                    num_medians=loaded_meta.get("num_medians", {}),
                    cat_modes=loaded_meta.get("cat_modes", {}),
                    cat_values_map=loaded_meta.get("cat_values_map", {}),
                    selected_features=loaded_meta.get("selected_features", []),
                    categorical_features=loaded_meta.get("categorical_features", [])
                )
            else:
                preproc_meta = loaded_meta

            print(f"[Server] Models loaded from {MODELS_DIR}")

            # ---------------- SHAP LOADING -------------------
            shap_bg_path = os.path.join(MODELS_DIR, "symbiosis_shap_background.npy")
            if os.path.exists(shap_bg_path):
                try:
                    bg_data = np.load(shap_bg_path)
                    explainer = shap.TreeExplainer(reg_model, data=bg_data)
                    print(f"[Server] SHAP explainer initialized with {shap_bg_path}")
                except Exception as e:
                    print("[Server] SHAP initialization failed:", e)
                    explainer = None
            else:
                print("[Server] No SHAP background file found. SHAP disabled.")
                explainer = None

            # >>> DASHBOARD ADDITIONS START
            # Load global importances
            gi_path = os.path.join(MODELS_DIR, "symbiosis_global_importances.json")
            if os.path.exists(gi_path):
                with open(gi_path, "r") as fh:
                    global_importances = json.load(fh)

            # Load histograms
            fh_path = os.path.join(MODELS_DIR, "symbiosis_feature_histograms.json")
            if os.path.exists(fh_path):
                with open(fh_path, "r") as fh:
                    feature_histograms = json.load(fh)

            # Load global SHAP
            sg_path = os.path.join(MODELS_DIR, "symbiosis_shap_global_abs.json")
            if os.path.exists(sg_path):
                with open(sg_path, "r") as fh:
                    shap_global_abs = json.load(fh)

            print("[Server] Dashboard artifacts loaded")
            # >>> DASHBOARD ADDITIONS END

        else:
            missing = [p for p in (REG_PATH, CLF_PATH, META_PATH) if not os.path.exists(p)]
            load_error = f"Model files missing: {missing}"
            print(f"[Server] Model files missing: {missing}")
            reg_model = clf_model = preproc_meta = None

    except Exception as e:
        load_error = f"{e}\n{traceback.format_exc()}"
        print(f"[Server] Error loading models:", e)
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

    for c in meta.selected_features:
        if c not in df.columns:
            df[c] = np.nan

    df = df[meta.selected_features].copy()

    for c, med in getattr(meta, "num_medians", {}).items():
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(med)

    for c, mode in getattr(meta, "cat_modes", {}).items():
        df[c] = df[c].fillna(mode).astype(str)

    if getattr(meta, "categorical_features", None):
        try:
            from sklearn.preprocessing import OrdinalEncoder
            enc = OrdinalEncoder(
                categories=[meta.cat_values_map[c] for c in meta.categorical_features],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            df[meta.categorical_features] = enc.fit_transform(df[meta.categorical_features].astype(str))
        except Exception:
            pass

    return df.fillna(0).values.astype(float)



# ---------------------- Prediction + SHAP ----------------------
def predict_from_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    if reg_model is None:
        raise RuntimeError("Models not loaded")

    # fill missing fields
    row = {}
    for f in preproc_meta.selected_features:
        if f in input_dict and input_dict[f] is not None:
            row[f] = input_dict[f]
        else:
            if f in preproc_meta.num_medians:
                row[f] = preproc_meta.num_medians[f]
            elif f in preproc_meta.cat_modes:
                row[f] = preproc_meta.cat_modes[f]
            else:
                row[f] = 0

    df_row = pd.DataFrame([row])
    X_ready = preprocess_input_rows(df_row, preproc_meta)

    # Predictions
    reg_pred = float(reg_model.predict(X_ready)[0])
    clf_pred = clf_model.predict(X_ready)[0]
    probs = dict(zip(clf_model.classes_.tolist(), clf_model.predict_proba(X_ready)[0].tolist()))

    response = {
        "predicted_percentage": reg_pred,
        "predicted_label": str(clf_pred),
        "label_probabilities": probs,
        "used_feature_values": row,
    }

    # ---------------------- SHAP EXPLANATION ----------------------
    try:
        global explainer
        if explainer is not None:
            shap_vals = explainer.shap_values(X_ready)

            if isinstance(shap_vals, list):
                shap_arr = np.array(shap_vals[0])[0]
            else:
                shap_arr = shap_vals[0]

            shap_values_map = {feat: float(val)
                for feat, val in zip(preproc_meta.selected_features, shap_arr)}

            shap_top = sorted(
                shap_values_map.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            response["shap_values"] = shap_values_map
            response["shap_top_features"] = [
                {"feature": f, "contribution": v} for f, v in shap_top
            ]
        else:
            response["shap_values"] = None
            response["shap_top_features"] = None
    except Exception as e:
        print("[Server] SHAP computation failed:", e)

    return response
# --------------------------------------------------------------



# ---------------------- Predict endpoint ----------------------
@app.route('/predict', methods=['POST'])
def predict():
    client_ip = request.remote_addr or "local"
    if not check_rate_limit(client_ip):
        return jsonify({"error": "Rate limit exceeded"}), 429

    if reg_model is None:
        return jsonify({"error": "Models not loaded", "load_error": load_error}), 503

    try:
        data = request.get_json(force=True)
    except:
        return jsonify({"error": "Invalid JSON"}), 400

    required = ["recycled_percentage", "sustainability_certified",
                "eco_industrial_park_member", "recycling_rate"]

    for f in required:
        if f not in data:
            return jsonify({"error": f"Missing required field: {f}"}), 400

    try:
        data["recycled_percentage"] = float(data["recycled_percentage"])
        data["recycling_rate"] = float(data["recycling_rate"])
        data["sustainability_certified"] = int(data["sustainability_certified"])
        data["eco_industrial_park_member"] = int(data["eco_industrial_park_member"])
    except:
        return jsonify({"error": "Invalid field types"}), 400

    if not (0 <= data["recycled_percentage"] <= 100):
        return jsonify({"error": "recycled_percentage must be 0..100"}), 400

    if data["recycling_rate"] < 0:
        return jsonify({"error": "recycling_rate must be non-negative"}), 400

    try:
        res = predict_from_input(data)
        return jsonify(res), 200
    except Exception as e:
        print("[Server] Prediction error:", e)
        return jsonify({"error": "Internal server error"}), 500



# ---------------------- Dashboard endpoint ----------------------
@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    """
    Returns:
      selected_features
      dataset_medians
      global_importances
      shap_global_abs
      feature_histograms
    """
    if preproc_meta is None:
        return jsonify({"error": "Models not loaded"}), 503

    resp = {
        "selected_features": preproc_meta.selected_features,
        "dataset_medians": preproc_meta.num_medians,
        "global_importances": global_importances,
        "shap_global_abs": shap_global_abs,
        "feature_histograms": feature_histograms,
    }
    return jsonify(resp), 200



@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy" if reg_model is not None else "unhealthy",
        "models_loaded": reg_model is not None
    }), 200


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "service": "Industrial Symbiosis Predictor API",
        "endpoints": {
            "/predict": "POST",
            "/dashboard_data": "GET",
            "/health": "GET"
        },
        "version": "1.2"
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
