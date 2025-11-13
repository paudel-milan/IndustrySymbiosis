"""
train_symbiosis_model.py

A slightly-unconventional but clear pipeline that:
- finds which features matter for two targets,
- trains a RandomForest regressor + classifier on those important features,
- and saves models + a resilient predictor function that handles missing fields.

Author: ChatGPT (GPT-5 Thinking mini) — tailored for Nithesh's style preferences
"""

import json
from collections import defaultdict, namedtuple
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
import joblib
import os

models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

# ---------- Helper data structure (a little different from typical pipelines) ----------
PreprocessingMeta = namedtuple("PreprocessingMeta",
                               ["num_medians", "cat_modes", "cat_values_map", "selected_features", "categorical_features"])

# ---------- 1) Load data ----------
def load_csv(path: str) -> pd.DataFrame:
    """Load CSV into dataframe safely. Try given path, then backend/data/ subfolder."""
    # If path is already absolute or exists as given, use it. Otherwise, try backend/data/<path>
    if os.path.isabs(path) and os.path.exists(path):
        csv_path = path
    elif os.path.exists(path):
        csv_path = path
    else:
        candidate = os.path.join(os.path.dirname(__file__), "data", path)
        if os.path.exists(candidate):
            csv_path = candidate
        else:
            raise FileNotFoundError(f"CSV not found. Tried: '{path}' and '{candidate}'")
    df = pd.read_csv(csv_path)
    print(f"[load_csv] loaded {csv_path} with shape {df.shape}")
    return df

# ---------- 2) Find feature importances for both targets and choose top K ----------
def compute_combined_feature_ranking(df: pd.DataFrame,
                                     reg_target: str,
                                     clf_target: str,
                                     top_k: int = 8,
                                     rf_estimators: int = 200,
                                     random_state: int = 42) -> List[str]:
    """
    Fit two RandomForest models (regressor & classifier) on quickly-encoded features,
    normalize importances and combine them to pick the top_k feature names.
    """
    X = df.drop(columns=[reg_target, clf_target])
    y_reg = df[reg_target].values
    y_clf = df[clf_target].values

    # quick encoding for categorical columns for tree models (keeps pipeline simple)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # temporary ordinal encode
    if cat_cols:
        oe = OrdinalEncoder()
        X_enc = X.copy()
        X_enc[cat_cols] = oe.fit_transform(X[cat_cols].astype(str))
    else:
        X_enc = X.copy()

    # Fit
    rf_reg = RandomForestRegressor(n_estimators=rf_estimators, random_state=random_state)
    rf_reg.fit(X_enc, y_reg)
    rf_clf = RandomForestClassifier(n_estimators=rf_estimators, random_state=random_state)
    rf_clf.fit(X_enc, y_clf)

    # combine normalized importances
    ir = rf_reg.feature_importances_.astype(float)
    ic = rf_clf.feature_importances_.astype(float)

    def norm(a):
        s = a.sum()
        return a / s if s > 0 else a

    combined = norm(ir) + norm(ic)
    feature_names = X_enc.columns.tolist()
    combined_pairs = sorted(zip(feature_names, combined), key=lambda x: x[1], reverse=True)
    top_features = [f for f, _ in combined_pairs[:top_k]]

    print(f"[compute_combined_feature_ranking] top {top_k} features: {top_features}")
    return top_features

# ---------- 3) Build preprocessing metadata for selected features ----------
def build_preprocessing_metadata(df: pd.DataFrame, selected_features: List[str]) -> PreprocessingMeta:
    """Compute numeric medians and categorical modes for robust missing-value handling,
       also produce the list of categories per categorical feature to use a stable ordinal encoding later."""
    subset = df[selected_features]
    num_cols = subset.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = subset.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    num_medians = subset[num_cols].median().to_dict()
    cat_modes = subset[cat_cols].mode().iloc[0].to_dict() if cat_cols else {}
    # map categories -> list for each cat col
    cat_values_map = {c: list(subset[c].astype(str).unique()) for c in cat_cols}

    meta = PreprocessingMeta(num_medians=num_medians,
                             cat_modes=cat_modes,
                             cat_values_map=cat_values_map,
                             selected_features=selected_features,
                             categorical_features=cat_cols)
    print(f"[build_preprocessing_metadata] numeric medians for {len(num_medians)} cols, categorical modes for {len(cat_modes)} cols")
    return meta

# ---------- 4) Preprocess function (makes model input stable) ----------
def preprocess_input_rows(rows: pd.DataFrame, meta: PreprocessingMeta) -> Tuple[np.ndarray, OrdinalEncoder]:
    """Fill missing, enforce order of features, and ordinal-encode categorical columns.
       Returns X_ready and the encoder used (so it may be saved)."""
    df = rows.copy()
    df = df[meta.selected_features].copy()

    # fill numeric medians
    for c, med in meta.num_medians.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(med)

    # fill categorical modes
    for c, mode in meta.cat_modes.items():
        if c in df.columns:
            df[c] = df[c].fillna(mode).astype(str)

    # ordinal encode categorical features using the stored category lists for stability
    encoder = None
    if meta.categorical_features:
        encoder = OrdinalEncoder(categories=[meta.cat_values_map[c] for c in meta.categorical_features], handle_unknown='use_encoded_value', unknown_value=-1)
        df[meta.categorical_features] = encoder.fit_transform(df[meta.categorical_features].astype(str))

    return df.values.astype(float), encoder

# ---------- 5) Train final models on selected features and save artifacts ----------
def train_and_save(df: pd.DataFrame, selected_features: List[str], reg_target: str, clf_target: str, out_prefix: str = "symbiosis"):
    """
    Train RandomForest regressor & classifier on selected_features and save:
      - {out_prefix}_regressor.joblib
      - {out_prefix}_classifier.joblib
      - {out_prefix}_preproc_meta.joblib
    """
    meta = build_preprocessing_metadata(df, selected_features)
    X_df = df[selected_features].copy()
    # fill on training set using same approach
    for c, med in meta.num_medians.items():
        if c in X_df.columns:
            X_df[c] = pd.to_numeric(X_df[c], errors='coerce').fillna(med)
    for c, mode in meta.cat_modes.items():
        if c in X_df.columns:
            X_df[c] = X_df[c].fillna(mode).astype(str)

    # encode categories
    if meta.categorical_features:
        oe = OrdinalEncoder(categories=[meta.cat_values_map[c] for c in meta.categorical_features], handle_unknown='use_encoded_value', unknown_value=-1)
        X_df[meta.categorical_features] = oe.fit_transform(X_df[meta.categorical_features].astype(str))
    else:
        oe = None

    X = X_df.values.astype(float)
    y_reg = df[reg_target].values
    y_clf = df[clf_target].values

    # train-test split to show simple metrics
    X_tr, X_val, ytr_reg, yval_reg, ytr_clf, yval_clf = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=300, random_state=42)
    clf = RandomForestClassifier(n_estimators=300, random_state=42)

    reg.fit(X_tr, ytr_reg)
    clf.fit(X_tr, ytr_clf)
    
    
    # ---------- DASHBOARD ARTIFACTS ----------

# 1) Save model feature importances (model-level)
    global_importances = dict(zip(selected_features, reg.feature_importances_.tolist()))
    with open(os.path.join(models_dir, f"{out_prefix}_global_importances.json"), "w") as fh:
        json.dump(global_importances, fh, indent=2)

    # 2) Save per-feature histograms (for simple distributions on frontend)
    # We'll compute histogram bins + counts for numeric features in selected_features
    feature_hist = {}
    for idx, feat in enumerate(selected_features):
        col_vals = X_df[feat].dropna().astype(float).values
        if col_vals.size > 0:
            counts, bins = np.histogram(col_vals, bins=10)
            feature_hist[feat] = {
                "bins": bins.tolist(),    # length 11
                "counts": counts.tolist() # length 10
            }
        else:
            feature_hist[feat] = {"bins": [], "counts": []}

    with open(os.path.join(models_dir, f"{out_prefix}_feature_histograms.json"), "w") as fh:
        json.dump(feature_hist, fh, indent=2)

    # 3) Compute & save global mean |SHAP| for each feature (uses background X_tr sample)
    try:
        import shap
        # use a small sample for SHAP global estimation (you already saved shap_background)
        bg_sample_size = min(200, X_tr.shape[0])
        sample_idx = np.random.choice(X_tr.shape[0], bg_sample_size, replace=False)
        sample_X = X_tr[sample_idx]

        explainer_local = shap.TreeExplainer(reg)
        shap_vals = explainer_local.shap_values(sample_X)
        # shap_vals may be list or array; handle regressor common case
        shap_arr = np.array(shap_vals[0]) if isinstance(shap_vals, list) else np.array(shap_vals)
        # shap_arr shape: (n_samples, n_features)
        mean_abs_shap = np.mean(np.abs(shap_arr), axis=0).tolist()
        shap_global = dict(zip(selected_features, mean_abs_shap))
        with open(os.path.join(models_dir, f"{out_prefix}_shap_global_abs.json"), "w") as fh:
            json.dump(shap_global, fh, indent=2)
        print(f"[train_and_save] saved shap_global_abs to {models_dir}")
    except Exception as e:
        print("[train_and_save] Could not compute/save SHAP global:", e)
    # ---------- END DASHBOARD ARTIFACTS ----------

    
    bg_sample_size = min(100, X_tr.shape[0])
    bg_indices = np.random.choice(X_tr.shape[0], bg_sample_size, replace=False)
    shap_background = X_tr[bg_indices]

    # save background
    shap_bg_path = os.path.join(models_dir, f"{out_prefix}_shap_background.npy")
    np.save(shap_bg_path, shap_background)
    print(f"[train_and_save] saved shap background to {shap_bg_path}")

    # metrics
    reg_pred = reg.predict(X_val)
    clf_pred = clf.predict(X_val)
    print(f"[train_and_save] reg R2: {r2_score(yval_reg, reg_pred):.4f}, MAE: {mean_absolute_error(yval_reg, reg_pred):.4f}")
    print(f"[train_and_save] clf accuracy: {accuracy_score(yval_clf, clf_pred):.4f}")

    # persistence
    joblib.dump(reg, os.path.join(models_dir, f"{out_prefix}_regressor.joblib"))
    joblib.dump(clf, os.path.join(models_dir, f"{out_prefix}_classifier.joblib"))
    joblib.dump(meta, os.path.join(models_dir, f"{out_prefix}_preproc_meta.joblib"))
    with open(os.path.join(models_dir, f"{out_prefix}_selected_features.json"), "w") as fh:
        json.dump(selected_features, fh, indent=2)
    print(f"[train_and_save] saved models to {models_dir}")
    return reg, clf, meta, oe

# ---------- 6) Predict function you can call from a UI (handles partial input) ----------
def load_models_and_meta(prefix: str = "symbiosis"):
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    reg_path = os.path.join(models_dir, f"{prefix}_regressor.joblib")
    clf_path = os.path.join(models_dir, f"{prefix}_classifier.joblib")
    meta_path = os.path.join(models_dir, f"{prefix}_preproc_meta.joblib")
    if not (os.path.exists(reg_path) and os.path.exists(clf_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Model files missing in {models_dir}: {reg_path}, {clf_path}, {meta_path}")
    reg = joblib.load(reg_path)
    clf = joblib.load(clf_path)
    meta = joblib.load(meta_path)
    return reg, clf, meta

def predict_from_partial_input(input_dict: Dict[str, Any], meta: PreprocessingMeta, reg_model, clf_model) -> Dict[str, Any]:
    """
    Accepts partial dictionary of inputs (some fields may be missing).
    Fills missing using meta, runs models, and returns:
      - predicted_percentage,
      - predicted_label,
      - label_probabilities (per class),
      - used_feature_values (the actual values used after filling),
      - simple feature importance slice (local: model.feature_importances_ for selected_features)
    """
    # build a single-row dataframe using selected features ordering
    row = {}
    for f in meta.selected_features:
        if f in input_dict and input_dict[f] is not None:
            row[f] = input_dict[f]
        else:
            # use median or mode from metadata
            if f in meta.num_medians:
                row[f] = meta.num_medians[f]
            elif f in meta.cat_modes:
                row[f] = meta.cat_modes[f]
            else:
                row[f] = 0  # fallback
    df_row = pd.DataFrame([row])
    X_ready, encoder = preprocess_input_rows(df_row, meta)  # uses same meta-based filling
    # if there is a categorical encoder, make sure to map consistent unknown -> -1 (handled in preprocess_input_rows)
    reg_pred = float(reg_model.predict(X_ready)[0])
    clf_pred = clf_model.predict(X_ready)[0]
    if hasattr(clf_model, "predict_proba"):
        probs = dict(zip(clf_model.classes_.tolist(), clf_model.predict_proba(X_ready)[0].tolist()))
    else:
        probs = {str(clf_pred): 1.0}

    # return helpful context that a UI can render
    resp = {
        "predicted_percentage": reg_pred,
        "predicted_label": str(clf_pred),
        "label_probabilities": probs,
        "used_feature_values": row,
        "model_feature_importances": {
            "regressor_importances_top": sorted(list(zip(meta.selected_features, reg_model.feature_importances_)), key=lambda x: x[1], reverse=True)[:8],
            "classifier_importances_top": sorted(list(zip(meta.selected_features, clf_model.feature_importances_)), key=lambda x: x[1], reverse=True)[:8]
        }
    }
    return resp

# ---------- 7) Main executable flow ----------
FORCED_FEATURES = [
    "CO2_emissions_tons",
    "waste_generated_tons",
    "byproduct_reuse_index",
    "energy_recovery_rate",
]

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "realistic_industrial_symbiosis_dataset.csv")
    reg_target = "Symbiosis_Score_Percentage"
    clf_target = "Symbiosis_Potential"

    df = load_csv(csv_path)

    # pick top features combining importance for both targets
    top_features = compute_combined_feature_ranking(df, reg_target, clf_target, top_k=8)

    # Ensure the optional/advanced features are included so user-provided values affect predictions.
    # If a forced feature is missing from the dataset we warn and skip it.
    final_features = list(top_features)  # copy
    for f in FORCED_FEATURES:
        if f not in final_features:
            if f in df.columns:
                final_features.append(f)
            else:
                print(f"[train] Warning: forced feature '{f}' not found in dataset; skipping.")

    print(f"[train] final selected features (count={len(final_features)}): {final_features}")

    # train and save (saves into backend/models/)
    reg, clf, meta, _ = train_and_save(df, final_features, reg_target, clf_target, out_prefix="symbiosis")

    # quick demo: how to call the predictor with partial inputs
    demo_input = {
        # provide some values only, leave others out
        "recycled_percentage": 45.0,
        "sustainability_certified": 1,
        # eco_industrial_park_member intentionally omitted to show fallback
        "recycling_rate": 12.0,
        # optional advanced fields demo
        "CO2_emissions_tons": 1000,
        "waste_generated_tons": 20,
        "byproduct_reuse_index": 0.5,
        "energy_recovery_rate": 0.1,
    }
    resp = predict_from_partial_input(demo_input, meta, reg, clf)
    print("[main] demo prediction result:", resp)

if __name__ == "__main__":
    main()
