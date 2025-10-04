
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# === Paths (usar carpeta instance que ya existe y es escribible) ============
BASE_DIR = Path(__file__).resolve().parent.parent  # <repo>/
INSTANCE_DIR = BASE_DIR / "instance"
MODELS_DIR = INSTANCE_DIR / "models"
DATA_CSV = INSTANCE_DIR / "combined.csv"

INSTANCE_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# === Esquema de features consistent con UI (FIELDS en routes.py) ============
FEATURES = [
    "orbital_period_d",
    "transit_duration_h",
    "transit_depth_ppm",
    "t0_bjd",
    "pl_radius_re",
    "insol_flux_fe",
    "eq_temp_k",
    "ra_deg",
    "dec_deg",
    "st_teff_k",
    "st_logg_cms2",
    "st_radius_rsun",
]

LABEL_COL = "label"  # texto: 'CONFIRMED' | 'FALSE POSITIVE' | 'CANDIDATE' | etc.

# === Utilidades ==============================================================
def _ensure_csv_header():
    """Si no existe el CSV, crear con encabezados (features + label)."""
    if not DATA_CSV.exists():
        df = pd.DataFrame(columns=FEATURES + [LABEL_COL])
        df.to_csv(DATA_CSV, index=False)

def load_dataset(strict: bool = True) -> pd.DataFrame:
    """
    Cargar dataset desde instance/combined.csv.
    strict=True -> lanzar error si no existe o no tiene suficientes filas.
    """
    if not DATA_CSV.exists():
        if strict:
            raise FileNotFoundError("No existe instance/combined.csv. Agrega muestras guardando experimentos o sube tu dataset.")
        _ensure_csv_header()
        return pd.read_csv(DATA_CSV)
    df = pd.read_csv(DATA_CSV)
    # Tipos numéricos (ignorar filas con NaN en features)
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES)
    # Filtrar filas sin etiqueta
    df = df[df[LABEL_COL].notna()].copy()
    if strict and len(df) < 10:
        raise ValueError("El dataset tiene menos de 10 filas útiles. Agrega más ejemplos para entrenar.")
    return df

def features_from_criterios(criterios: list[dict], feature_means: Optional[dict] = None) -> np.ndarray:
    """
    Construye un vector en el ORDEN de FEATURES a partir de 'criterios' (lista con 'key','mag').
    Si algún valor falta, usa 'feature_means' si se proporciona; de lo contrario, 0.0.
    """
    vals = []
    means = feature_means or {}
    lookup = {c["key"]: c.get("mag", "") for c in criterios}
    for key in FEATURES:
        raw = str(lookup.get(key, "")).strip()
        if raw == "":
            fill = means.get(key, 0.0)
            vals.append(float(fill))
        else:
            try:
                vals.append(float(raw))
            except Exception:
                vals.append(means.get(key, 0.0))
    return np.array(vals, dtype=float).reshape(1, -1)

def build_dataset_means(df: pd.DataFrame) -> dict:
    return {c: float(df[c].mean()) for c in FEATURES}

def _encode_labels(y_text: pd.Series) -> Tuple[np.ndarray, list[str], dict]:
    classes = sorted(y_text.dropna().unique().tolist())
    mapping = {c: i for i, c in enumerate(classes)}
    y_idx = y_text.map(mapping).values
    return y_idx, classes, mapping

def _make_pipeline(model_key: str, hp: Dict[str, Any]) -> Pipeline:
    if model_key == "lr":
        clf = LogisticRegression(C=hp.get("C", 1.0), max_iter=hp.get("max_iter", 200), random_state=hp.get("random_state", 42))
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif model_key == "dt":
        clf = DecisionTreeClassifier(max_depth=hp.get("max_depth", 8),
                                     min_samples_split=hp.get("min_samples_split", 10),
                                     min_samples_leaf=hp.get("min_samples_leaf", 4),
                                     random_state=hp.get("random_state", 42))
        pipe = Pipeline([("clf", clf)])
    elif model_key == "rf":
        clf = RandomForestClassifier(n_estimators=hp.get("n_estimators", 800),
                                     max_depth=hp.get("max_depth", 20),
                                     min_samples_split=hp.get("min_samples_split", 10),
                                     random_state=hp.get("random_state", 42),
                                     n_jobs=-1)
        pipe = Pipeline([("clf", clf)])
    else:
        raise ValueError("Modelo no soportado")
    return pipe

def train_and_eval(model_key: str, hp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entrena el modelo elegido y guarda pipeline + clases en instance/models.
    Devuelve métricas y objetos útiles.
    """
    df = load_dataset(strict=True)
    X = df[FEATURES].copy()
    y_text = df[LABEL_COL].astype(str)
    y, classes, mapping = _encode_labels(y_text)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = _make_pipeline(model_key, hp)
    t0 = time.time()
    pipe.fit(Xtr, ytr)
    train_secs = round(time.time() - t0, 3)

    yhat = pipe.predict(Xte)
    metrics = {
        "accuracy": float(accuracy_score(yte, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, yhat)),
        "recall": float(recall_score(yte, yhat, average="macro")),
        "f1": float(f1_score(yte, yhat, average="macro")),
        "time": f"{train_secs}s",
    }

    # Guardar modelo
    joblib.dump(pipe, MODELS_DIR / f"model_{model_key}.pkl")
    with open(MODELS_DIR / "classes.json", "w") as f:
        json.dump(classes, f)

    # medias por si el UI manda campos vacíos
    means = build_dataset_means(df)

    return {
        "pipe": pipe,
        "classes": classes,
        "means": means,
        "metrics": metrics,
    }

def _load_model(model_key: str):
    p = MODELS_DIR / f"model_{model_key}.pkl"
    if not p.exists():
        return None, None
    pipe = joblib.load(p)
    classes = None
    cj = MODELS_DIR / "classes.json"
    if cj.exists():
        with open(cj) as f:
            classes = json.load(f)
    return pipe, classes

def predict_from_criterios(model_key: str, criterios: list[dict]) -> Dict[str, Any]:
    """
    Usa el modelo guardado (o entrena rápido con defaults si no existe dataset).
    Devuelve dict con 'prob', 'clase' y 'classes'.
    """
    pipe, classes = _load_model(model_key)
    if pipe is None or classes is None:
        # Intentar entrenamiento con parámetros por defecto si hay dataset
        try:
            info = train_and_eval(model_key, {})
            pipe, classes = info["pipe"], info["classes"]
            means = info["means"]
        except Exception:
            # sin dataset, devolvemos un resultado suave pero consistente
            # (similar al _fake_pred previo) basado en la cantidad de campos llenos
            filled = sum(1 for r in criterios if str(r.get("mag","")).strip() != "")
            score = min(0.99, 0.55 + 0.03 * filled)
            clase = "Posible exoplaneta" if score >= 0.70 else "No candidato"
            return {"prob": round(score, 3), "clase": clase, "classes": ["NEG","POS"]}
    else:
        # medias para imputar vacíos
        try:
            df = load_dataset(strict=False)
            means = build_dataset_means(df) if len(df) else {}
        except Exception:
            means = {}

    # Construir vector
    X = features_from_criterios(criterios, means)
    # Probabilidad para la mejor clase (tomamos el índice de prob máx)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        prob = float(proba[idx])
        label = classes[idx]
    else:
        idx = int(pipe.predict(X)[0])
        label = classes[idx]
        prob = 1.0
    clase = "Posible exoplaneta" if prob >= 0.70 else "No candidato"
    return {"prob": prob, "clase": clase, "classes": classes}

def append_row_and_retrain(criterios: list[dict], label: str, model_key: str, hp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agrega una fila al CSV (features + label) y reentrena el modelo indicado.
    """
    _ensure_csv_header()
    # cargar para medias y garantizar orden
    try:
        df = load_dataset(strict=False)
    except Exception:
        df = pd.DataFrame(columns=FEATURES+[LABEL_COL])

    means = build_dataset_means(df) if len(df) else {}
    X = features_from_criterios(criterios, means)
    row = {FEATURES[i]: float(X[0, i]) for i in range(len(FEATURES))}
    row[LABEL_COL] = str(label)
    # append
    pd.DataFrame([row]).to_csv(DATA_CSV, mode="a", header=not DATA_CSV.exists() or DATA_CSV.stat().st_size==0, index=False)

    # reentrenar
    info = train_and_eval(model_key, hp or {})
    return {"ok": True, "metrics": info["metrics"]}
