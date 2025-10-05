from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

FEATURES = [
    "ra", "dec", "period", "duration", "depth",
    "planet_radius", "insolation", "equilibrium_temp",
    "stellar_teff", "stellar_logg", "stellar_radius",
]

UI_LABELS: Dict[str, str] = {
    "ra": "Ascensión recta (°)",
    "dec": "Declinación (°)",
    "period": "Período orbital (días)",
    "duration": "Duración del tránsito (horas)",
    "depth": "Profundidad del tránsito (ppm)",
    "planet_radius": "Radio del planeta (R⊕)",
    "insolation": "Flujo de insolación (F⊕)",
    "equilibrium_temp": "Temperatura de equilibrio (K)",
    "stellar_teff": "Tₑff estelar (K)",
    "stellar_logg": "log g estelar (cm/s²)",
    "stellar_radius": "Radio estelar (R☉)",
}

BASE_DIR = Path(__file__).resolve().parents[1]
INSTANCE_DIR = BASE_DIR / "instance"
MODELS_DIR = INSTANCE_DIR / "models"
CACHE_DIR = INSTANCE_DIR / "cache"

ARTIFACTS_BASENAME = "bundle"

DEFAULT_NOVICE_MODEL = ("Random Forest", {"n_estimators": 1000, "random_state": 10})

def _fetch_nasa() -> pd.DataFrame:
    cache_file = CACHE_DIR / "nasa_merged.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    urls = {
        "koi": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&select=koi_disposition,ra,dec,koi_period,koi_duration,koi_depth,koi_prad,koi_insol,koi_teq,koi_steff,koi_slogg,koi_srad&format=csv",
        "toi": "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=toi&select=tfopwg_disp,ra,dec,pl_orbper,pl_trandurh,pl_trandep,pl_rade,pl_insol,pl_eqt,st_teff,st_logg,st_rad&format=csv",
        "k2":  "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+disposition,ra,dec,pl_orbper,pl_trandur,(pl_trandep*10000),pl_rade,pl_insol,pl_eqt,st_teff,st_logg,st_rad+from+k2pandc+order+by+hostname+asc,pl_letter+asc,pl_name+asc&format=csv",
    }
    df_koi = pd.read_csv(urls["koi"], low_memory=False).rename(columns={
        "koi_disposition": "disposition",
        "koi_period": "period",
        "koi_duration": "duration",
        "koi_depth": "depth",
        "koi_prad": "planet_radius",
        "koi_insol": "insolation",
        "koi_teq": "equilibrium_temp",
        "koi_steff": "stellar_teff",
        "koi_slogg": "stellar_logg",
        "koi_srad": "stellar_radius",
    })
    df_toi = pd.read_csv(urls["toi"], low_memory=False).rename(columns={
        "tfopwg_disp": "disposition",
        "pl_orbper": "period",
        "pl_trandurh": "duration",
        "pl_trandep": "depth",
        "pl_rade": "planet_radius",
        "pl_insol": "insolation",
        "pl_eqt": "equilibrium_temp",
        "st_teff": "stellar_teff",
        "st_logg": "stellar_logg",
        "st_rad": "stellar_radius",
    })
    df_k2 = pd.read_csv(urls["k2"], low_memory=False).rename(columns={
        "pl_orbper": "period",
        "pl_trandur": "duration",
        "(pl_trandep*10000)": "depth",
        "pl_rade": "planet_radius",
        "pl_insol": "insolation",
        "pl_eqt": "equilibrium_temp",
        "st_teff": "stellar_teff",
        "st_logg": "stellar_logg",
        "st_rad": "stellar_radius",
    })
    df = pd.concat([df_koi, df_toi, df_k2], ignore_index=True, sort=False)
    disposition_mapping = {
        "APC": "CANDIDATE",
        "CP": "CONFIRMED",
        "FA": "FALSE POSITIVE",
        "FP": "FALSE POSITIVE",
        "KP": "CONFIRMED",
        "PC": "CANDIDATE",
        "REFUTED": "FALSE POSITIVE",
    }
    if "disposition" in df.columns:
        df["disposition"] = df["disposition"].replace(disposition_mapping)
    df = df.dropna(subset=FEATURES + ["disposition"]).reset_index(drop=True)
    df = df[df["disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].reset_index(drop=True)
    df = df[FEATURES + ["disposition"]]
    df.to_parquet(cache_file, index=False)
    return df

def _assemble_training_df(user_rows: Optional[pd.DataFrame]):
    base = _fetch_nasa()
    if user_rows is not None and len(user_rows) > 0:
        base = pd.concat([base, user_rows], ignore_index=True)
    le = LabelEncoder()
    base["disposition"] = le.fit_transform(base["disposition"])
    return base, le

def _model_from_choice(name: str, params: Dict[str, Any]):
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=int(params.get("max_depth", 12)),
            min_samples_split=int(params.get("min_samples_split", 10)),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            random_state=int(params.get("random_state", 42)),
        )
    elif name == "Logistic Regression":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 400)),
            random_state=int(params.get("random_state", 42)),
        )
    else:
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 800)),
            max_depth=int(params.get("max_depth", 20)) if params.get("max_depth") is not None else None,
            min_samples_split=int(params.get("min_samples_split", 10)),
            random_state=int(params.get("random_state", 42)),
        )

def _key_for(name: str, params: Dict[str, Any], use_user_data: bool, user_id: Optional[int]) -> str:
    param_items = ",".join(f"{k}={params[k]}" for k in sorted(params.keys()))
    raw = f"{name}|{param_items}|useraug={use_user_data}|uid={user_id}"
    import hashlib
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def _paths_for(user_id: Optional[int], key: str):
    root = MODELS_DIR / (str(user_id) if user_id is not None else "shared") / key
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "scaler": root / "scaler.joblib",
        "model": root / "model.joblib",
        "label_encoder": root / "label_encoder.joblib",
        "meta": root / "meta.json",
    }

def artifacts_exist(user_id: Optional[int], key: str) -> bool:
    p = _paths_for(user_id, key)
    return p["scaler"].exists() and p["model"].exists() and p["label_encoder"].exists() and p["meta"].exists()

def train_and_persist(model_name: str, params: Dict[str, Any], use_user_data: bool, user_id: Optional[int], user_rows_df: Optional[pd.DataFrame]=None):
    key = _key_for(model_name, params, use_user_data, user_id)
    paths = _paths_for(user_id, key)

    df, le = _assemble_training_df(user_rows_df if use_user_data else None)
    X = df[FEATURES].values
    y = df["disposition"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = _model_from_choice(model_name, params)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "model_name": model_name,
        "params": params,
        "use_user_data": use_user_data,
        "user_id": user_id,
        "key": key,
    }
    import joblib, json
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(model, paths["model"])
    joblib.dump(le, paths["label_encoder"])
    paths["meta"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

def get_model_bundle(model_name: str, params: Dict[str, Any], use_user_data: bool, user_id: Optional[int], user_rows_df: Optional[pd.DataFrame]=None):
    key = _key_for(model_name, params, use_user_data, user_id)
    paths = _paths_for(user_id, key)
    try:
        if not artifacts_exist(user_id, key):
            metrics = train_and_persist(model_name, params, use_user_data, user_id, user_rows_df=user_rows_df)
        else:
            import json
            metrics = json.loads(paths["meta"].read_text(encoding="utf-8"))
        import joblib
        bundle = {
            "scaler": joblib.load(paths["scaler"]),
            "model": joblib.load(paths["model"]),
            "label_encoder": joblib.load(paths["label_encoder"]),
            "meta": metrics,
        }
        return bundle, {"trained": True, **metrics}
    except Exception as e:
        return {}, {"trained": False, "error": str(e)}

def predict_one(bundle: Dict[str, Any], values: list[float]):
    scaler = bundle["scaler"]
    model = bundle["model"]
    le = bundle["label_encoder"]

    import numpy as np
    x = np.array(values, dtype=float).reshape(1, -1)
    xs = scaler.transform(x)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(xs)[0]
    else:
        proba = np.array([0.5, 0.5])

    classes = list(le.classes_)  # ["CONFIRMED", "FALSE POSITIVE"]
    idx = int(np.argmax(proba))
    pred_str = classes[idx]

    name_map = {"CONFIRMED": "Confirmado", "FALSE POSITIVE": "Falso positivo"}
    pred_label = name_map.get(pred_str, pred_str)

    proba_map = {
        "Confirmado": float(proba[classes.index("CONFIRMED")]) if "CONFIRMED" in classes else float(proba[0]),
        "Falso positivo": float(proba[classes.index("FALSE POSITIVE")]) if "FALSE POSITIVE" in classes else float(proba[1]),
    }
    return pred_label, proba_map

DEFAULT_NOVICE_MODEL = ("Random Forest", {"n_estimators": 1000, "random_state": 10})
