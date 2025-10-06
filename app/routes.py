from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_required, current_user
import json
import pandas as pd 
from datetime import datetime

from .ml_service import (
    FEATURES, UI_LABELS, get_model_bundle, predict_one, DEFAULT_NOVICE_MODEL, train_and_persist
)
from .extensions import db
from .models import UserSample

bp = Blueprint("main", __name__)

def _get_user_rows_df():
    if not current_user.is_authenticated:
        return pd.DataFrame(columns=FEATURES + ["disposition"])
    rows = UserSample.query.filter_by(user_id=current_user.id).order_by(UserSample.created_at.desc()).all()
    data = [
        {**{f: getattr(r, f) for f in FEATURES}, "disposition": r.disposition, "created_at": r.created_at}
        for r in rows
    ]
    return pd.DataFrame(data) if data else pd.DataFrame(columns=FEATURES + ["disposition", "created_at"])

@bp.route("/")
def root():
    return redirect(url_for("main.novato"))

# -------- NOVATO --------
@bp.route("/novato", methods=["GET"])
@login_required
def novato():
    model_name, params = DEFAULT_NOVICE_MODEL
    use_user_data = False
    bundle, status = get_model_bundle(model_name, params, use_user_data, current_user.id, user_rows_df=_get_user_rows_df())
    return render_template("novato.html", features=FEATURES, labels=UI_LABELS, status=status)

@bp.route("/predict_novato", methods=["POST"])
@login_required
def predict_novato():
    values = [float(request.form.get(key, "").strip()) for key in FEATURES]
    model_name, params = DEFAULT_NOVICE_MODEL
    use_user_data = False
    bundle, status = get_model_bundle(model_name, params, use_user_data, current_user.id, user_rows_df=_get_user_rows_df())
    
    # Realizar la predicción
    pred_label, proba_map = predict_one(bundle, values)
    

    vals = dict(zip(FEATURES, values))

    # helper: semieje mayor aprox (M⋆~1 M☉)
    def _estimate_sma_au(period_days):
        try:
            P = float(period_days) / 365.25
            return (P**2) ** (1/3)   # P^{2/3}
        except:
            return None

    session["viz3d_payload"] = {
        "pred_label": pred_label,    # "Confirmado" / "Falso positivo"
        "score": proba_map,          # {"CONFIRMED": 0.58, "FALSE POSITIVE": 0.42, ...}

        # NO te cambio nombres: uso tus keys tal cual
        "star": {
            "st_rad":  vals.get("st_rad"),     # R☉
            "st_teff": vals.get("st_teff")     # K
        },
        "planet": {
            "pl_orbper": vals.get("pl_orbper"),  # días
            "pl_rade":   vals.get("pl_rade"),    # R⊕
            "pl_eqt":    vals.get("pl_eqt"),     # K (opcional)
        },
        "derived": {
            "semi_major_au": _estimate_sma_au(vals.get("pl_orbper"))
        }
    }
    session.modified = True

    # Almacenar la predicción en la sesión
    session["viz3d_payload"] = {
        "pred_label": pred_label,  # Asegúrate de que esto sea "Confirmado" cuando el modelo clasifique como tal
        "score": proba_map
    }

    # Ahora podemos usar los valores en el template
    p_confirmed = proba_map.get("CONFIRMED", 0.0)
    
    return render_template(
        "result.html", 
        mode="novato", 
        labels=UI_LABELS, 
        values=dict(zip(FEATURES, values)),
        pred_label=pred_label, 
        proba_map=proba_map, 
        status=status, 
        model_name=model_name, 
        params=params,
        p_confirmed=p_confirmed  # Pasa p_confirmed aquí
    )


# -------- EXPERTO --------
@bp.route("/experto", methods=["GET"])
@login_required
def experto():
    current_cfg = session.get("current_model")      # dict: model_name, params, use_user_data
    current_status = session.get("current_status")  # metrics dict
    return render_template("experto.html", features=FEATURES, labels=UI_LABELS, current_cfg=current_cfg, status=current_status)

@bp.route("/predict_experto", methods=["POST"])
@login_required
def predict_experto():
    # Primero se calculan los valores
    values = [float(request.form.get(k, "").strip()) for k in FEATURES]
    cfg = session.get("current_model")
    user_df = _get_user_rows_df()
    bundle, status = get_model_bundle(cfg["model_name"], cfg["params"], cfg["use_user_data"], current_user.id, user_rows_df=user_df)
    pred_label, proba_map = predict_one(bundle, values)

    vals = dict(zip(FEATURES, values))

    # helper: semieje mayor aprox (M⋆~1 M☉)
    def _estimate_sma_au(period_days):
        try:
            P = float(period_days) / 365.25
            return (P**2) ** (1/3)   # P^{2/3}
        except:
            return None

    session["viz3d_payload"] = {
        "pred_label": pred_label,    # "Confirmado" / "Falso positivo"
        "score": proba_map,          # {"CONFIRMED": 0.58, "FALSE POSITIVE": 0.42, ...}

        # NO te cambio nombres: uso tus keys tal cual
        "star": {
            "st_rad":  vals.get("st_rad"),     # R☉
            "st_teff": vals.get("st_teff")     # K
        },
        "planet": {
            "pl_orbper": vals.get("pl_orbper"),  # días
            "pl_rade":   vals.get("pl_rade"),    # R⊕
            "pl_eqt":    vals.get("pl_eqt"),     # K (opcional)
        },
        "derived": {
            "semi_major_au": _estimate_sma_au(vals.get("pl_orbper"))
        }
    }
    session.modified = True


    # Aquí puedes ver el valor de pred_label
    print(f"pred_label: {pred_label}")  # Verifica el valor de pred_label
    flash(f"pred_label: {pred_label}", "info")  # Agrega esto si prefieres usar flash

    # Calcula p_confirmed
    p_confirmed = proba_map.get("CONFIRMED", 0.0)

    return render_template("result.html", 
                           mode="experto", 
                           labels=UI_LABELS, 
                           values=dict(zip(FEATURES, values)),
                           pred_label=pred_label, 
                           proba_map=proba_map, 
                           status=status,
                           model_name=cfg["model_name"], 
                           params=cfg["params"], 
                           p_confirmed=p_confirmed)

@bp.route("/train_experto", methods=["POST"])
@login_required
def train_experto():
    model_name = request.form.get("modelo", "Random Forest")
    params = {}
    if model_name == "Random Forest":
        params["n_estimators"] = int(request.form.get("n_estimators", 800))
        params["max_depth"] = int(request.form.get("max_depth", 20))
        params["min_samples_split"] = int(request.form.get("min_samples_split", 10))
        params["random_state"] = int(request.form.get("random_state", 42))
    elif model_name == "Decision Tree":
        params["max_depth"] = int(request.form.get("max_depth", 12))
        params["min_samples_split"] = int(request.form.get("min_samples_split", 10))
        params["min_samples_leaf"] = int(request.form.get("min_samples_leaf", 2))
        params["random_state"] = int(request.form.get("random_state", 42))
    else:  # Logistic Regression
        params["C"] = float(request.form.get("C", 1.0))
        params["max_iter"] = int(request.form.get("max_iter", 400))
        params["random_state"] = int(request.form.get("random_state", 42))

    use_user_data = request.form.get("use_user_data") == "on"
    user_df = _get_user_rows_df()

    try:
        metrics = train_and_persist(model_name, params, use_user_data, current_user.id, user_rows_df=user_df)
        session["current_model"] = {"model_name": model_name, "params": params, "use_user_data": use_user_data}
        session["current_status"] = metrics
        flash(f"Modelo entrenado: {model_name}. Accuracy: {metrics['accuracy']:.2%}", "success")
    except Exception as e:
        flash(f"Error al entrenar: {e}", "error")
    return redirect(url_for("main.experto"))

# -------- Dataset del usuario --------
@bp.route("/mis_datos", methods=["GET"])
@login_required
def mis_datos():
    df = _get_user_rows_df()
    rows = []
    for _, r in df.iterrows():
        rows.append({"created_at": r.get("created_at"), "disposition": r.get("disposition"), **{f: r.get(f) for f in FEATURES}})
    return render_template("mis_datos.html", rows=rows, features=FEATURES, labels=UI_LABELS)

# -------- Acciones post-predicción --------
@bp.route("/save_sample", methods=["POST"])
@login_required
def save_sample():
    label_text = request.form.get("predicted_raw", "CONFIRMED")
    if label_text == "Confirmado":
        disposition = "CONFIRMED"
    elif label_text == "Falso positivo":
        disposition = "FALSE POSITIVE"
    else:
        disposition = label_text
    try:
        row_kwargs = {f: float(request.form.get(f)) for f in FEATURES}
        sample = UserSample(user_id=current_user.id, disposition=disposition, **row_kwargs)
        db.session.add(sample)
        db.session.commit()
        flash("Muestra guardada en tu dataset.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"No se pudo guardar la muestra: {e}", "error")
    mode = request.form.get("mode", "novato")
    return redirect(url_for(f"main.{mode}"))

@bp.route("/retrain", methods=["POST"])
@login_required
def retrain():
    mode = request.form.get("mode", "novato")
    if mode == "novato":
        model_name, params = ("Random Forest", {"n_estimators": 1000, "random_state": 10})
        use_user_data = True
    else:
        cfg = session.get("current_model") or {"model_name": "Random Forest", "params": {"n_estimators": 1200, "max_depth": 20, "min_samples_split": 10, "random_state": 42}, "use_user_data": True}
        model_name, params, use_user_data = cfg["model_name"], cfg["params"], True

    user_df = _get_user_rows_df()
    try:
        metrics = train_and_persist(model_name, params, use_user_data=True, user_id=current_user.id, user_rows_df=user_df)
        session["current_model"] = {"model_name": model_name, "params": params, "use_user_data": True}
        session["current_status"] = metrics
        flash(f"Reentrenado con tus datos. Accuracy: {metrics['accuracy']:.2%}", "success")
    except Exception as e:
        flash(f"Error al reentrenar: {e}", "error")
    return redirect(url_for(f"main.{mode}"))

@bp.route("/clear_my_data", methods=["POST"])
@login_required
def clear_my_data():
    try:
        UserSample.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        flash("Tus muestras añadidas han sido eliminadas.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"No se pudo eliminar: {e}", "error")
    mode = request.form.get("mode", "novato")
    return redirect(url_for(f"main.{mode}"))
