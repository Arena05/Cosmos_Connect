from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
import json
import pandas as pd

from .ml_service import (
    FEATURES, UI_LABELS, get_model_bundle, predict_one, DEFAULT_NOVICE_MODEL, train_and_persist
)
from .extensions import db
from .models import UserSample

bp = Blueprint("main", __name__)

def _get_user_rows_df():
    if not current_user.is_authenticated:
        return pd.DataFrame(columns=FEATURES + ["disposition"])
    rows = UserSample.query.filter_by(user_id=current_user.id).all()
    data = [
        {**{f: getattr(r, f) for f in FEATURES}, "disposition": r.disposition}
        for r in rows
    ]
    return pd.DataFrame(data) if data else pd.DataFrame(columns=FEATURES + ["disposition"])

@bp.route("/")
def root():
    return redirect(url_for("main.novato"))

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
    values = []
    for key in FEATURES:
        values.append(float(request.form.get(key, "").strip()))
    model_name, params = DEFAULT_NOVICE_MODEL
    use_user_data = False
    bundle, status = get_model_bundle(model_name, params, use_user_data, current_user.id, user_rows_df=_get_user_rows_df())
    pred_label, proba_map = predict_one(bundle, values)
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
    )

@bp.route("/experto", methods=["GET"])
@login_required
def experto():
    return render_template("experto.html", features=FEATURES, labels=UI_LABELS)

@bp.route("/predict_experto", methods=["POST"])
@login_required
def predict_experto():
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
    else:
        params["C"] = float(request.form.get("C", 1.0))
        params["max_iter"] = int(request.form.get("max_iter", 400))
        params["random_state"] = int(request.form.get("random_state", 42))

    use_user_data = request.form.get("use_user_data") == "on"
    values = [float(request.form.get(k, "").strip()) for k in FEATURES]

    user_df = _get_user_rows_df()
    bundle, status = get_model_bundle(model_name, params, use_user_data, current_user.id, user_rows_df=user_df)
    pred_label, proba_map = predict_one(bundle, values)

    return render_template(
        "result.html",
        mode="experto",
        labels=UI_LABELS,
        values=dict(zip(FEATURES, values)),
        pred_label=pred_label,
        proba_map=proba_map,
        status=status,
        model_name=model_name,
        params=params,
        use_user_data=use_user_data,
    )

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
    model_name = request.form.get("modelo") or "Random Forest"
    params = json.loads(request.form.get("params_json") or "{}")
    if mode == "novato":
        model_name, params = ("Random Forest", {"n_estimators": 1000, "random_state": 10})
    user_df = _get_user_rows_df()

    try:
        metrics = train_and_persist(model_name, params, use_user_data=True, user_id=current_user.id, user_rows_df=user_df)
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
