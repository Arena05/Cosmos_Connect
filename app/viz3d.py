# app/viz3d.py
from flask import Blueprint, render_template, session, redirect, url_for
from flask_login import current_user

bp = Blueprint("viz3d", __name__)

@bp.route("/viz3d", methods=["GET"])
def viz3d():
    data = session.get("viz3d_payload")
    if not data or not current_user.is_authenticated:
        return redirect(url_for("auth.login"))

    # abrir solo si la clasificación es Confirmado (da igual el %)
    if data.get("pred_label") != "Confirmado":
        return render_template("viz3d_denied_standalone.html",
                               pred_label=data.get("pred_label"))

    return render_template("viz3d_standalone.html", system_json=data)
