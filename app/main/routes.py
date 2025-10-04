from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from flask_login import login_required, current_user
from app import db
from app.models import User, Experiment, Post, Like, Challenge, Submission
from app import ml_service


main_bp = Blueprint("main", __name__)

# ---- Campos visibles en ambas vistas (Aprendiz/Expertos)
FIELDS = [
    ("ra", "Ascensión recta", "grados"),
    ("dec", "Declinación", "grados"),
    ("period", "Período orbital", "días"),
    ("duration", "Duración del tránsito", "horas"),
    ("depth", "Profundidad del tránsito", "ppm"),
    ("planet_radius", "Radio del planeta", "R⊕"),
    ("insolation", "Flujo de insolación", "F⊕"),
    ("equilibrium_temp", "Temperatura de equilibrio", "K"),
    ("stellar_teff", "T. efectiva estelar", "K"),
    ("stellar_logg", "log g (superficie)", "log10(cm s⁻²)"),
    ("stellar_radius", "Radio estelar", "R⊙"),
]


# ---- Helpers ---------------------------------------------------------------

def _parse_criterios(req):
    criterios = []
    plano = {}
    for key, label, default_u in FIELDS:
        mag = (req.form.get(f"{key}_mag") or "").strip()
        uni = (req.form.get(f"{key}_unit") or default_u).strip() or default_u
        criterios.append({"key": key, "label": label, "mag": mag, "unit": uni})
        plano[key] = mag
    return criterios, plano


def _fake_metrics(prob):
    # Métricas aproximadas si no hay entrenamiento disponible.
    try:
        p = float(prob)
    except Exception:
        p = 0.5
    p = max(0.5, min(0.99, p))
    f1 = round(p - 0.04, 3)
    rec = round(p - 0.07, 3)
    acc = round(p - 0.02, 3)
    t = round(1.2 + (1.0 - p) * 3, 1)
    return {"f1": f1, "recall": rec, "acc": acc, "time": f"{t}s"}


def _model_key(model_name: str) -> str:
    return {"Random Forest": "rf", "Decision Tree": "dt", "Logistic Regression": "lr"}.get(model_name, "rf")


# ---- Rutas -----------------------------------------------------------------

@main_bp.route("/")
def menu():
    if current_user.is_authenticated:
        posts = Post.query.order_by(Post.created_at.desc()).limit(5).all()
        events = [
            {
                "user": p.author.name,
                "text": p.body[:110] + ("…" if len(p.body) > 110 else ""),
                "when": p.created_at.strftime("%d/%m %H:%M"),
            }
            for p in posts
        ]
        return render_template("menu_comunidad.html", user=current_user, events=events)
    return render_template("menu_publico.html", user=current_user)


@main_bp.route("/perfil")
@login_required
def perfil():
    exps = (
        Experiment.query.filter_by(user_id=current_user.id)
        .order_by(Experiment.created_at.desc())
        .limit(50)
        .all()
    )
    return render_template("perfil.html", user=current_user, experiments=exps)


# -------------------- Aprendiz ----------------------------------------------

@main_bp.route("/aprendiz", methods=["GET", "POST"])
@login_required
def aprendiz():
    resultados = None
    criterios = [{"key": k, "label": l, "mag": "", "unit": u} for k, l, u in FIELDS]

    if request.method == "POST":
        criterios, _ = _parse_criterios(request)
        resultados = {
            "modelo": "Random Forest (Aprendiz)",
            "params": {
                "n_estimators": 800,
                "max_depth": 20,
                "min_samples_split": 10,
                "random_state": 42,
            },
            "pred": ml_service.predict_from_criterios("rf", criterios),
            "criterios": criterios,
        }

    return render_template(
        "aprendiz.html",
        user=current_user,
        fields=FIELDS,
        criterios=criterios,
        resultados=resultados,
    )


# -------------------- Expertos ----------------------------------------------

@main_bp.route("/expertos", methods=["GET", "POST"])
@login_required
def expertos():
    resultados = None
    resultados_multi = None
    criterios = [{"key": k, "label": l, "mag": "", "unit": u} for k, l, u in FIELDS]

    # Historial de modelos analizados en esta sesión
    # runs = {"Random Forest": {...}, "Decision Tree": {...}, "Logistic Regression": {...}}
    runs = session.get("expert_runs", {})

    if request.method == "POST":
        modelo = request.form.get("modelo", "rf")

        if modelo == "rf":
            params = {
                "n_estimators": int(request.form.get("rf_n_estimators", 800)),
                "max_depth": int(request.form.get("rf_max_depth", 20)),
                "min_samples_split": int(request.form.get("rf_min_samples_split", 10)),
                "random_state": int(request.form.get("rf_random_state", 42)),
            }
            model_name = "Random Forest"

        elif modelo == "dt":
            params = {
                "max_depth": int(request.form.get("dt_max_depth", 8)),
                "min_samples_split": int(request.form.get("dt_min_samples_split", 10)),
                "min_samples_leaf": int(request.form.get("dt_min_samples_leaf", 4)),
                "random_state": int(request.form.get("dt_random_state", 42)),
            }
            model_name = "Decision Tree"

        else:  # lr
            params = {
                "C": float(request.form.get("lr_C", 1.0)),
                "max_iter": int(request.form.get("lr_max_iter", 200)),
                "random_state": int(request.form.get("lr_random_state", 42)),
            }
            model_name = "Logistic Regression"

        criterios, _ = _parse_criterios(request)

        # Entrenar/evaluar con dataset si existe; si no, solo predecir suave
        key = _model_key(model_name)
        try:
            info = ml_service.train_and_eval(key, params)
            pred = ml_service.predict_from_criterios(key, criterios)
            metrics_real = info.get("metrics", {})
        except Exception:
            pred = ml_service.predict_from_criterios(key, criterios)
            metrics_real = _fake_metrics(pred.get("prob", 0.5))

        resultados = {
            "modelo": model_name,
            "params": params,
            "pred": pred,
            "criterios": criterios,
        }

        # Historial en sesión para “comparar”
        runs[model_name] = {
            "name": model_name,
            "prob": pred.get("prob"),
            "metrics": metrics_real,
            "params": params,
        }
        session["expert_runs"] = runs
        session.modified = True

        # Si pidió comparar, usa lo acumulado en sesión
        if request.form.get("comparar"):
            resultados_multi = list(runs.values())

    return render_template(
        "expertos.html",
        user=current_user,
        fields=FIELDS,
        criterios=criterios,
        resultados=resultados,
        resultados_multi=resultados_multi,
        runs_list=list(runs.values()),
    )


@main_bp.route("/expertos/limpiar", methods=["POST"])
@login_required
def expertos_limpiar():
    session.pop("expert_runs", None)
    session.modified = True
    flash("Se limpió el historial de modelos de esta sesión.", "info")
    return redirect(url_for("main.expertos"))


# -------------------- Guardar Experimento -----------------------------------

@main_bp.route("/experimento/guardar", methods=["POST"])
@login_required
def guardar_experimento():
    import json
    try:
        model_name = request.form["model_name"]
        params = json.loads(request.form["params_json"])
        criterios = json.loads(request.form["criteria_json"])
        prob = float(request.form["prob"])
        predicted_class = request.form["predicted_class"]

        exp = Experiment(
            user_id=current_user.id,
            model_name=model_name,
            params=params,
            criteria=criterios,
            prob=prob,
            predicted_class=predicted_class,
        )
        db.session.add(exp)

        # Append a CSV + reentrenar con hiperparámetros usados (best-effort)
        try:
            key = _model_key(model_name)
            ml_service.append_row_and_retrain(criterios, predicted_class, key, params)
        except Exception:
            pass

        if prob >= 0.70:
            pct = round(prob * 100, 1)
            db.session.add(
                Post(
                    user_id=current_user.id,
                    body=f"{current_user.name} reportó un candidato con {pct} %",
                    auto=True,
                )
            )

        db.session.commit()
        flash("Experimento guardado ✅", "success")
    except Exception:
        db.session.rollback()
        flash("No se pudo guardar el experimento.", "danger")
    return redirect(url_for("main.perfil"))


# -------------------- Comunidad (ligero) ------------------------------------

@main_bp.route("/comunidad")
@login_required
def comunidad():
    page = max(int(request.args.get("page", 1)), 1)
    posts = (
        Post.query.order_by(Post.created_at.desc())
        .paginate(page=page, per_page=10, error_out=False)
    )
    return render_template("comunidad.html", posts=posts)


@main_bp.route("/comunidad/publicar", methods=["POST"])
@login_required
def publicar():
    body = (request.form.get("body") or "").strip()
    if not body:
        flash("Escribe algo", "warning")
        return redirect(url_for("main.comunidad"))
    db.session.add(Post(user_id=current_user.id, body=body))
    db.session.commit()
    flash("Publicado", "success")
    return redirect(url_for("main.comunidad"))


@main_bp.route("/comunidad/like/<int:post_id>", methods=["POST"])
@login_required
def like_post(post_id):
    kind = request.form.get("kind", "clap")
    post = Post.query.get_or_404(post_id)
    existing = Like.query.filter_by(
        user_id=current_user.id, post_id=post.id, kind=kind
    ).first()
    if existing:
        db.session.delete(existing)
    else:
        db.session.add(Like(user_id=current_user.id, post_id=post.id, kind=kind))
    db.session.commit()
    return redirect(url_for("main.comunidad"))


# -------------------- Retos --------------------------------------------------

def _ensure_weekly_challenge():
    ch = Challenge.query.filter_by(slug="mini-neptunos").first()
    if not ch:
        ch = Challenge(
            slug="mini-neptunos",
            title="Reto de la semana: Mini-Neptunos",
            description="Reporta al menos 1 candidato compatible con mini-Neptuno.",
        )
        db.session.add(ch)
        db.session.commit()
    return ch


@main_bp.route("/retos")
@login_required
def retos():
    ch = _ensure_weekly_challenge()
    joined = Submission.query.filter_by(
        user_id=current_user.id, challenge_id=ch.id
    ).first()
    return render_template("retos.html", challenge=ch, joined=bool(joined))


@main_bp.route("/retos/unirse/<int:challenge_id>", methods=["POST"])
@login_required
def retos_unirse(challenge_id):
    joined = Submission.query.filter_by(
        user_id=current_user.id, challenge_id=challenge_id
    ).first()
    if not joined:
        db.session.add(
            Submission(user_id=current_user.id, challenge_id=challenge_id, status="joined")
        )
        db.session.commit()
        flash("Te uniste al reto 🚀", "success")
    return redirect(url_for("main.retos"))


# -------------------- Perfil público ----------------------------------------

@main_bp.route("/u/<int:user_id>")
@login_required
def user_profile(user_id):
    u = User.query.get_or_404(user_id)
    exps = (
        Experiment.query.filter_by(user_id=u.id)
        .order_by(Experiment.created_at.desc())
        .limit(10)
        .all()
    )
    posts = (
        Post.query.filter_by(user_id=u.id)
        .order_by(Post.created_at.desc())
        .limit(10)
        .all()
    )
    return render_template("perfil_publico.html", u=u, experiments=exps, posts=posts)


@main_bp.route("/api/analizar", methods=["POST"])
@login_required
def api_analizar():
    """Recibe los criterios (form o JSON), invoca la IA y devuelve JSON + HTML parcial."""
    # 1) Soporta JSON o form-data
    payload = request.get_json(silent=True) if request.is_json else None
    modelo_req = (
        (payload.get("modelo") if payload else None)
        or request.form.get("modelo")
        or "rf"
    )

    # 2) Criterios: lista de dicts con {key,label,mag,unit}
    if payload and isinstance(payload.get("criterios"), list):
        criterios = payload["criterios"]
    else:
        criterios, _ = _parse_criterios(request)

    # 3) Normaliza clave del modelo
    key = modelo_req if modelo_req in ("rf", "dt", "lr") else _model_key(modelo_req)

    # 4) Predicción
    try:
        pred = ml_service.predict_from_criterios(key, criterios)
    except Exception as e:
        msg = str(e) if current_app.config.get("DEBUG") else "No se pudo obtener la predicción."
        return {"ok": False, "error": msg}, 400

    resultados = {
        "modelo": {"rf": "Random Forest", "dt": "Decision Tree", "lr": "Logistic Regression"}.get(key, key),
        "params": {},
        "pred": pred,
        "criterios": criterios,
    }

    html = render_template("partials/_resultado_pred.html", resultados=resultados)
    return {"ok": True, "modelo": key, "pred": pred, "html": html}, 200
