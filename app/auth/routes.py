from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models import User
from .forms import LoginForm, RegisterForm

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    # Si ya está logueado, mándalo al menú de la comunidad
    if current_user.is_authenticated:
        return redirect(url_for("main.menu"))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower()).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=True)
            flash("¡Bienvenido de nuevo!", "success")
            # Si venía de una página protegida, respétala; si no, menú de comunidad
            next_page = request.args.get("next")
            return redirect(next_page or url_for("main.menu"))
        flash("Credenciales inválidas", "danger")
    return render_template("auth/login.html", form=form)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.menu"))
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data.lower()).first():
            flash("Ese correo ya está registrado.", "warning")
        else:
            user = User(email=form.email.data.lower(), name=form.name.data.strip())
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash("Cuenta creada. Inicia sesión.", "success")
            return redirect(url_for("auth.login"))
    return render_template("auth/register.html", form=form)

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Sesión cerrada.", "info")
    return redirect(url_for("main.menu"))
