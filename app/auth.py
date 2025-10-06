from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, current_user
from .extensions import db, login_manager
from .models import User

bp = Blueprint("auth", __name__, url_prefix="/auth")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        if not email or not password:
            flash("Please enter email and password.", "error")
            return redirect(url_for("auth.register"))
        if User.query.filter_by(email=email).first():
            flash("That email is already registered.", "error")
            return redirect(url_for("auth.register"))
        user = User(email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please sign in.", "success")
        return redirect(url_for("auth.login"))
    return render_template("auth_register.html")

@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            flash("Signed in.", "success")
            return redirect(url_for("main.novato"))
        flash("Invalid credentials.", "error")
        return redirect(url_for("auth.login"))
    return render_template("auth_login.html")

@bp.route("/logout")
def logout():
    if current_user.is_authenticated:
        logout_user()
    flash("Signed out.", "success")
    return redirect(url_for("auth.login"))
