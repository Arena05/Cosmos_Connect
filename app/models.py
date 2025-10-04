from datetime import datetime
from flask_login import UserMixin
from . import db
from werkzeug.security import generate_password_hash, check_password_hash

# ---- Usuario ----
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    bio = db.Column(db.String(160), default="")

    # Relaciones
    posts = db.relationship("Post", backref="author", lazy="dynamic", cascade="all, delete-orphan")
    experiments = db.relationship("Experiment", backref="user", lazy="dynamic", cascade="all, delete-orphan")

        # --- Password helpers ---
    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

    # Alias opcional si tu código usa 'verify_password'
    def verify_password(self, raw_password: str) -> bool:
        return self.check_password(raw_password)

# ---- Experimentos ----
class Experiment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    model_name = db.Column(db.String(50), nullable=False)
    params = db.Column(db.JSON, nullable=False)      # hiperparámetros
    criteria = db.Column(db.JSON, nullable=False)    # lista [{label, mag, unit}]
    prob = db.Column(db.Float, nullable=False)       # 0..1
    predicted_class = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ---- Comunidad: Posts y Reacciones ----
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    auto = db.Column(db.Boolean, default=False)  # true si es un logro automático
    likes = db.relationship("Like", backref="post", lazy="dynamic", cascade="all, delete-orphan")

class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey("post.id"), nullable=False)
    kind = db.Column(db.String(16), default="clap")  # 'clap' | 'star' | 'sat'
    __table_args__ = (db.UniqueConstraint("user_id", "post_id", "kind", name="uix_like_kind"),)

# ---- Retos ----
class Challenge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(50), unique=True, nullable=False)
    title = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, default="")
    active = db.Column(db.Boolean, default=True)

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    challenge_id = db.Column(db.Integer, db.ForeignKey("challenge.id"), nullable=False)
    status = db.Column(db.String(20), default="joined")  # joined | completed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relación para acceder a los datos del reto desde la submission
    challenge = db.relationship("Challenge")
