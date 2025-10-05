from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from .extensions import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

class UserSample(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    ra = db.Column(db.Float, nullable=False)
    dec = db.Column(db.Float, nullable=False)
    period = db.Column(db.Float, nullable=False)
    duration = db.Column(db.Float, nullable=False)
    depth = db.Column(db.Float, nullable=False)
    planet_radius = db.Column(db.Float, nullable=False)
    insolation = db.Column(db.Float, nullable=False)
    equilibrium_temp = db.Column(db.Float, nullable=False)
    stellar_teff = db.Column(db.Float, nullable=False)
    stellar_logg = db.Column(db.Float, nullable=False)
    stellar_radius = db.Column(db.Float, nullable=False)
    disposition = db.Column(db.String(32), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
