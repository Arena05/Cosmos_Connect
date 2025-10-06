from flask import Flask
from pathlib import Path
from .extensions import db, login_manager

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config["SECRET_KEY"] = "zOMp1RRllojiYyl3NoEZoWZ9fUNd86g1"
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + str(Path(app.instance_path) / "app.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    (Path(app.instance_path) / "models").mkdir(parents=True, exist_ok=True)
    (Path(app.instance_path) / "cache").mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    from .routes import bp as main_bp
    from .auth import bp as auth_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)

    with app.app_context():
        db.create_all()

    return app
