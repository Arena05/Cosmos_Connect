from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    load_dotenv()
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object("app.config.Config")

    # Inicializar extensiones
    db.init_app(app)
    login_manager.init_app(app)

    # Endpoint de login (ajústalo si tu vista no es auth.login)
    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "warning"

    # Importar modelos y definir el user_loader
    from app.models import User

    @login_manager.user_loader
    def load_user(user_id: str):
        try:
            return User.query.get(int(user_id))
        except Exception:
            return None

    # Registrar blueprints
    from app.auth.routes import auth_bp
    from app.main.routes import main_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    # Crear tablas
    with app.app_context():
        db.create_all()

    return app
