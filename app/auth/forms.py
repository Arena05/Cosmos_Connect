from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length, EqualTo

class LoginForm(FlaskForm):
    email = StringField("Correo", validators=[DataRequired(), Email()])
    password = PasswordField("Contraseña", validators=[DataRequired(), Length(min=6)])
    submit = SubmitField("Ingresar")

class RegisterForm(FlaskForm):
    name = StringField("Nombre", validators=[DataRequired(), Length(min=2, max=120)])
    email = StringField("Correo", validators=[DataRequired(), Email()])
    password = PasswordField("Contraseña", validators=[DataRequired(), Length(min=6)])
    confirm = PasswordField("Confirmar contraseña", validators=[DataRequired(), EqualTo("password")])
    submit = SubmitField("Crear cuenta")
