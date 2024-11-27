from flask import Flask
from app.app import home, login, qrcode_status, dashboard, logout


def create_app():
    app = Flask(__name__)

    # 注册路由
    app.add_url_rule("/", "home", home)
    app.add_url_rule("/login", "login", login)
    app.add_url_rule("/qrcode_status", "qrcode_status", qrcode_status, methods=["GET"])
    app.add_url_rule("/dashboard", "dashboard", dashboard)
    app.add_url_rule("/logout", "logout", logout, methods=["POST"])

    return app
