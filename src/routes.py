from flask import render_template


def register_routes(app):
    @app.route("/")
    def home():
        return render_template("stockpulse.html")