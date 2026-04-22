"""
Routes: home page and StockPulse search.
To enable AI chat/RAG, USE_LLM = True is set below. See llm_routes.py for LLM-specific routes.
"""
from flask import render_template

# ── AI toggle ──
USE_LLM = True
# ───────────────

def register_routes(app):
    @app.route("/")
    def home():
        return render_template("stockpulse.html")

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app)