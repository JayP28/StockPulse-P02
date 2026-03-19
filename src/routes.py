"""
Routes: home page and episode search.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for LLM specific routes.
"""
import json
from flask import render_template, request
from models import db, Episode, Review

# ── AI toggle ──
USE_LLM = False
# USE_LLM = True
# ───────────────


def json_search(query):
    if not query or not query.strip():
        query = "Kardashian"
    results = db.session.query(Episode, Review).join(
        Review, Episode.id == Review.id
    ).filter(
        Episode.title.ilike(f'%{query}%')
    ).all()
    matches = []
    for episode, review in results:
        matches.append({
            'title': episode.title,
            'descr': episode.descr,
            'imdb_rating': review.imdb_rating
        })
    return json.dumps(matches)


def register_routes(app):
    @app.route("/")
    def home():
        return render_template("stockpulse.html")

    @app.route("/episodes")
    def episodes_search():
        text = request.args.get("title", "")
        return json_search(text)

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
