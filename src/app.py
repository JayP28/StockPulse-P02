import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

from models import db, Episode, Review
from routes import register_routes
from final_rank import ALIAS_MAP, get_default_model

current_directory = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

register_routes(app)


def init_db():
    with app.app_context():
        db.create_all()

        if Episode.query.count() == 0:
            json_file_path = os.path.join(current_directory, "init.json")
            with open(json_file_path, "r") as file:
                data = json.load(file)

                for episode_data in data["episodes"]:
                    episode = Episode(
                        id=episode_data["id"],
                        title=episode_data["title"],
                        descr=episode_data["descr"],
                    )
                    db.session.add(episode)

                for review_data in data["reviews"]:
                    review = Review(
                        id=review_data["id"],
                        imdb_rating=review_data["imdb_rating"],
                    )
                    db.session.add(review)

            db.session.commit()
            print("Database initialized with episodes and reviews data")


init_db()


@app.route("/")
def stockpulse_home():
    return jsonify(
        {
            "message": "StockPulse Flask server is running",
            "available_routes": [
                "/api/stock/tickers",
                "/api/stock/analyze",
                "/api/stock/rankings",
            ],
        }
    )


@app.route("/api/stock/tickers", methods=["GET"])
def get_tickers():
    return jsonify({"tickers": sorted(ALIAS_MAP.keys())})


@app.route("/api/stock/analyze", methods=["POST"])
def analyze_stock():
    data = request.get_json(silent=True) or {}

    ticker = str(data.get("ticker", "")).strip().upper()
    top_k = data.get("top_k", 5)

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400

    if not ticker:
        return jsonify({"error": "ticker is required"}), 400

    try:
        model = get_default_model()
        result = model.analyze_ticker(ticker, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stock/rankings", methods=["GET"])
def get_rankings():
    top_k = request.args.get("top_k", 25)

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400

    try:
        model = get_default_model()
        ranked_df = model.rank_all_tickers(top_k=top_k)
        return jsonify(ranked_df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)