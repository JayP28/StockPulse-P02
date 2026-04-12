import os

from flask import Flask, jsonify, request

from retrieval import get_default_retriever, get_methodology_overview
from routes import register_routes

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

try:
    from flask_cors import CORS
except ImportError:
    def CORS(_app):
        return _app

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

register_routes(app)


def _read_request_data():
    if request.method == "GET":
        return request.args
    return request.get_json(silent=True) or {}


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "message": "StockPulse Flask server is running",
            "available_routes": [
                "/",
                "/api/health",
                "/api/stock/search",
                "/api/stock/methodology",
            ],
        }
    )


@app.route("/api/stock/search", methods=["GET", "POST"])
def search_stock_posts():
    data = _read_request_data()

    query = str(data.get("query", "")).strip()
    top_k = data.get("top_k", 5)

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400

    if top_k <= 0:
        return jsonify({"error": "top_k must be greater than 0"}), 400

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        retriever = get_default_retriever()
        return jsonify(retriever.search(query, top_k=top_k))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stock/methodology", methods=["GET"])
def get_methodology():
    try:
        return jsonify(get_methodology_overview())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
