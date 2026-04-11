import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

from final_rank import ALIAS_MAP, get_default_model
from routes import register_routes

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

register_routes(app)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "message": "StockPulse Flask server is running",
            "available_routes": [
                "/",
                "/api/health",
                "/api/stock/tickers",
                "/api/stock/analyze",
                "/api/stock/rankings",
                "/api/stock/methodology",
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

    if top_k <= 0:
        return jsonify({"error": "top_k must be greater than 0"}), 400

    if not ticker:
        return jsonify({"error": "ticker is required"}), 400

    try:
        model = get_default_model()
        result = model.analyze_ticker(ticker, top_k=top_k)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stock/rankings", methods=["GET"])
def get_rankings():
    top_k = request.args.get("top_k", 25)

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400

    if top_k <= 0:
        return jsonify({"error": "top_k must be greater than 0"}), 400

    try:
        model = get_default_model()
        ranked_df = model.rank_all_tickers(top_k=top_k)
        return jsonify(ranked_df.to_dict(orient="records"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stock/methodology", methods=["GET"])
def get_methodology():
    try:
        model = get_default_model()
        return jsonify(model.get_methodology())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)