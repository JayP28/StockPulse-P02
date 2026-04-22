"""
LLM chat route — only loaded when USE_LLM = True in routes.py.
Adds:
  POST /api/rag/analyze  — RAG: runs IR search + yfinance linear regression,
                           then streams an LLM summary combining both.
  GET  /api/stock/predict — returns raw linear regression prediction data for a ticker.

Setup:
  1. Add SPARK_API_KEY=your_key to .env
  2. USE_LLM = True is already set in routes.py
"""
import json
import logging
import os

import numpy as np
from flask import Response, jsonify, request, stream_with_context

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_client():
    """Return an LLMClient or raise RuntimeError if key is missing."""
    api_key = os.getenv("SPARK_API_KEY")
    if not api_key:
        raise RuntimeError("SPARK_API_KEY not set — add it to your .env file")
    from infosci_spark_client import LLMClient
    return LLMClient(api_key=api_key)


def get_stock_prediction(ticker: str) -> dict:
    """
    Fetch 1 year of daily closing prices via yfinance, fit a linear regression
    (OLS via numpy), and return a structured prediction dict.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance is not installed. Run: pip install yfinance"}

    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
    except Exception as exc:
        return {"error": f"yfinance download failed: {exc}"}

    if data is None or data.empty or "Close" not in data.columns:
        return {"error": f"No price data found for ticker '{ticker}'."}

    closes = data["Close"].dropna()
    # Newer yfinance returns a multi-index DataFrame — squeeze to a 1D Series
    if hasattr(closes, "squeeze"):
        closes = closes.squeeze()
    if len(closes) < 10:
        return {"error": f"Not enough price history for '{ticker}' (need >= 10 days)."}

    x = np.arange(len(closes), dtype=np.float64)
    # Flatten to 1D array to avoid shape issues
    y = np.array(closes, dtype=np.float64).flatten()

    # OLS linear regression via numpy polyfit degree 1
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs.flat[0]), float(coeffs.flat[1])

    y_hat = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    last_idx = len(closes) - 1
    last_price = float(y[-1])
    pred_30 = float(np.polyval(coeffs, last_idx + 30))
    pred_90 = float(np.polyval(coeffs, last_idx + 90))

    pct_30 = (pred_30 - last_price) / last_price * 100
    pct_90 = (pred_90 - last_price) / last_price * 100

    daily_pct = slope / last_price * 100
    if daily_pct > 0.05:
        direction = "upward"
        strength = "strong" if daily_pct > 0.15 else "moderate"
    elif daily_pct < -0.05:
        direction = "downward"
        strength = "strong" if daily_pct < -0.15 else "moderate"
    else:
        direction = "sideways"
        strength = "weak"

    history_sample = []
    step = max(1, len(closes) // 52)
    for i in range(0, len(closes), step):
        history_sample.append({
            "date": str(closes.index[i].date()),
            "close": round(float(closes.iloc[i]), 4),
        })

    explanation = (
        f"Linear regression was fitted on {len(closes)} trading days of closing prices "
        f"for {ticker.upper()} using ordinary least squares (OLS). "
        f"The trend line has a slope of {slope:+.4f} per trading day "
        f"(approx. {daily_pct:+.3f}% of current price per day). "
        f"R squared = {r_squared:.4f} — "
        f"{'a strong' if r_squared > 0.7 else 'a moderate' if r_squared > 0.4 else 'a weak'} "
        f"linear fit to the historical data. "
        f"Projecting the regression line 30 trading days forward gives ${pred_30:.2f} ({pct_30:+.1f}%) "
        f"and 90 days forward gives ${pred_90:.2f} ({pct_90:+.1f}%). "
        f"Note: linear regression captures only the long-term directional trend. "
        f"It does not model volatility, earnings, or macro shocks — treat projections as directional guides only."
    )

    return {
        "ticker": ticker.upper(),
        "trading_days_used": len(closes),
        "slope": round(slope, 6),
        "intercept": round(intercept, 4),
        "r_squared": round(r_squared, 4),
        "last_price": round(last_price, 4),
        "predicted_price_30d": round(pred_30, 4),
        "predicted_price_90d": round(pred_90, 4),
        "pct_change_30d": round(pct_30, 2),
        "pct_change_90d": round(pct_90, 2),
        "trend_direction": direction,
        "trend_strength": strength,
        "daily_slope_pct": round(daily_pct, 4),
        "explanation": explanation,
        "history": history_sample,
    }


def _ir_context_text(ir_results: dict) -> str:
    """Format IR results into a concise text block for the LLM prompt."""
    lines = []
    query = ir_results.get("query", "")
    lines.append(f"StockPulse IR Query: {query}\n")

    for method, key in [("Baseline TF-IDF", "baseline_results"), ("SVD", "svd_results")]:
        results = ir_results.get(key, [])
        if not results:
            continue
        lines.append(f"\n--- {method} Top Results ---")
        for r in results[:3]:
            lines.append(
                f"  #{r.get('rank','?')} | Similarity: {r.get('similarity',0):.4f} | "
                f"Score: {r.get('score',0)} | Comments: {r.get('comms_num',0)}"
            )
            lines.append(f"  Title: {r.get('title','')}")
            lines.append(f"  Preview: {r.get('preview','')[:300]}")
            dims = r.get("aligned_dimensions", [])
            if dims:
                dim_str = "; ".join(
                    f"D{d['dimension']} ({d['short_label']})" for d in dims[:2]
                )
                lines.append(f"  SVD Dims: {dim_str}")

    expl = ir_results.get("svd_explainability", {})
    dims = expl.get("important_dimensions", [])
    if dims:
        lines.append("\n--- Top SVD Latent Dimensions ---")
        for d in dims[:3]:
            lines.append(
                f"  D{d['dimension']}: {d['short_label']} "
                f"(strength={d['query_strength']:+.4f}, pole={d['query_pole']})"
            )
            lines.append(f"    Active terms: {', '.join(d.get('active_terms', [])[:5])}")

    return "\n".join(lines)


def _prediction_context_text(pred: dict) -> str:
    """Format the linear regression prediction into a text block for the LLM prompt."""
    if "error" in pred:
        return f"Price prediction unavailable: {pred['error']}"

    return (
        f"Stock Price Prediction for {pred['ticker']}:\n"
        f"  Current (last close): ${pred['last_price']:.2f}\n"
        f"  Linear regression R squared: {pred['r_squared']:.4f} "
        f"({'strong' if pred['r_squared'] > 0.7 else 'moderate' if pred['r_squared'] > 0.4 else 'weak'} fit)\n"
        f"  Slope: {pred['slope']:+.6f}/day ({pred['daily_slope_pct']:+.3f}%/day)\n"
        f"  30-day projection: ${pred['predicted_price_30d']:.2f} ({pred['pct_change_30d']:+.1f}%)\n"
        f"  90-day projection: ${pred['predicted_price_90d']:.2f} ({pred['pct_change_90d']:+.1f}%)\n"
        f"  Trend: {pred['trend_strength']} {pred['trend_direction']}\n"
        f"  Methodology: {pred['explanation']}"
    )


# ── route registration ─────────────────────────────────────────────────────────

def register_chat_route(app):
    """Register /api/rag/analyze (SSE stream) and /api/stock/predict (JSON)."""

    @app.route("/api/stock/predict", methods=["GET", "POST"])
    def stock_predict():
        if request.method == "GET":
            ticker = request.args.get("ticker", "").strip().upper()
        else:
            body = request.get_json(silent=True) or {}
            ticker = str(body.get("ticker", "")).strip().upper()

        if not ticker:
            return jsonify({"error": "ticker is required"}), 400

        pred = get_stock_prediction(ticker)
        if "error" in pred:
            return jsonify(pred), 400
        return jsonify(pred)

    @app.route("/api/rag/analyze", methods=["POST"])
    def rag_analyze():
        """
        Body: { "query": "NVDA", "top_k": 5 }

        SSE events emitted:
          data: {"type": "ir_results",   "payload": {...}}
          data: {"type": "prediction",   "payload": {...}}
          data: {"type": "llm_chunk",    "content": "..."}
          data: {"type": "done"}
          data: {"type": "error",        "message": "..."}
        """
        body = request.get_json(silent=True) or {}
        query = str(body.get("query", "")).strip()
        top_k = int(body.get("top_k", 5))

        if not query:
            return jsonify({"error": "query is required"}), 400

        def generate():
            # ── Step 1: IR search ────────────────────────────────────────────
            try:
                from retrieval import get_default_retriever
                retriever = get_default_retriever()
                ir_results = retriever.search(query, top_k=top_k)
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': f'IR search failed: {exc}'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'ir_results', 'payload': ir_results})}\n\n"

            # ── Step 2: Price prediction ─────────────────────────────────────
            pred = get_stock_prediction(query)
            yield f"data: {json.dumps({'type': 'prediction', 'payload': pred})}\n\n"

            # ── Step 3: LLM summary ──────────────────────────────────────────
            try:
                client = _get_client()
            except RuntimeError as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                return

            ir_text = _ir_context_text(ir_results)
            pred_text = _prediction_context_text(pred)

            system_prompt = (
                "You are StockPulse AI, a financial sentiment and technical analysis assistant. "
                "You have been given two sources of data about a stock ticker:\n"
                "1. Reddit discussion data retrieved by an information retrieval (IR) system "
                "using TF-IDF and SVD latent semantic analysis.\n"
                "2. A linear regression price projection built from 1 year of historical closing prices.\n\n"
                "Your job is to write a concise, insightful, well-structured summary (around 200-300 words) that:\n"
                "- Summarises the overall community sentiment captured in the Reddit posts.\n"
                "- Explains the key topics and themes the SVD dimensions found.\n"
                "- Describes what the linear regression predicts for the next 30 and 90 days, "
                "and whether the trend is upward, downward, or sideways.\n"
                "- Integrates both signals into an overall outlook statement.\n"
                "- Ends with a clear one-sentence verdict: bullish, bearish, or neutral, and why.\n\n"
                "Use plain language. Do NOT invent statistics or prices not in the data provided. "
                "Always include a reminder that this is not financial advice."
            )

            user_prompt = (
                f"Ticker / Query: {query.upper()}\n\n"
                f"=== IR RETRIEVAL DATA ===\n{ir_text}\n\n"
                f"=== PRICE PREDICTION DATA ===\n{pred_text}\n\n"
                "Please write the integrated summary now."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]

            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'type': 'llm_chunk', 'content': chunk['content']})}\n\n"
            except Exception as exc:
                logger.error("LLM streaming error: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'message': f'LLM streaming error: {exc}'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )