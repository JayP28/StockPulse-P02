"""
LLM chat route — optional.
Adds a POST /chat endpoint that performs LLM-grounded Q&A over StockPulse results.

Setup:
  1. Add API_KEY=your_key to .env
  2. Wire register_chat_route(app, model) in app.py if desired
"""
import json
import logging
import os
import re

from flask import Response, jsonify, request, stream_with_context
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)


def llm_search_decision(client, user_message):
    """
    Ask the LLM if a stock analysis lookup is needed, and if so which ticker.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are routing questions for a stock sentiment app. "
                "If the user is asking about a specific stock, reply exactly: "
                "YES followed by one ticker symbol, for example: YES AAPL. "
                "If not, reply exactly: NO."
            ),
        },
        {"role": "user", "content": user_message},
    ]

    response = client.chat(messages)
    content = (response.get("content") or "").strip().upper()
    logger.info("LLM search decision: %s", content)

    if re.fullmatch(r"NO\.?", content):
        return False, None

    match = re.search(r"\bYES\s+([A-Z.$]{1,10})\b", content)
    if match:
        return True, match.group(1).replace("$", "")

    return False, None


def register_chat_route(app, model):
    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "API_KEY not set — add it to your .env file"}), 500

        client = LLMClient(api_key=api_key)
        use_search, ticker = llm_search_decision(client, user_message)

        if use_search and ticker:
            result = model.analyze_ticker(ticker, top_k=5)
            context_text = json.dumps(result, indent=2)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a stock sentiment assistant. "
                        "Use only the provided StockPulse analysis and snippets. "
                        "Do not invent outside facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"StockPulse analysis:\n{context_text}\n\n"
                        f"User question: {user_message}"
                    ),
                },
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for the StockPulse app.",
                },
                {"role": "user", "content": user_message},
            ]

        def generate():
            if use_search and ticker:
                yield f"data: {json.dumps({'ticker': ticker})}\n\n"
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as exc:
                logger.error("Streaming error: %s", exc)
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )