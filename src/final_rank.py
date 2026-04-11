import math
import os
import re
from functools import lru_cache
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


class StockPulseSentiment:
    """
    Sentence-level stock IR + sentiment model.

    Core design:
    - Build a sentence corpus from Reddit title/body text
    - Keep only sentences that actually mention supported tickers/aliases
    - Retrieve relevant sentences with TF-IDF and latent semantic retrieval (SVD)
    - Score sentiment on the sentence where the ticker appears
    - Aggregate sentence evidence into post-level and ticker-level results
    """

    def __init__(self, csv_path: str, alias_map: Dict[str, List[str]]):
        self.df = pd.read_csv(csv_path)
        self.alias_map = {k.upper(): v for k, v in alias_map.items()}
        self.supported_tickers = list(self.alias_map.keys())

        # Normalize expected columns
        for col in ["title", "body", "url"]:
            if col not in self.df.columns:
                self.df[col] = ""

        self.df["title"] = self.df["title"].fillna("").astype(str)
        self.df["body"] = self.df["body"].fillna("").astype(str)
        self.df["url"] = self.df["url"].fillna("").astype(str)

        self.df["score"] = pd.to_numeric(self.df.get("score", 0), errors="coerce").fillna(0)
        self.df["comms_num"] = pd.to_numeric(self.df.get("comms_num", 0), errors="coerce").fillna(0)

        # Optional time field support
        self.time_col = None
        for candidate in ["created_utc", "timestamp", "created_at", "date"]:
            if candidate in self.df.columns:
                self.time_col = candidate
                break

        self.word_scores = {
            "bull": 1.4,
            "bullish": 2.0,
            "buy": 1.3,
            "long": 0.9,
            "calls": 1.6,
            "call": 1.3,
            "undervalued": 1.7,
            "beat": 1.5,
            "beats": 1.5,
            "green": 0.8,
            "rally": 1.3,
            "rip": 1.1,
            "moon": 2.0,
            "mooning": 2.0,
            "rocket": 1.8,
            "rockets": 1.8,
            "squeeze": 1.4,
            "hold": 0.4,
            "hodl": 0.8,
            "upgrade": 1.4,
            "outperform": 1.5,
            "strong": 0.7,
            "great": 0.6,
            "winner": 1.3,
            "bear": -1.4,
            "bearish": -2.0,
            "sell": -1.3,
            "short": -1.0,
            "puts": -1.6,
            "put": -1.3,
            "overvalued": -1.7,
            "miss": -1.5,
            "missed": -1.5,
            "red": -0.8,
            "dump": -1.6,
            "crash": -2.0,
            "crashing": -2.0,
            "bagholder": -1.7,
            "bankrupt": -2.4,
            "bankruptcy": -2.4,
            "fraud": -2.2,
            "downgrade": -1.7,
            "plunge": -1.8,
            "tank": -1.8,
            "tanking": -1.8,
            "weak": -0.8,
            "terrible": -1.3,
            "awful": -1.4,
            "loser": -1.5,
        }

        self.phrase_scores = {
            "buy the dip": 2.2,
            "short squeeze": 2.4,
            "to the moon": 2.6,
            "beats earnings": 2.1,
            "beat earnings": 2.1,
            "price target raised": 1.8,
            "going to zero": -2.8,
            "miss earnings": -2.1,
            "missed earnings": -2.1,
            "price target cut": -1.8,
            "dead cat bounce": -1.8,
            "sell the rip": -1.4,
            "load the boat": 1.8,
            "strong buy": 2.2,
            "hard pass": -1.5,
        }

        self.negations = {
            "not", "no", "never", "isnt", "wasnt", "dont", "doesnt", "cant", "wont", "ain't"
        }

        self._ticker_to_patterns = {
            ticker: self._ticker_patterns(ticker) for ticker in self.supported_tickers
        }

        self.sentence_df = self._build_sentence_corpus()

        if self.sentence_df.empty:
            raise ValueError(
                "No ticker-containing sentences were found in the dataset. "
                "Check prototype_posts.csv and alias coverage."
            )

        self.vectorizer, self.doc_tfidf, self.svd_model, self.doc_lsi = self._fit_retrieval_models()

    def _clean_text(self, text: str) -> str:
        text = str(text)
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = text.replace("🚀", " rocket ")
        text = text.replace("🟢", " green ")
        text = text.replace("🔴", " red ")
        text = text.replace("📈", " bullish ")
        text = text.replace("📉", " bearish ")
        text = text.replace("&amp;", "and")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _normalize_for_vectorizer(self, text: str) -> str:
        text = self._clean_text(text).lower()
        text = re.sub(r"[^a-z0-9$.\-+\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _split_sentences(self, text: str) -> List[str]:
        text = self._clean_text(text)
        if not text:
            return []

        text = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        parts = [p.strip() for p in parts if p and p.strip()]

        cleaned = []
        for sent in parts:
            sent = sent.strip(" -•\t")
            if len(sent) < 6:
                continue
            cleaned.append(sent)

        return cleaned

    def _ticker_patterns(self, ticker: str) -> List[re.Pattern]:
        aliases = self.alias_map.get(ticker.upper(), [ticker.upper()])
        patterns = []

        for alias in aliases:
            alias = alias.strip()
            if not alias:
                continue

            if alias.startswith("$"):
                raw = alias[1:]
                if raw:
                    patterns.append(
                        re.compile(rf"(?<![A-Za-z0-9])\$?{re.escape(raw)}(?![A-Za-z0-9])", re.IGNORECASE)
                    )
            elif alias.isupper() and len(alias) <= 6:
                patterns.append(
                    re.compile(rf"(?<![A-Za-z0-9])\$?{re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)
                )
            else:
                patterns.append(re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE))

        return patterns

    def _sentence_mentions_ticker(self, sentence: str, ticker: str) -> bool:
        for pat in self._ticker_to_patterns[ticker]:
            if pat.search(sentence):
                return True
        return False

    def _extract_mentioned_tickers(self, sentence: str) -> List[str]:
        tickers = []
        for ticker in self.supported_tickers:
            if self._sentence_mentions_ticker(sentence, ticker):
                tickers.append(ticker)
        return tickers

    def _count_ticker_mentions(self, text: str, ticker: str) -> int:
        count = 0
        for pat in self._ticker_to_patterns[ticker]:
            count += len(pat.findall(text))
        return count

    def _make_query_text(self, ticker: str) -> str:
        aliases = self.alias_map.get(ticker.upper(), [ticker.upper()])
        unique_aliases = []
        seen = set()

        for alias in aliases:
            norm = alias.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                unique_aliases.append(alias)

        query_terms = " ".join(unique_aliases)
        return self._normalize_for_vectorizer(query_terms)

    def _build_sentence_corpus(self) -> pd.DataFrame:
        records = []

        for idx, row in self.df.iterrows():
            title = self._clean_text(row["title"])
            body = self._clean_text(row["body"])
            url = str(row["url"]).strip()
            score = float(row["score"])
            comms_num = float(row["comms_num"])

            title_sentences = self._split_sentences(title)
            body_sentences = self._split_sentences(body)

            all_sentences = []
            for sent in title_sentences:
                all_sentences.append(("title", sent))
            for sent in body_sentences:
                all_sentences.append(("body", sent))

            if not all_sentences:
                continue

            for source_part, sentence in all_sentences:
                mentioned = self._extract_mentioned_tickers(sentence)
                if not mentioned:
                    continue

                vector_text = self._normalize_for_vectorizer(sentence)
                if not vector_text:
                    continue

                full_post_text = f"{title} {body}".strip()

                records.append(
                    {
                        "post_id": int(idx),
                        "source_part": source_part,
                        "sentence": sentence,
                        "vector_text": vector_text,
                        "title": title,
                        "body": body,
                        "full_post_text": full_post_text,
                        "mentioned_tickers": mentioned,
                        "score": score,
                        "comms_num": comms_num,
                        "url": url,
                        "time_value": row[self.time_col] if self.time_col else None,
                    }
                )

        sentence_df = pd.DataFrame(records)

        if sentence_df.empty:
            return sentence_df

        sentence_df = sentence_df.drop_duplicates(
            subset=["post_id", "sentence"]
        ).reset_index(drop=True)

        return sentence_df

    def _fit_retrieval_models(self):
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.92,
            sublinear_tf=True,
        )

        doc_tfidf = vectorizer.fit_transform(self.sentence_df["vector_text"])

        n_features = doc_tfidf.shape[1]
        n_docs = doc_tfidf.shape[0]
        svd_dim = max(2, min(100, n_features - 1, n_docs - 1))

        if svd_dim < 2:
            svd_dim = 2

        svd_model = make_pipeline(
            TruncatedSVD(n_components=svd_dim, random_state=42),
            Normalizer(copy=False),
        )

        doc_lsi = svd_model.fit_transform(doc_tfidf)

        return vectorizer, doc_tfidf, svd_model, doc_lsi

    def _text_sentiment(self, text: str):
        text = self._normalize_for_vectorizer(text)
        score = 0.0
        matched_terms = 0

        for phrase, value in self.phrase_scores.items():
            if phrase in text:
                score += value
                matched_terms += 1

        tokens = text.split()

        for i, tok in enumerate(tokens):
            if tok not in self.word_scores:
                continue

            val = self.word_scores[tok]
            window_start = max(0, i - 3)
            window = tokens[window_start:i]

            if any(w in self.negations for w in window):
                val *= -0.8

            score += val
            matched_terms += 1

        exclam_bonus = min(text.count("!"), 4) * 0.08
        score *= 1.0 + exclam_bonus

        normalized = math.tanh(score / 4.0)
        confidence = min(1.0, 0.25 + 0.18 * matched_terms + 0.35 * abs(normalized))

        return float(normalized), float(confidence), int(matched_terms)

    def _engagement_weight(self, row: pd.Series) -> float:
        score_part = 0.22 * math.log1p(max(float(row["score"]), 0))
        comment_part = 0.15 * math.log1p(max(float(row["comms_num"]), 0))
        return 1.0 + score_part + comment_part

    def _mention_prominence(self, row: pd.Series, ticker: str) -> float:
        sentence_hits = self._count_ticker_mentions(row["sentence"], ticker)
        title_hits = self._count_ticker_mentions(row["title"], ticker)
        part_bonus = 0.25 if row["source_part"] == "title" else 0.0

        prominence = 1.0
        prominence += 0.18 * min(sentence_hits, 3)
        prominence += 0.10 * min(title_hits, 2)
        prominence += part_bonus
        return prominence

    def _recency_weight(self, row: pd.Series) -> float:
        if self.time_col is None:
            return 1.0

        val = row["time_value"]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 1.0

        try:
            if self.time_col == "created_utc":
                ts = pd.to_datetime(float(val), unit="s", utc=True)
            else:
                ts = pd.to_datetime(val, utc=True, errors="coerce")
            if pd.isna(ts):
                return 1.0
        except Exception:
            return 1.0

        newest = None
        try:
            if self.time_col == "created_utc":
                newest = pd.to_datetime(
                    pd.to_numeric(self.df[self.time_col], errors="coerce").max(),
                    unit="s",
                    utc=True,
                )
            else:
                newest = pd.to_datetime(self.df[self.time_col], utc=True, errors="coerce").max()
        except Exception:
            return 1.0

        if newest is None or pd.isna(newest):
            return 1.0

        age_days = max((newest - ts).total_seconds() / 86400.0, 0.0)
        return 1.0 / (1.0 + 0.015 * age_days)

    def _retrieve_sentences_for_ticker(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.upper().strip()
        if ticker not in self.alias_map:
            return pd.DataFrame()

        subset = self.sentence_df[
            self.sentence_df["mentioned_tickers"].apply(lambda xs: ticker in xs)
        ].copy()

        if subset.empty:
            return subset

        query_text = self._make_query_text(ticker)
        query_tfidf = self.vectorizer.transform([query_text])
        query_lsi = self.svd_model.transform(query_tfidf)

        tfidf_sim_all = cosine_similarity(query_tfidf, self.doc_tfidf).flatten()
        lsi_sim_all = cosine_similarity(query_lsi, self.doc_lsi).flatten()

        subset["tfidf_similarity"] = tfidf_sim_all[subset.index]
        subset["svd_similarity"] = lsi_sim_all[subset.index]

        subset["retrieval_score"] = (
            0.55 * subset["tfidf_similarity"] + 0.45 * subset["svd_similarity"]
        )

        return subset.sort_values("retrieval_score", ascending=False)

    def _format_sentence_for_display(self, sentence: str) -> str:
        sentence = self._clean_text(sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if len(sentence) > 320:
            sentence = sentence[:317].rstrip() + "..."
        return sentence

    def analyze_ticker(self, ticker: str, top_k: int = 5) -> Dict[str, Any]:
        ticker = str(ticker).upper().strip()

        if ticker not in self.alias_map:
            return {
                "ticker": ticker,
                "label": "Unknown ticker",
                "score_0_to_100": None,
                "avg_sentiment": None,
                "posts_used": 0,
                "top_posts": [],
                "methodology": self.get_methodology(),
            }

        candidates = self._retrieve_sentences_for_ticker(ticker)

        if candidates.empty:
            return {
                "ticker": ticker,
                "label": "No data",
                "score_0_to_100": None,
                "avg_sentiment": None,
                "posts_used": 0,
                "top_posts": [],
                "methodology": self.get_methodology(),
            }

        sentiment_triplets = candidates["sentence"].apply(self._text_sentiment)
        candidates["sentence_sentiment"] = sentiment_triplets.apply(lambda x: x[0])
        candidates["sentiment_confidence"] = sentiment_triplets.apply(lambda x: x[1])
        candidates["lexicon_hits"] = sentiment_triplets.apply(lambda x: x[2])

        candidates["engagement_weight"] = candidates.apply(self._engagement_weight, axis=1)
        candidates["mention_prominence"] = candidates.apply(
            lambda r: self._mention_prominence(r, ticker), axis=1
        )
        candidates["recency_weight"] = candidates.apply(self._recency_weight, axis=1)

        candidates["evidence_weight"] = (
            (0.40 + 0.60 * candidates["retrieval_score"].clip(lower=0.0)) *
            candidates["engagement_weight"] *
            candidates["mention_prominence"] *
            candidates["recency_weight"] *
            (0.70 + 0.30 * candidates["sentiment_confidence"])
        )

        candidates["weighted_contribution"] = (
            candidates["sentence_sentiment"] * candidates["evidence_weight"]
        )

        total_weight = candidates["evidence_weight"].sum()
        avg_sentiment = (
            candidates["weighted_contribution"].sum() / total_weight if total_weight > 0 else 0.0
        )

        sentiment_variance = np.average(
            (candidates["sentence_sentiment"] - avg_sentiment) ** 2,
            weights=np.maximum(candidates["evidence_weight"], 1e-8),
        )
        disagreement_penalty = min(0.18, 0.35 * math.sqrt(max(sentiment_variance, 0.0)))

        coverage = min(1.0, math.log1p(len(candidates)) / math.log(12))
        average_retrieval = float(candidates["retrieval_score"].clip(lower=0).mean())
        average_confidence = float(candidates["sentiment_confidence"].mean())

        raw_stock_score = (
            0.58 * avg_sentiment +
            0.14 * (coverage - 0.5) +
            0.16 * (average_retrieval - 0.35) +
            0.12 * (average_confidence - 0.5) -
            disagreement_penalty
        )

        raw_stock_score = max(-1.0, min(1.0, raw_stock_score))
        score_0_to_100 = round(50 + 50 * raw_stock_score, 1)

        if raw_stock_score > 0.18:
            label = "Bullish"
        elif raw_stock_score < -0.18:
            label = "Bearish"
        else:
            label = "Neutral"

        candidates["display_rank"] = (
            0.45 * candidates["retrieval_score"] +
            0.20 * candidates["engagement_weight"] / candidates["engagement_weight"].max() +
            0.20 * candidates["mention_prominence"] / candidates["mention_prominence"].max() +
            0.15 * candidates["sentiment_confidence"]
        )

        display_cols = [
            "title",
            "sentence",
            "score",
            "comms_num",
            "url",
            "sentence_sentiment",
            "sentiment_confidence",
            "retrieval_score",
            "tfidf_similarity",
            "svd_similarity",
            "mention_prominence",
            "engagement_weight",
            "recency_weight",
            "source_part",
        ]

        top_posts_df = candidates.sort_values("display_rank", ascending=False).head(top_k).copy()

        top_posts = []
        for _, row in top_posts_df[display_cols].iterrows():
            top_posts.append(
                {
                    "title": self._format_sentence_for_display(row["title"]) or "(No title)",
                    "snippet": self._format_sentence_for_display(row["sentence"]),
                    "score": int(row["score"]) if not pd.isna(row["score"]) else 0,
                    "comms_num": int(row["comms_num"]) if not pd.isna(row["comms_num"]) else 0,
                    "url": row["url"],
                    "sentence_sentiment": round(float(row["sentence_sentiment"]), 4),
                    "sentiment_confidence": round(float(row["sentiment_confidence"]), 4),
                    "retrieval_score": round(float(row["retrieval_score"]), 4),
                    "tfidf_similarity": round(float(row["tfidf_similarity"]), 4),
                    "svd_similarity": round(float(row["svd_similarity"]), 4),
                    "mention_prominence": round(float(row["mention_prominence"]), 4),
                    "engagement_weight": round(float(row["engagement_weight"]), 4),
                    "recency_weight": round(float(row["recency_weight"]), 4),
                    "source_part": row["source_part"],
                }
            )

        return {
            "ticker": ticker,
            "label": label,
            "score_0_to_100": float(score_0_to_100),
            "avg_sentiment": round(float(avg_sentiment), 4),
            "raw_stock_score": round(float(raw_stock_score), 4),
            "posts_used": int(len(candidates)),
            "summary_metrics": {
                "coverage": round(float(coverage), 4),
                "average_retrieval": round(float(average_retrieval), 4),
                "average_confidence": round(float(average_confidence), 4),
                "disagreement_penalty": round(float(disagreement_penalty), 4),
            },
            "top_posts": top_posts,
            "methodology": self.get_methodology(),
        }

    def rank_all_tickers(self, top_k: int = 25) -> pd.DataFrame:
        rows = []

        for ticker in self.supported_tickers:
            result = self.analyze_ticker(ticker, top_k=3)
            if result["posts_used"] > 0 and result["score_0_to_100"] is not None:
                rows.append(
                    {
                        "ticker": result["ticker"],
                        "label": result["label"],
                        "score_0_to_100": result["score_0_to_100"],
                        "avg_sentiment": result["avg_sentiment"],
                        "raw_stock_score": result["raw_stock_score"],
                        "posts_used": result["posts_used"],
                        "coverage": result["summary_metrics"]["coverage"],
                        "average_retrieval": result["summary_metrics"]["average_retrieval"],
                        "average_confidence": result["summary_metrics"]["average_confidence"],
                        "disagreement_penalty": result["summary_metrics"]["disagreement_penalty"],
                    }
                )

        if not rows:
            return pd.DataFrame()

        ranked = pd.DataFrame(rows).sort_values(
            by=["score_0_to_100", "posts_used", "average_retrieval"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        return ranked.head(top_k)

    def get_methodology(self) -> Dict[str, Any]:
        return {
            "retrieval": {
                "summary": (
                    "StockPulse retrieves evidence with a hybrid semantic search system. "
                    "Each Reddit post is split into sentences, only sentences that actually mention a supported stock "
                    "ticker or alias are kept, and those sentences are indexed with TF-IDF and Truncated SVD."
                ),
                "details": [
                    "TF-IDF captures exact lexical relevance between the stock query and the sentence.",
                    "Truncated SVD captures latent semantic similarity beyond exact word overlap.",
                    "The final retrieval score is 55% TF-IDF similarity and 45% SVD similarity."
                ],
            },
            "sentiment": {
                "summary": (
                    "Sentiment is measured on the sentence containing the stock mention, not on the entire Reddit post."
                ),
                "details": [
                    "A finance-oriented lexicon scores bullish and bearish words and phrases.",
                    "Negation handling reduces errors for phrases such as 'not bullish' or 'not a buy'.",
                    "A confidence score increases when more finance-relevant sentiment evidence appears."
                ],
            },
            "stock_score": {
                "summary": (
                    "The final stock score is a weighted aggregation of sentence-level evidence."
                ),
                "details": [
                    "Evidence weight increases with retrieval relevance, engagement, prominence of the ticker mention, recency if available, and sentiment confidence.",
                    "Average sentiment is adjusted by coverage, average retrieval strength, average confidence, and a disagreement penalty.",
                    "The score is mapped to 0-100, where above 50 is net positive, below 50 is net negative."
                ],
                "formula": {
                    "sentence_evidence_weight": (
                        "(0.40 + 0.60 * retrieval_score) * engagement_weight * mention_prominence "
                        "* recency_weight * (0.70 + 0.30 * sentiment_confidence)"
                    ),
                    "retrieval_score": "0.55 * tfidf_similarity + 0.45 * svd_similarity",
                    "stock_raw_score": (
                        "0.58 * avg_sentiment + 0.14 * (coverage - 0.5) + "
                        "0.16 * (average_retrieval - 0.35) + 0.12 * (average_confidence - 0.5) "
                        "- disagreement_penalty"
                    ),
                    "final_score_0_to_100": "50 + 50 * stock_raw_score",
                },
            },
        }


ALIAS_MAP = {
    "AAPL": ["AAPL", "$AAPL", "Apple"],
    "MSFT": ["MSFT", "$MSFT", "Microsoft"],
    "AMZN": ["AMZN", "$AMZN", "Amazon"],
    "GOOGL": ["GOOGL", "$GOOGL", "Alphabet", "Google"],
    "GOOG": ["GOOG", "$GOOG", "Alphabet", "Google"],
    "META": ["META", "$META", "Meta", "Facebook"],
    "TSLA": ["TSLA", "$TSLA", "Tesla"],
    "NVDA": ["NVDA", "$NVDA", "Nvidia"],
    "AMD": ["AMD", "$AMD", "Advanced Micro Devices"],
    "INTC": ["INTC", "$INTC", "Intel"],
    "ORCL": ["ORCL", "$ORCL", "Oracle"],
    "CRM": ["CRM", "$CRM", "Salesforce"],
    "ADBE": ["ADBE", "$ADBE", "Adobe"],
    "QCOM": ["QCOM", "$QCOM", "Qualcomm"],
    "JPM": ["JPM", "$JPM", "JPMorgan"],
    "BAC": ["BAC", "$BAC", "Bank of America"],
    "WFC": ["WFC", "$WFC", "Wells Fargo"],
    "GS": ["GS", "$GS", "Goldman Sachs"],
    "MS": ["MS", "$MS", "Morgan Stanley"],
    "DIS": ["DIS", "$DIS", "Disney"],
    "NKE": ["NKE", "$NKE", "Nike"],
    "WMT": ["WMT", "$WMT", "Walmart"],
    "TGT": ["TGT", "$TGT", "Target"],
    "COST": ["COST", "$COST", "Costco"],
    "SBUX": ["SBUX", "$SBUX", "Starbucks"],
    "PYPL": ["PYPL", "$PYPL", "PayPal"],
    "SQ": ["SQ", "$SQ", "Block", "Square"],
    "UBER": ["UBER", "$UBER", "Uber"],
    "LYFT": ["LYFT", "$LYFT", "Lyft"],
    "SHOP": ["SHOP", "$SHOP", "Shopify"],
    "SPY": ["SPY", "$SPY", "S&P 500", "sp500"],
}


def get_default_csv_path():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "prototype_posts.csv")


@lru_cache(maxsize=1)
def get_default_model():
    return StockPulseSentiment(get_default_csv_path(), ALIAS_MAP)