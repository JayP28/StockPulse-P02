import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_TOKEN_RE = re.compile(r"[^A-Za-z0-9$._+\-\s]")
WHITESPACE_RE = re.compile(r"\s+")
NUMERIC_TERM_RE = re.compile(r"^\d+(?:\.\d+)?$")
DISPLAY_STOP_TERMS = {
    "amp",
    "com",
    "don",
    "don t",
    "doesn",
    "doesn t",
    "http",
    "https",
    "https www",
    "ll",
    "re",
    "t",
    "ve",
    "u",
    "www",
}


def _build_methodology(dataset_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    methodology = {
        "baseline": {
            "summary": (
                "Baseline retrieval uses TF-IDF over the combined post text and ranks documents with cosine similarity."
            ),
            "details": [
                "The searchable field combines title, body, and a truncated comments excerpt so the system can be tuned later.",
                "Ticker-like tokens such as AAPL or $NVDA are preserved during preprocessing.",
                "Cosine similarity in TF-IDF space rewards strong lexical overlap between the query and post text.",
            ],
        },
        "svd": {
            "summary": (
                "SVD retrieval projects TF-IDF vectors into a lower-dimensional latent space before ranking with cosine similarity."
            ),
            "details": [
                "TruncatedSVD captures broader co-occurrence structure so posts can match even when they do not share all of the same surface words.",
                "This latent-space ranking is useful for comparing exact-term retrieval against semantic retrieval in the milestone demo.",
                "The current implementation retrieves directly in SVD space, but the structure also supports reranking later.",
            ],
        },
        "explainability": {
            "summary": (
                "For each query, StockPulse surfaces the most activated SVD dimensions and shows the terms on each side of those dimensions."
            ),
            "details": [
                "Top positive and top negative terms are read from each SVD component.",
                "A short label summarizes the active side of the dimension versus the opposite side.",
                "Each retrieved result reports which of the important SVD dimensions it aligns with most strongly.",
            ],
        },
    }

    if dataset_summary is not None:
        methodology["dataset"] = dataset_summary

    return methodology


class StockPulseRetrieval:
    """
    Retrieval model for the StockPulse milestone.

    This version focuses on post retrieval only:
    - baseline ad-hoc IR with TF-IDF + cosine similarity
    - latent semantic retrieval with TruncatedSVD
    - light SVD explainability for screenshots and analysis
    """

    def __init__(
        self,
        csv_path: str,
        *,
        include_comments: bool = True,
        comment_char_limit: int = 900,
        max_features: int = 35000,
        max_svd_components: int = 120,
    ):
        self.csv_path = csv_path
        self.include_comments = include_comments
        self.comment_char_limit = comment_char_limit

        use_columns = {
            "title",
            "body",
            "comments_text",
            "url",
            "score",
            "comms_num",
            "datetime",
            "tag",
        }

        self.df = pd.read_csv(
            csv_path,
            usecols=lambda column: column in use_columns,
            low_memory=False,
        )

        for column in ["title", "body", "comments_text", "url", "datetime", "tag"]:
            if column not in self.df.columns:
                self.df[column] = ""
            self.df[column] = self.df[column].fillna("").astype(str)

        if "score" not in self.df.columns:
            self.df["score"] = 0
        if "comms_num" not in self.df.columns:
            self.df["comms_num"] = 0

        self.df["score"] = pd.to_numeric(self.df["score"], errors="coerce").fillna(0)
        self.df["comms_num"] = pd.to_numeric(self.df["comms_num"], errors="coerce").fillna(0)

        search_texts: List[str] = []
        previews: List[str] = []

        for row in self.df.itertuples(index=False):
            search_text = self._compose_search_text(row)
            search_texts.append(search_text)
            previews.append(self._build_preview(row))

        non_empty_mask = np.array([bool(text.strip()) for text in search_texts], dtype=bool)
        self.df = self.df.loc[non_empty_mask].reset_index(drop=True)
        self.df["preview"] = [preview for preview, keep in zip(previews, non_empty_mask) if keep]
        self.search_texts = [text for text, keep in zip(search_texts, non_empty_mask) if keep]

        if not self.search_texts:
            raise ValueError("The retrieval corpus is empty after preprocessing.")

        # Preserve ticker-like tokens and finance terms while still normalizing noisy Reddit text.
        self.vectorizer = TfidfVectorizer(
            preprocessor=self._preprocess_text,
            lowercase=False,
            stop_words="english",
            token_pattern=r"(?u)(?<!\w)[$A-Za-z0-9][A-Za-z0-9$._+-]{0,19}(?!\w)",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )

        self.doc_tfidf = self.vectorizer.fit_transform(self.search_texts)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

        n_docs = self.doc_tfidf.shape[0]
        n_features = self.doc_tfidf.shape[1]
        svd_components = max(2, min(max_svd_components, n_docs - 1, n_features - 1))

        if svd_components < 2:
            raise ValueError("Not enough data to build the SVD retrieval model.")

        self.svd = TruncatedSVD(
            n_components=svd_components,
            random_state=42,
            n_iter=8,
        )
        self.normalizer = Normalizer(copy=False)

        # Keep both raw and normalized latent vectors so we can retrieve and explain results.
        self.doc_lsi_raw = self.svd.fit_transform(self.doc_tfidf).astype(np.float32)
        self.doc_lsi = self.normalizer.fit_transform(self.doc_lsi_raw.copy()).astype(np.float32)

    def _clean_value(self, value: Any) -> str:
        return str(value or "").strip()

    def _truncate(self, text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text

        clipped = text[: limit - 3]
        last_space = clipped.rfind(" ")
        if last_space >= limit // 2:
            clipped = clipped[:last_space]
        return clipped.rstrip() + "..."

    def _compose_search_text(self, row: Any) -> str:
        parts = []

        title = self._clean_value(row.title)
        body = self._clean_value(row.body)
        comments_text = self._clean_value(getattr(row, "comments_text", ""))

        if title:
            parts.append(title)
        if body:
            parts.append(body)
        if self.include_comments and comments_text:
            parts.append(self._truncate(comments_text, self.comment_char_limit))

        return "\n".join(part for part in parts if part)

    def _build_preview(self, row: Any) -> str:
        body = self._clean_value(row.body)
        comments_text = self._clean_value(getattr(row, "comments_text", ""))

        preview_source = body or comments_text or self._clean_value(row.title)
        preview_source = WHITESPACE_RE.sub(" ", preview_source).strip()
        return self._truncate(preview_source, 280)

    def _preprocess_text(self, text: str) -> str:
        text = str(text or "")
        text = URL_RE.sub(" ", text)
        text = text.replace("&amp;", "and")
        text = text.lower()
        text = NON_TOKEN_RE.sub(" ", text)
        text = WHITESPACE_RE.sub(" ", text).strip()
        return text

    def _top_indices(self, scores: np.ndarray, top_k: int) -> np.ndarray:
        if scores.size == 0:
            return np.array([], dtype=int)

        top_k = max(1, min(int(top_k), scores.size))
        if top_k == scores.size:
            return np.argsort(scores)[::-1]

        candidate_idx = np.argpartition(scores, -top_k)[-top_k:]
        return candidate_idx[np.argsort(scores[candidate_idx])[::-1]]

    def _dimension_summary(self, active_terms: List[str], opposing_terms: List[str]) -> str:
        active = ", ".join(active_terms[:3]) or "query terms"
        opposing = ", ".join(opposing_terms[:3]) or "other discussion"
        return f"{active} vs {opposing}"

    def _dimension_interpretation(
        self,
        active_terms: List[str],
        opposing_terms: List[str],
        query_pole: str,
    ) -> str:
        active = ", ".join(active_terms[:3]) or "the active terms"
        opposing = ", ".join(opposing_terms[:3]) or "the opposite side"
        return (
            f"The query leans on the {query_pole} side of this latent factor, which looks related "
            f"to {active} and contrasted with {opposing}."
        )

    def _is_informative_dimension_term(self, term: str) -> bool:
        if not term:
            return False
        if term in DISPLAY_STOP_TERMS:
            return False
        if term.startswith("http"):
            return False
        if NUMERIC_TERM_RE.fullmatch(term):
            return False
        if len(term) == 1:
            return False
        return True

    def _select_component_terms(
        self,
        component: np.ndarray,
        *,
        descending: bool,
        top_terms: int,
    ) -> List[str]:
        ordered_indices = np.argsort(component)[::-1] if descending else np.argsort(component)
        selected: List[str] = []

        for index in ordered_indices:
            term = str(self.feature_names[index])
            if term in selected:
                continue
            if not self._is_informative_dimension_term(term):
                continue
            selected.append(term)
            if len(selected) == top_terms:
                break

        return selected

    def _build_dimension_explanations(
        self,
        query_lsi_raw: np.ndarray,
        *,
        top_n: int = 4,
        top_terms: int = 6,
    ) -> List[Dict[str, Any]]:
        if query_lsi_raw.size == 0:
            return []

        important_dims = np.argsort(np.abs(query_lsi_raw))[-top_n:][::-1]
        explanations: List[Dict[str, Any]] = []

        for dim_idx in important_dims:
            component = self.svd.components_[dim_idx]
            positive_terms = self._select_component_terms(
                component,
                descending=True,
                top_terms=top_terms,
            )
            negative_terms = self._select_component_terms(
                component,
                descending=False,
                top_terms=top_terms,
            )

            query_pole = "positive" if query_lsi_raw[dim_idx] >= 0 else "negative"
            active_terms = positive_terms if query_pole == "positive" else negative_terms
            opposing_terms = negative_terms if query_pole == "positive" else positive_terms

            explanations.append(
                {
                    "dimension": int(dim_idx + 1),
                    "index": int(dim_idx),
                    "query_strength": round(float(query_lsi_raw[dim_idx]), 4),
                    "query_pole": query_pole,
                    "short_label": self._dimension_summary(active_terms, opposing_terms),
                    "interpretation": self._dimension_interpretation(
                        active_terms, opposing_terms, query_pole
                    ),
                    "active_terms": active_terms,
                    "opposing_terms": opposing_terms,
                    "top_positive_terms": positive_terms,
                    "top_negative_terms": negative_terms,
                }
            )

        return explanations

    def _result_dimension_alignment(
        self,
        doc_index: int,
        query_lsi_raw: np.ndarray,
        important_dimensions: List[Dict[str, Any]],
        *,
        max_dims: int = 3,
    ) -> List[Dict[str, Any]]:
        doc_vector = self.doc_lsi_raw[doc_index]
        alignments: List[Dict[str, Any]] = []

        for dimension in important_dimensions:
            dim_idx = dimension["index"]
            query_value = float(query_lsi_raw[dim_idx])
            doc_value = float(doc_vector[dim_idx])
            alignment = query_value * doc_value

            if alignment <= 0:
                continue

            alignments.append(
                {
                    "dimension": dimension["dimension"],
                    "alignment": round(float(alignment), 4),
                    "pole": "positive" if doc_value >= 0 else "negative",
                    "short_label": dimension["short_label"],
                }
            )

        if not alignments:
            fallback = sorted(
                [
                    {
                        "dimension": dimension["dimension"],
                        "alignment": round(
                            float(abs(query_lsi_raw[dimension["index"]] * doc_vector[dimension["index"]])),
                            4,
                        ),
                        "pole": "positive" if float(doc_vector[dimension["index"]]) >= 0 else "negative",
                        "short_label": dimension["short_label"],
                    }
                    for dimension in important_dimensions
                ],
                key=lambda item: item["alignment"],
                reverse=True,
            )
            return fallback[:max_dims]

        return sorted(alignments, key=lambda item: item["alignment"], reverse=True)[:max_dims]

    def _format_results(
        self,
        scores: np.ndarray,
        *,
        top_k: int,
        query_lsi_raw: np.ndarray,
        important_dimensions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for rank, doc_index in enumerate(self._top_indices(scores, top_k), start=1):
            row = self.df.iloc[doc_index]
            results.append(
                {
                    "rank": rank,
                    "title": self._clean_value(row["title"]) or "(No title)",
                    "preview": self._clean_value(row["preview"]),
                    "similarity": round(float(scores[doc_index]), 4),
                    "url": self._clean_value(row["url"]),
                    "score": int(row["score"]) if not pd.isna(row["score"]) else 0,
                    "comms_num": int(row["comms_num"]) if not pd.isna(row["comms_num"]) else 0,
                    "datetime": self._clean_value(row["datetime"]),
                    "tag": self._clean_value(row["tag"]),
                    "aligned_dimensions": self._result_dimension_alignment(
                        doc_index,
                        query_lsi_raw,
                        important_dimensions,
                    ),
                }
            )

        return results

    def search(self, query: str, *, top_k: int = 5) -> Dict[str, Any]:
        query = str(query or "").strip()
        if not query:
            raise ValueError("query is required")

        query_tfidf = self.vectorizer.transform([query])
        if query_tfidf.nnz == 0:
            raise ValueError(
                "No query terms matched the indexed vocabulary. Try a ticker like NVDA or a phrase like 'bank earnings'."
            )

        baseline_scores = cosine_similarity(query_tfidf, self.doc_tfidf).flatten()

        query_lsi_raw = self.svd.transform(query_tfidf).astype(np.float32)
        query_lsi = self.normalizer.transform(query_lsi_raw.copy()).astype(np.float32)
        svd_scores = cosine_similarity(query_lsi, self.doc_lsi).flatten()

        upvote_boost = np.log1p(self.df["score"].clip(lower=0).to_numpy(dtype=np.float32))

        UPVOTE_WEIGHT = 0.03

        baseline_scores = baseline_scores + UPVOTE_WEIGHT * upvote_boost
        svd_scores = svd_scores + UPVOTE_WEIGHT * upvote_boost


        important_dimensions = self._build_dimension_explanations(query_lsi_raw[0])

        return {
            "query": query,
            "baseline_results": self._format_results(
                baseline_scores,
                top_k=top_k,
                query_lsi_raw=query_lsi_raw[0],
                important_dimensions=important_dimensions,
            ),
            "svd_results": self._format_results(
                svd_scores,
                top_k=top_k,
                query_lsi_raw=query_lsi_raw[0],
                important_dimensions=important_dimensions,
            ),
            "svd_explainability": {
                "summary": (
                    "The query is projected into latent semantic dimensions learned from the Reddit corpus. "
                    "The dimensions below have the strongest absolute query activation."
                ),
                "important_dimensions": important_dimensions,
            },
            "stats": self.get_corpus_summary(),
        }

    def get_methodology(self) -> Dict[str, Any]:
        return _build_methodology(self.get_corpus_summary())

    def get_corpus_summary(self) -> Dict[str, Any]:
        return {
            "document_count": int(self.doc_tfidf.shape[0]),
            "vocabulary_size": int(self.doc_tfidf.shape[1]),
            "svd_components": int(self.doc_lsi_raw.shape[1]),
            "search_field": "title + body + comments_text excerpt",
            "data_path": self.csv_path,
        }


#def get_default_csv_path() -> str:
#    env_path = os.getenv("STOCKPULSE_DATA_PATH")
#    if env_path:
#        return env_path

#    current_directory = os.path.dirname(os.path.abspath(__file__))
#    repo_root = os.path.dirname(current_directory)

#    cleaned_threads_path = os.path.join(repo_root, "cleaned_threads_cut.csv")
#    if os.path.exists(cleaned_threads_path):
#        return cleaned_threads_path

   # return os.path.join(current_directory, "prototype_posts.csv")

def get_default_csv_path() -> str:
    if os.path.exists("cleaned_threads_cut.csv"):
        return "cleaned_threads_cut.csv"

    if os.path.exists("prototype_posts.csv"):
        return "prototype_posts.csv"

    raise FileNotFoundError(
        f"No dataset found. Files: {os.listdir(os.getcwd())}"
    )


@lru_cache(maxsize=1)
def get_default_retriever() -> StockPulseRetrieval:
    return StockPulseRetrieval(get_default_csv_path())


def get_methodology_overview() -> Dict[str, Any]:
    return _build_methodology()
