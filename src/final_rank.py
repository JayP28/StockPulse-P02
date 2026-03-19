import math
import os
import re
from functools import lru_cache

import pandas as pd


class StockPulseSentiment:
    def __init__(self, csv_path, alias_map):
        self.df = pd.read_csv(csv_path)
        print("DEBUG csv_path:", csv_path)
        print("DEBUG file exists:", os.path.exists(csv_path))
        print("DEBUG loaded rows:", len(self.df))
        print("DEBUG columns:", list(self.df.columns))
        self.alias_map = {k.upper(): v for k, v in alias_map.items()}

        if "title" not in self.df.columns:
            self.df["title"] = ""
        if "body" not in self.df.columns:
            self.df["body"] = ""
        if "url" not in self.df.columns:
            self.df["url"] = ""

        self.df["title"] = self.df["title"].fillna("").astype(str)
        self.df["body"] = self.df["body"].fillna("").astype(str)
        self.df["score"] = pd.to_numeric(
            self.df.get("score", 0), errors="coerce"
        ).fillna(0)
        self.df["comms_num"] = pd.to_numeric(
            self.df.get("comms_num", 0), errors="coerce"
        ).fillna(0)

        self.df["combined_text"] = (
            self.df["title"] + " " + self.df["body"]
        ).str.strip()

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
        }

        self.negations = {
            "not",
            "no",
            "never",
            "isnt",
            "wasnt",
            "dont",
            "doesnt",
            "cant",
            "wont",
        }

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = text.replace("🚀", " rocket ")
        text = text.replace("🟢", " green ")
        text = text.replace("🔴", " red ")
        text = re.sub(r"[^a-z0-9$!\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _ticker_patterns(self, ticker):
        aliases = self.alias_map.get(ticker.upper(), [ticker.upper()])
        patterns = []

        for alias in aliases:
            alias = alias.strip()
            if not alias:
                continue

            if alias.startswith("$"):
                raw = alias[1:]
                patterns.append(
                    re.compile(
                        rf"(?<![A-Z])\$?{re.escape(raw)}(?![A-Z])",
                        re.IGNORECASE,
                    )
                )
            elif alias.isupper() and len(alias) <= 5:
                patterns.append(
                    re.compile(
                        rf"(?<![A-Z])\$?{re.escape(alias)}(?![A-Z])",
                        re.IGNORECASE,
                    )
                )
            else:
                patterns.append(
                    re.compile(rf"\b{re.escape(alias.lower())}\b", re.IGNORECASE)
                )

        return patterns

    def _matches_ticker(self, row, ticker):
        patterns = self._ticker_patterns(ticker)
        for pat in patterns:
            if pat.search(row["title"]) or pat.search(row["body"]):
                return True
        return False

    def _ticker_relevance(self, row, ticker):
        patterns = self._ticker_patterns(ticker)
        title_hits = 0
        body_hits = 0

        for pat in patterns:
            if pat.search(row["title"]):
                title_hits += 1
            if pat.search(row["body"]):
                body_hits += 1

        if title_hits == 0 and body_hits == 0:
            return 0.0

        relevance = 1.0
        if title_hits > 0:
            relevance += 0.8
        if body_hits > 0:
            relevance += 0.4
        relevance += 0.15 * min(title_hits + body_hits, 3)
        return relevance

    def _text_sentiment(self, text):
        text = self._clean_text(text)
        score = 0.0

        for phrase, value in self.phrase_scores.items():
            if phrase in text:
                score += value

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

        exclam_bonus = min(text.count("!"), 4) * 0.08
        score *= 1.0 + exclam_bonus

        return math.tanh(score / 4.0)

    def _engagement_weight(self, row):
        score_part = 0.18 * math.log1p(max(row["score"], 0))
        comment_part = 0.12 * math.log1p(max(row["comms_num"], 0))
        return 1.0 + score_part + comment_part

    def analyze_ticker(self, ticker, top_k=5):
        ticker = str(ticker).upper().strip()

        if ticker not in self.alias_map:
            return {
                "ticker": ticker,
                "label": "Unknown ticker",
                "score_0_to_100": None,
                "avg_sentiment": None,
                "posts_used": 0,
                "top_posts": [],
            }

        candidates = self.df[
            self.df.apply(lambda r: self._matches_ticker(r, ticker), axis=1)
        ].copy()

        if candidates.empty:
            return {
                "ticker": ticker,
                "label": "No data",
                "score_0_to_100": None,
                "avg_sentiment": None,
                "posts_used": 0,
                "top_posts": [],
            }

        candidates["ticker_relevance"] = candidates.apply(
            lambda r: self._ticker_relevance(r, ticker), axis=1
        )
        candidates["text_sentiment"] = candidates["combined_text"].apply(
            self._text_sentiment
        )
        candidates["engagement_weight"] = candidates.apply(
            self._engagement_weight, axis=1
        )

        candidates["final_weight"] = (
            candidates["ticker_relevance"] * candidates["engagement_weight"]
        )
        candidates["contribution"] = (
            candidates["text_sentiment"] * candidates["final_weight"]
        )

        total_weight = candidates["final_weight"].sum()
        avg_sentiment = (
            candidates["contribution"].sum() / total_weight
            if total_weight > 0
            else 0.0
        )

        if avg_sentiment > 0.15:
            label = "Bullish"
        elif avg_sentiment < -0.15:
            label = "Bearish"
        else:
            label = "Neutral"

        score_0_to_100 = round(50 + 50 * avg_sentiment, 1)

        candidates["display_rank"] = (
            0.55 * candidates["ticker_relevance"]
            + 0.30 * candidates["engagement_weight"]
            + 0.15 * candidates["text_sentiment"].abs()
        )

        top_posts = candidates.sort_values(
            "display_rank", ascending=False
        ).head(top_k)

        return {
            "ticker": ticker,
            "label": label,
            "score_0_to_100": float(score_0_to_100),
            "avg_sentiment": round(float(avg_sentiment), 4),
            "posts_used": int(len(candidates)),
            "top_posts": top_posts[
                [
                    "title",
                    "body",
                    "score",
                    "comms_num",
                    "url",
                    "text_sentiment",
                    "ticker_relevance",
                ]
            ].to_dict(orient="records"),
        }

    def rank_all_tickers(self, top_k=25):
        rows = []

        for ticker in self.alias_map.keys():
            result = self.analyze_ticker(ticker, top_k=3)
            if result["posts_used"] > 0 and result["score_0_to_100"] is not None:
                rows.append(
                    {
                        "ticker": result["ticker"],
                        "label": result["label"],
                        "score_0_to_100": result["score_0_to_100"],
                        "avg_sentiment": result["avg_sentiment"],
                        "posts_used": result["posts_used"],
                    }
                )

        if not rows:
            return pd.DataFrame()

        ranked = pd.DataFrame(rows).sort_values(
            by=["score_0_to_100", "posts_used"],
            ascending=[False, False],
        ).reset_index(drop=True)

        return ranked.head(top_k)


ALIAS_MAP = {
    "AAPL": ["AAPL", "$AAPL", "Apple"],
    "MSFT": ["MSFT", "$MSFT", "Microsoft"],
    "AMZN": ["AMZN", "$AMZN", "Amazon"],
    "GOOGL": ["GOOGL", "$GOOGL", "Alphabet", "Google"],
    "GOOG": ["GOOG", "$GOOG", "Alphabet", "Google"],
    "META": ["META", "$META", "Meta", "Facebook"],
    "TSLA": ["TSLA", "$TSLA", "Tesla"],
    "NVDA": ["NVDA", "$NVDA", "Nvidia"],
    "BRK.B": ["BRK.B", "$BRK.B", "Berkshire Hathaway"],
    "UNH": ["UNH", "$UNH", "UnitedHealth"],
    "JNJ": ["JNJ", "$JNJ", "Johnson & Johnson"],
    "V": ["V", "$V", "Visa"],
    "PG": ["PG", "$PG", "Procter & Gamble"],
    "XOM": ["XOM", "$XOM", "Exxon Mobil"],
    "HD": ["HD", "$HD", "Home Depot"],
    "MA": ["MA", "$MA", "Mastercard"],
    "CVX": ["CVX", "$CVX", "Chevron"],
    "ABBV": ["ABBV", "$ABBV", "AbbVie"],
    "PFE": ["PFE", "$PFE", "Pfizer"],
    "KO": ["KO", "$KO", "Coca-Cola"],
    "PEP": ["PEP", "$PEP", "Pepsi"],
    "TMO": ["TMO", "$TMO", "Thermo Fisher"],
    "MRK": ["MRK", "$MRK", "Merck"],
    "COST": ["COST", "$COST", "Costco"],
    "DIS": ["DIS", "$DIS", "Disney"],
    "AVGO": ["AVGO", "$AVGO", "Broadcom"],
    "ACN": ["ACN", "$ACN", "Accenture"],
    "ABT": ["ABT", "$ABT", "Abbott"],
    "DHR": ["DHR", "$DHR", "Danaher"],
    "ADBE": ["ADBE", "$ADBE", "Adobe"],
    "CRM": ["CRM", "$CRM", "Salesforce"],
    "NKE": ["NKE", "$NKE", "Nike"],
    "LLY": ["LLY", "$LLY", "Eli Lilly"],
    "TXN": ["TXN", "$TXN", "Texas Instruments"],
    "WMT": ["WMT", "$WMT", "Walmart"],
    "MCD": ["MCD", "$MCD", "McDonald's"],
    "NEE": ["NEE", "$NEE", "NextEra Energy"],
    "LIN": ["LIN", "$LIN", "Linde"],
    "ORCL": ["ORCL", "$ORCL", "Oracle"],
    "INTC": ["INTC", "$INTC", "Intel"],
    "AMD": ["AMD", "$AMD", "Advanced Micro Devices"],
    "QCOM": ["QCOM", "$QCOM", "Qualcomm"],
    "UPS": ["UPS", "$UPS", "UPS"],
    "LOW": ["LOW", "$LOW", "Lowe's"],
    "PM": ["PM", "$PM", "Philip Morris"],
    "UNP": ["UNP", "$UNP", "Union Pacific"],
    "RTX": ["RTX", "$RTX", "Raytheon"],
    "HON": ["HON", "$HON", "Honeywell"],
    "IBM": ["IBM", "$IBM", "IBM"],
    "GE": ["GE", "$GE", "General Electric"],
    "CAT": ["CAT", "$CAT", "Caterpillar"],
    "BA": ["BA", "$BA", "Boeing"],
    "GS": ["GS", "$GS", "Goldman Sachs"],
    "MS": ["MS", "$MS", "Morgan Stanley"],
    "JPM": ["JPM", "$JPM", "JPMorgan"],
    "BAC": ["BAC", "$BAC", "Bank of America"],
    "WFC": ["WFC", "$WFC", "Wells Fargo"],
    "C": ["C", "$C", "Citigroup"],
    "BLK": ["BLK", "$BLK", "BlackRock"],
    "SPGI": ["SPGI", "$SPGI", "S&P Global"],
    "AXP": ["AXP", "$AXP", "American Express"],
    "PLD": ["PLD", "$PLD", "Prologis"],
    "AMT": ["AMT", "$AMT", "American Tower"],
    "CCI": ["CCI", "$CCI", "Crown Castle"],
    "EQIX": ["EQIX", "$EQIX", "Equinix"],
    "NOW": ["NOW", "$NOW", "ServiceNow"],
    "INTU": ["INTU", "$INTU", "Intuit"],
    "ISRG": ["ISRG", "$ISRG", "Intuitive Surgical"],
    "MDT": ["MDT", "$MDT", "Medtronic"],
    "SYK": ["SYK", "$SYK", "Stryker"],
    "ZTS": ["ZTS", "$ZTS", "Zoetis"],
    "BDX": ["BDX", "$BDX", "Becton Dickinson"],
    "GILD": ["GILD", "$GILD", "Gilead"],
    "REGN": ["REGN", "$REGN", "Regeneron"],
    "VRTX": ["VRTX", "$VRTX", "Vertex"],
    "AMGN": ["AMGN", "$AMGN", "Amgen"],
    "CVS": ["CVS", "$CVS", "CVS"],
    "CI": ["CI", "$CI", "Cigna"],
    "HUM": ["HUM", "$HUM", "Humana"],
    "ELV": ["ELV", "$ELV", "Elevance"],
    "ADP": ["ADP", "$ADP", "ADP"],
    "PAYX": ["PAYX", "$PAYX", "Paychex"],
    "FIS": ["FIS", "$FIS", "FIS"],
    "FISV": ["FISV", "$FISV", "Fiserv"],
    "SQ": ["SQ", "$SQ", "Block"],
    "PYPL": ["PYPL", "$PYPL", "PayPal"],
    "UBER": ["UBER", "$UBER", "Uber"],
    "LYFT": ["LYFT", "$LYFT", "Lyft"],
    "BKNG": ["BKNG", "$BKNG", "Booking"],
    "EXPE": ["EXPE", "$EXPE", "Expedia"],
    "MAR": ["MAR", "$MAR", "Marriott"],
    "HLT": ["HLT", "$HLT", "Hilton"],
    "SBUX": ["SBUX", "$SBUX", "Starbucks"],
    "YUM": ["YUM", "$YUM", "Yum Brands"],
    "DPZ": ["DPZ", "$DPZ", "Dominos"],
    "CMG": ["CMG", "$CMG", "Chipotle"],
    "TGT": ["TGT", "$TGT", "Target"],
    "DG": ["DG", "$DG", "Dollar General"],
    "DLTR": ["DLTR", "$DLTR", "Dollar Tree"],
    "ROST": ["ROST", "$ROST", "Ross"],
    "BBY": ["BBY", "$BBY", "Best Buy"],
    "KSS": ["KSS", "$KSS", "Kohl's"],
    "ETSY": ["ETSY", "$ETSY", "Etsy"],
    "EBAY": ["EBAY", "$EBAY", "eBay"],
    "SHOP": ["SHOP", "$SHOP", "Shopify"],
    "AFL": ["AFL", "$AFL", "Aflac"],
    "AIG": ["AIG", "$AIG", "AIG"],
    "ALL": ["ALL", "$ALL", "Allstate"],
    "MET": ["MET", "$MET", "MetLife"],
    "PRU": ["PRU", "$PRU", "Prudential"],
    "TRV": ["TRV", "$TRV", "Travelers"],
    "CB": ["CB", "$CB", "Chubb"],
    "MMC": ["MMC", "$MMC", "Marsh"],
    "AON": ["AON", "$AON", "Aon"],
    "ICE": ["ICE", "$ICE", "Intercontinental Exchange"],
    "CME": ["CME", "$CME", "CME Group"],
    "SLB": ["SLB", "$SLB", "Schlumberger"],
    "EOG": ["EOG", "$EOG", "EOG Resources"],
    "PSX": ["PSX", "$PSX", "Phillips 66"],
    "MPC": ["MPC", "$MPC", "Marathon Petroleum"],
    "OXY": ["OXY", "$OXY", "Occidental"],
    "DVN": ["DVN", "$DVN", "Devon Energy"],
    "DUK": ["DUK", "$DUK", "Duke Energy"],
    "SO": ["SO", "$SO", "Southern Company"],
    "EXC": ["EXC", "$EXC", "Exelon"],
    "AEP": ["AEP", "$AEP", "American Electric Power"],
    "SPY": ["SPY", "$SPY", "S&P 500", "sp500"],
}


def get_default_csv_path():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "prototype_posts.csv")


@lru_cache(maxsize=1)
def get_default_model():
    return StockPulseSentiment(get_default_csv_path(), ALIAS_MAP)