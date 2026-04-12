#!/usr/bin/env python3
"""
Reconstruct thread-level Reddit records from a mixed posts/comments CSV.

This script is designed for large WallStreetBets-style exports where:
- `post_id` identifies a thread
- rows with empty `comment_id` are original posts
- rows with non-empty `comment_id` are comments attached to that thread

It produces one cleaned output row per retained `post_id` with:
    post_id,title,body,comments_text,url,score,comms_num,datetime,tag

Why SQLite?
The input can be large and may not be sorted by `post_id`, so this script uses
SQLite as a disk-backed staging area instead of loading the full CSV into RAM.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import sqlite3
import sys
import tempfile
from typing import Optional, Tuple

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MARKDOWN_TOKEN_RE = re.compile(r"[`*_~>#\[\]\(\)\|]+")
EMOTE_ARTIFACT_RE = re.compile(r"\b(?:img|emote|gif)\b(?:\s+t5[_\s]\w+)?(?:\s+\d+)*", re.IGNORECASE)
PREVIEW_TOKEN_RE = re.compile(r"\bpreview(?:\.\w+)?\b", re.IGNORECASE)
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "data.csv")
    default_output = os.path.join(script_dir, "cleaned_threads_new.csv")

    parser = argparse.ArgumentParser(
        description="Convert a mixed post/comment Reddit CSV into a clean thread-level CSV."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=default_input,
        help=(
            "Path to the raw input CSV. Defaults to data.csv in the same folder as this script."
        ),
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default=default_output,
        help=(
            "Path to write the cleaned thread-level CSV. Defaults to cleaned_threads.csv "
            "in the same folder as this script."
        ),
    )
    parser.add_argument(
        "--top-comments",
        type=int,
        default=0,
        help=(
            "Number of highest-score comments to retain per thread. "
            "Use 0 to keep all comments (default: 0)."
        ),
    )
    parser.add_argument(
        "--max-comment-chars",
        type=int,
        default=0,
        help=(
            "Maximum characters to keep from each selected comment. "
            "Use 0 for no per-comment cap (default: 0)."
        ),
    )
    parser.add_argument(
        "--max-total-comment-chars",
        type=int,
        default=0,
        help=(
            "Maximum total characters for the concatenated comments_text field. "
            "Use 0 for no total cap (default: 0)."
        ),
    )
    parser.add_argument(
        "--sqlite-path",
        default="",
        help="Optional path for the temporary SQLite database. Defaults to a temp file.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print progress every N input rows (default: 100000).",
    )
    parser.add_argument(
        "--comment-separator",
        default=" | ",
        help=(
            "Separator used between selected comments in comments_text "
            "(default: ' | ')."
        ),
    )
    return parser.parse_args()


def safe_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0

def clean_text(text: object) -> str:
    cleaned = html.unescape(str(text or ""))
    cleaned = ZERO_WIDTH_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\r", "\n")
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = EMOTE_ARTIFACT_RE.sub(" ", cleaned)
    cleaned = PREVIEW_TOKEN_RE.sub(" ", cleaned)
    cleaned = MARKDOWN_TOKEN_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("&amp;", "and")
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    cleaned = cleaned.strip(" \t\n\r-–—|:;,.!?/\\*_~`[](){}<>\"'")
    return cleaned


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    head = text[: max_chars - 3]
    last_space = head.rfind(" ")
    if last_space >= max_chars // 2:
        head = head[:last_space]
    return head.rstrip() + "..."


def better_post_candidate(
    current_rank: Optional[Tuple[int, int, int, float]],
    candidate_rank: Tuple[int, int, int, float],
) -> bool:
    if current_rank is None:
        return True
    return candidate_rank > current_rank


def open_database(sqlite_path: str) -> tuple[sqlite3.Connection, str, bool]:
    created_temp = False
    if sqlite_path:
        db_path = sqlite_path
    else:
        fd, db_path = tempfile.mkstemp(prefix="wsb_threads_", suffix=".sqlite")
        os.close(fd)
        created_temp = True

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -200000")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            post_id TEXT PRIMARY KEY,
            title TEXT,
            body TEXT,
            url TEXT,
            score REAL,
            comms_num REAL,
            datetime TEXT,
            tag TEXT,
            has_title INTEGER NOT NULL,
            has_url INTEGER NOT NULL,
            body_len INTEGER NOT NULL,
            score_rank REAL NOT NULL
        ) WITHOUT ROWID
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            comment_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT NOT NULL,
            comment_id TEXT,
            cleaned_text TEXT NOT NULL,
            score REAL NOT NULL,
            comment_datetime TEXT,
            input_order INTEGER NOT NULL
        )
        """
    )

    return conn, db_path, created_temp


def ingest_rows(args: argparse.Namespace, conn: sqlite3.Connection) -> dict[str, int]:
    stats = {
        "input_rows": 0,
        "post_rows_seen": 0,
        "comment_rows_seen": 0,
        "post_rows_kept": 0,
        "comments_kept": 0,
        "rows_skipped_missing_post_id": 0,
    }

    with open(args.input_csv, "r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        cursor = conn.cursor()
        conn.execute("BEGIN")

        for row in reader:
            stats["input_rows"] += 1
            if args.progress_every and stats["input_rows"] % args.progress_every == 0:
                print(f"Processed {stats['input_rows']:,} rows...", file=sys.stderr)

            post_id = str(row.get("post_id", "") or "").strip()
            if not post_id:
                stats["rows_skipped_missing_post_id"] += 1
                continue

            comment_id = str(row.get("comment_id", "") or "").strip()

            if not comment_id:
                stats["post_rows_seen"] += 1

                title = clean_text(row.get("title", ""))
                body = clean_text(row.get("text", ""))
                url = str(row.get("url", "") or "").strip()
                score = safe_float(row.get("score", 0))
                comms_num = safe_float(row.get("comments", 0))
                dt = str(row.get("datetime", "") or "").strip()
                tag = str(row.get("tag", "") or "").strip()

                candidate_rank = (
                    int(bool(title)),
                    int(bool(url)),
                    len(body),
                    score,
                )

                existing = cursor.execute(
                    """
                    SELECT has_title, has_url, body_len, score_rank
                    FROM posts
                    WHERE post_id = ?
                    """,
                    (post_id,),
                ).fetchone()

                current_rank = tuple(existing) if existing else None
                if better_post_candidate(current_rank, candidate_rank):
                    cursor.execute(
                        """
                        INSERT INTO posts (
                            post_id, title, body, url, score, comms_num, datetime, tag,
                            has_title, has_url, body_len, score_rank
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(post_id) DO UPDATE SET
                            title = excluded.title,
                            body = excluded.body,
                            url = excluded.url,
                            score = excluded.score,
                            comms_num = excluded.comms_num,
                            datetime = excluded.datetime,
                            tag = excluded.tag,
                            has_title = excluded.has_title,
                            has_url = excluded.has_url,
                            body_len = excluded.body_len,
                            score_rank = excluded.score_rank
                        """,
                        (
                            post_id,
                            title,
                            body,
                            url,
                            score,
                            comms_num,
                            dt,
                            tag,
                            candidate_rank[0],
                            candidate_rank[1],
                            candidate_rank[2],
                            candidate_rank[3],
                        ),
                    )
                    stats["post_rows_kept"] += 1

            else:
                stats["comment_rows_seen"] += 1
                raw_comment_text = str(row.get("text", "") or "")
                cleaned_text = clean_text(raw_comment_text)
                if not cleaned_text:
                    cleaned_text = WHITESPACE_RE.sub(" ", raw_comment_text).strip()
                cursor.execute(
                    """
                    INSERT INTO comments (post_id, comment_id, cleaned_text, score, comment_datetime, input_order)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        post_id,
                        str(row.get("comment_id", "") or "").strip(),
                        cleaned_text,
                        safe_float(row.get("score", 0)),
                        str(row.get("datetime", "") or "").strip(),
                        stats["input_rows"],
                    ),
                )
                stats["comments_kept"] += 1

            if stats["input_rows"] % 50000 == 0:
                conn.commit()
                conn.execute("BEGIN")

        conn.commit()

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_comments_post_score
        ON comments(post_id, score DESC, input_order ASC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_comments_post_datetime
        ON comments(post_id, comment_datetime ASC, input_order ASC)
        """
    )
    conn.commit()

    return stats


def fetch_top_comments(
    conn: sqlite3.Connection,
    post_id: str,
    top_comments: int,
    max_comment_chars: int,
    max_total_comment_chars: int,
    comment_separator: str,
) -> str:
    if top_comments and top_comments > 0:
        rows = conn.execute(
            """
            SELECT cleaned_text
            FROM comments
            WHERE post_id = ?
            ORDER BY score DESC, length(cleaned_text) DESC
            LIMIT ?
            """,
            (post_id, top_comments),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT cleaned_text
            FROM comments
            WHERE post_id = ?
            ORDER BY comment_datetime ASC, input_order ASC
            """,
            (post_id,),
        ).fetchall()

    separator = comment_separator if comment_separator is not None else " | "
    pieces = []
    total_chars = 0

    for (comment_text,) in rows:
        trimmed = truncate_text(comment_text, max_comment_chars)
        if not trimmed:
            continue

        added_chars = len(trimmed) + (len(separator) if pieces else 0)
        if max_total_comment_chars > 0 and total_chars + added_chars > max_total_comment_chars:
            remaining = max_total_comment_chars - total_chars
            if remaining <= 3:
                break
            trimmed = truncate_text(trimmed, remaining)
            if not trimmed:
                break

        pieces.append(trimmed)
        total_chars += len(trimmed) + (len(separator) if len(pieces) > 1 else 0)

    return separator.join(pieces)


def export_rows(args: argparse.Namespace, conn: sqlite3.Connection) -> dict[str, int]:
    stats = {
        "output_rows": 0,
        "posts_dropped_empty": 0,
        "orphan_comment_rows": 0,
        "orphan_comment_threads": 0,
    }

    stats["orphan_comment_rows"] = conn.execute(
        """
        SELECT COUNT(*)
        FROM comments c
        LEFT JOIN posts p ON c.post_id = p.post_id
        WHERE p.post_id IS NULL
        """
    ).fetchone()[0]

    stats["orphan_comment_threads"] = conn.execute(
        """
        SELECT COUNT(DISTINCT c.post_id)
        FROM comments c
        LEFT JOIN posts p ON c.post_id = p.post_id
        WHERE p.post_id IS NULL
        """
    ).fetchone()[0]

    with open(args.output_csv, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "post_id",
                "title",
                "body",
                "comments_text",
                "url",
                "score",
                "comms_num",
                "datetime",
                "tag",
            ],
        )
        writer.writeheader()

        query = conn.execute(
            """
            SELECT post_id, title, body, url, score, comms_num, datetime, tag
            FROM posts
            ORDER BY post_id
            """
        )

        for post_id, title, body, url, score, comms_num, dt, tag in query:
            title = title or ""
            body = body or ""

            if not title and not body:
                stats["posts_dropped_empty"] += 1
                continue

            comments_text = fetch_top_comments(
                conn=conn,
                post_id=post_id,
                top_comments=args.top_comments,
                max_comment_chars=args.max_comment_chars,
                max_total_comment_chars=args.max_total_comment_chars,
                comment_separator=args.comment_separator,
            )

            writer.writerow(
                {
                    "post_id": post_id,
                    "title": title,
                    "body": body,
                    "comments_text": comments_text,
                    "url": url or "",
                    "score": int(score) if float(score).is_integer() else score,
                    "comms_num": int(comms_num)
                    if float(comms_num).is_integer()
                    else comms_num,
                    "datetime": dt or "",
                    "tag": tag or "",
                }
            )
            stats["output_rows"] += 1

    return stats


def print_summary(ingest_stats: dict[str, int], export_stats: dict[str, int], output_csv: str) -> None:
    print("\nDone.", file=sys.stderr)
    print(f"Output written to: {output_csv}", file=sys.stderr)
    print(f"Input rows processed: {ingest_stats['input_rows']:,}", file=sys.stderr)
    print(f"Post rows seen: {ingest_stats['post_rows_seen']:,}", file=sys.stderr)
    print(f"Comment rows seen: {ingest_stats['comment_rows_seen']:,}", file=sys.stderr)
    print(f"Comment rows kept: {ingest_stats['comments_kept']:,}", file=sys.stderr)
    print(f"Orphan comment rows dropped: {export_stats['orphan_comment_rows']:,}", file=sys.stderr)
    print(
        f"Orphan comment threads dropped: {export_stats['orphan_comment_threads']:,}",
        file=sys.stderr,
    )
    print(f"Posts dropped for empty title/body: {export_stats['posts_dropped_empty']:,}", file=sys.stderr)
    print(f"Thread rows written: {export_stats['output_rows']:,}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(
            f"Input file not found: {args.input_csv}\n"
            "Put data.csv in the same folder as clean_wsb_threads.py, or pass a file path explicitly."
        )

    conn, db_path, delete_db = open_database(args.sqlite_path)

    try:
        ingest_stats = ingest_rows(args, conn)
        export_stats = export_rows(args, conn)
        print_summary(ingest_stats, export_stats, args.output_csv)
    finally:
        conn.close()
        if delete_db:
            try:
                os.remove(db_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
