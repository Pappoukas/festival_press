"""
enrich.py — Εμπλουτισμός της αποδελτίωσης με το πλήρες κείμενο και ποιοτική επισημείωση.

Βήματα (επαναλήψιμα· ό,τι έχει ήδη γίνει παραλείπεται):
  1. fetch     → data/articles.csv     (τίτλος, κείμενο, ημ/νία μεταδεδομένων, λέξεις)
  2. annotate  → data/annotations.csv  (sentiment, θεματικές, είδος κειμένου, περίληψη) με Gemini

Χρήση από γραμμή εντολών (π.χ. στον τοπικό server):
  python enrich.py fetch
  GEMINI_API_KEY=... python enrich.py annotate --model gemini-2.5-flash-lite --delay 6
Μετά: commit των δύο CSV στο repo ώστε να τα διαβάζει η εφαρμογή στο Streamlit Cloud.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from clippings import parse_docx, word_count

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DOCX = DATA / "Publicity5thCBF.docx"
ARTICLES = DATA / "articles.csv"
ANNOTATIONS = DATA / "annotations.csv"
THEMES = DATA / "themes.csv"

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")
SKIP_HOSTS = ("youtu.be", "youtube.com", "pressreader.com", "facebook.com")

ARTICLE_COLS = ["url_key", "url", "status", "title", "pub_date_meta", "author",
                "sitename", "text", "word_count", "fetched_at"]
ANNOT_COLS = ["url_key", "sentiment_label", "sentiment_score", "themes_llm",
              "article_type", "summary", "persons", "model", "annotated_at"]

SENTIMENT_LABELS = ["θετικό", "ουδέτερο", "αρνητικό", "μικτό"]
ARTICLE_TYPES = ["Αναπαραγωγή δελτίου τύπου", "Ρεπορτάζ", "Συνέντευξη",
                 "Άρθρο γνώμης / Κριτική", "Ατζέντα / Πρόγραμμα", "Απολογισμός"]


def _load(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype={"url_key": str})
    return pd.DataFrame(columns=cols)


# ─────────────────────────────────────────────
# 1. Λήψη κειμένων
# ─────────────────────────────────────────────
def fetch_one(url: str, timeout: int = 25) -> dict:
    import requests
    import trafilatura

    if not url:
        return {"status": "έντυπο (χωρίς URL)"}
    if any(h in url for h in SKIP_HOSTS):
        return {"status": "μη κειμενική πηγή"}
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "el,en;q=0.8"},
                         timeout=timeout)
        if r.status_code != 200:
            return {"status": f"HTTP {r.status_code}"}
        if not r.encoding or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding
        raw = trafilatura.extract(r.text, url=url, output_format="json", with_metadata=True,
                                  include_comments=False, include_tables=False,
                                  favor_precision=True)
        if not raw:
            return {"status": "χωρίς εξαγώγιμο κείμενο"}
        j = json.loads(raw)
        text = j.get("text") or ""
        return {"status": "ok", "title": j.get("title"), "pub_date_meta": j.get("date"),
                "author": j.get("author"), "sitename": j.get("sitename"),
                "text": text, "word_count": word_count(text)}
    except Exception as e:  # noqa: BLE001
        return {"status": f"σφάλμα: {type(e).__name__}"}


def fetch_all(clips: pd.DataFrame, existing: pd.DataFrame | None = None, delay: float = 1.0,
              retry_failed: bool = False, progress: Callable[[int, int, str], None] | None = None
              ) -> pd.DataFrame:
    existing = existing if existing is not None else _load(ARTICLES, ARTICLE_COLS)
    done_rows = existing[existing.status == "ok"] if retry_failed else existing
    done = set(done_rows.url_key)
    todo = clips.drop_duplicates("url_key")
    todo = todo[~todo.url_key.isin(done)]
    out = [existing[~existing.url_key.isin(todo.url_key)]]
    rows = []
    for i, r in enumerate(todo.itertuples(), 1):
        if progress:
            progress(i, len(todo), r.url or r.url_key)
        res = fetch_one(r.url)
        rows.append({"url_key": r.url_key, "url": r.url,
                     "fetched_at": datetime.now().isoformat(timespec="seconds"), **res})
        if r.url:
            time.sleep(delay)
    out.append(pd.DataFrame(rows))
    return pd.concat(out, ignore_index=True).reindex(columns=ARTICLE_COLS)


# ─────────────────────────────────────────────
# 2. Επισημείωση με Gemini
# ─────────────────────────────────────────────
def build_prompt(festival: str, themes: list[str], title: str, text: str) -> str:
    return f"""Είσαι αναλυτής δημοσίων σχέσεων. Ανάλυσε το παρακάτω δημοσίευμα για το «{festival}».

ΣΗΜΑΝΤΙΚΟ για το sentiment: αξιολόγησε τη στάση/τόνο του δημοσιεύματος ΑΠΕΝΑΝΤΙ ΣΤΟ ΦΕΣΤΙΒΑΛ
(οργάνωση, πρόγραμμα, καλεσμένοι, εμπειρία κοινού) — ΟΧΙ το θέμα των βιβλίων ή των συζητήσεων.
Το κεντρικό θέμα της διοργάνωσης («Κόσμοι σε σύγκρουση») περιέχει από τη φύση του λέξεις όπως
πόλεμος, βία, σύγκρουση· αυτές δεν σημαίνουν αρνητική κάλυψη.

Επέστρεψε ΜΟΝΟ JSON με τα πεδία:
- sentiment_label: ένα από {SENTIMENT_LABELS}
- sentiment_score: αριθμός από -1 (πολύ αρνητικό) έως 1 (πολύ θετικό), 0 = ουδέτερο/ενημερωτικό
- themes: λίστα με 0–4 από τις θεματικές {themes}
- article_type: ένα από {ARTICLE_TYPES}
- summary: μία πρόταση στα ελληνικά (έως 30 λέξεις)
- persons: λίστα με τα ονόματα προσώπων (συγγραφείς, ομιλητές, αξιωματούχοι) που αναφέρονται

ΤΙΤΛΟΣ: {title}
ΚΕΙΜΕΝΟ:
{text[:12000]}
"""


def annotate_one(client, model: str, prompt: str) -> dict:
    from google.genai import types
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
    )
    raw = (resp.text or "").strip().removeprefix("```json").removesuffix("```").strip()
    j = json.loads(raw)
    label = j.get("sentiment_label", "ουδέτερο")
    return {
        "sentiment_label": label if label in SENTIMENT_LABELS else "ουδέτερο",
        "sentiment_score": float(j.get("sentiment_score", 0) or 0),
        "themes_llm": "|".join(j.get("themes") or []),
        "article_type": j.get("article_type", ""),
        "summary": j.get("summary", ""),
        "persons": "|".join(j.get("persons") or []),
    }


def annotate_all(articles: pd.DataFrame, festival: str, api_key: str,
                 model: str = "gemini-2.5-flash-lite", delay: float = 6.0,
                 existing: pd.DataFrame | None = None, min_words: int = 40,
                 progress: Callable[[int, int, str], None] | None = None) -> pd.DataFrame:
    from google import genai
    client = genai.Client(api_key=api_key)
    themes = pd.read_csv(THEMES)["theme"].tolist()
    existing = existing if existing is not None else _load(ANNOTATIONS, ANNOT_COLS)
    ok = articles[(articles.status == "ok") & (articles.word_count >= min_words)]
    todo = ok[~ok.url_key.isin(existing.url_key)]
    rows = []
    for i, r in enumerate(todo.itertuples(), 1):
        if progress:
            progress(i, len(todo), r.url)
        try:
            res = annotate_one(client, model, build_prompt(festival, themes, r.title or "", r.text))
        except Exception as e:  # noqa: BLE001 — συνεχίζουμε· θα ξαναδοκιμαστεί στο επόμενο τρέξιμο
            print(f"  ! {r.url}: {type(e).__name__}: {e}")
            time.sleep(delay)
            continue
        rows.append({"url_key": r.url_key, "model": model,
                     "annotated_at": datetime.now().isoformat(timespec="seconds"), **res})
        time.sleep(delay)
    return pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).reindex(columns=ANNOT_COLS)


# ─────────────────────────────────────────────
# 3. Αναπαραγωγή δελτίων τύπου (ομοιότητα κειμένων)
# ─────────────────────────────────────────────
def originality(articles: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """
    Μέγιστη ομοιότητα συνημιτόνου (TF-IDF χαρακτήρων 3–5) κάθε κειμένου με οποιοδήποτε άλλο.
    Υψηλή τιμή ⇒ το κείμενο είναι κατά βάση αναδημοσίευση (π.χ. του ίδιου δελτίου τύπου).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    ok = articles[(articles.status == "ok") & (articles.word_count >= 40)].reset_index(drop=True)
    if len(ok) < 3:
        return pd.DataFrame(columns=["url_key", "max_similarity", "most_similar", "reuse"])
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                        max_features=60000).fit_transform(ok.text.str.lower())
    S = cosine_similarity(X)
    np.fill_diagonal(S, 0)
    idx = S.argmax(axis=1)
    return pd.DataFrame({
        "url_key": ok.url_key,
        "max_similarity": S.max(axis=1).round(3),
        "most_similar": ok.url.iloc[idx].values,
        "reuse": np.where(S.max(axis=1) >= threshold, "Αναδημοσίευση/κοινό κείμενο", "Πρωτότυπο κείμενο"),
    })


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def _cli_progress(i, n, label):
    print(f"[{i}/{n}] {label[:100]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "annotate"])
    ap.add_argument("--docx", default=str(DOCX))
    ap.add_argument("--delay", type=float, default=None)
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()

    title, clips = parse_docx(a.docx)
    if a.step == "fetch":
        df = fetch_all(clips, delay=a.delay if a.delay is not None else 1.0,
                       retry_failed=a.retry_failed, progress=_cli_progress)
        df.to_csv(ARTICLES, index=False)
        print(df.status.value_counts())
    else:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Ορίστε GEMINI_API_KEY στο περιβάλλον.")
        arts = _load(ARTICLES, ARTICLE_COLS)
        df = annotate_all(arts, title or "Φεστιβάλ", key, model=a.model,
                          delay=a.delay if a.delay is not None else 6.0, progress=_cli_progress)
        df.to_csv(ANNOTATIONS, index=False)
        print(df.sentiment_label.value_counts())


if __name__ == "__main__":
    main()
