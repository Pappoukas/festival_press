import re
from collections import Counter
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clippings import (NO_WAVE, assign_period, detect_themes, gini, load_themes, normalize,
                       parse_docx, resolve_dates, slug_text)

# ─────────────────────────────────────────────
# Ρύθμιση σελίδας
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Festival Press Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data"
DOCX_FILE = DATA / "Publicity5thCBF.docx"
OUTLETS_FILE = DATA / "outlets.csv"
THEMES_FILE = DATA / "themes.csv"
ARTICLES_FILE = DATA / "articles.csv"
ANNOT_FILE = DATA / "annotations.csv"
STOP_FILE = DATA / "stopwords_el.txt"

PERIOD_COLORS = {
    "1. Προαναγγελία": "#95a5a6",
    "2. Αντίστροφη μέτρηση (−14 ημ.)": "#4A90D9",
    "3. Διάρκεια φεστιβάλ": "#e67e22",
    "4. Απολογισμός": "#8e44ad",
    "Άγνωστη ημ/νία": "#bdc3c7",
}
SENT_COLORS = {"θετικό": "#2ecc71", "ουδέτερο": "#95a5a6", "μικτό": "#f1c40f", "αρνητικό": "#e74c3c"}
CAT_SEQ = px.colors.qualitative.Set2


# ─────────────────────────────────────────────
# Φόρτωση δεδομένων
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_clippings(payload: bytes | None):
    return parse_docx(payload if payload else str(DOCX_FILE))


@st.cache_data(show_spinner=False)
def load_csv(path: Path, mtime: float):
    return pd.read_csv(path, dtype={"url_key": str})


def csv_or_none(path: Path):
    return load_csv(path, path.stat().st_mtime) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_stopwords():
    if not STOP_FILE.exists():
        return set()
    return {normalize(w) for w in STOP_FILE.read_text(encoding="utf-8").split()}


@st.cache_data(show_spinner=False)
def compute_originality(articles: pd.DataFrame, threshold: float):
    from enrich import originality
    return originality(articles, threshold)


def to_excel(sheets: dict) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


def festival_band(fig, start, end, waves=None):
    """Σκίαση διάρκειας φεστιβάλ και κάθετες γραμμές για τα Δ.Τ."""
    fig.add_vrect(x0=pd.Timestamp(start) - pd.Timedelta(hours=12),
                  x1=pd.Timestamp(end) + pd.Timedelta(hours=12),
                  fillcolor="#e67e22", opacity=0.12, line_width=0, layer="below")
    fig.add_annotation(x=pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2,
                       y=1.04, yref="paper", text="Φεστιβάλ", showarrow=False,
                       font=dict(color="#e67e22", size=11))
    if waves is not None:
        for _, w in waves.iterrows():
            fig.add_shape(type="line", x0=w.wave_date, x1=w.wave_date, y0=0, y1=1, yref="paper",
                          line=dict(color="#7f8c8d", width=1, dash="dot"))
            fig.add_annotation(x=w.wave_date, y=0.98, yref="paper", text=w.short, showarrow=False,
                               textangle=-90, xanchor="right", font=dict(size=10, color="#7f8c8d"))
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("📊 Φίλτρα Ανάλυσης")

upl = st.sidebar.file_uploader("Άλλη αποδελτίωση (.docx)", type=["docx"],
                               help="Ίδια δομή με το Publicity5thCBF.docx: επικεφαλίδες ημερομηνιών, "
                                    "έντονες γραμμές «Δ. Τύπου», ένας σύνδεσμος ανά παράγραφο.")
title, clips = load_clippings(upl.getvalue() if upl else None)
if clips.empty:
    st.error("Δεν βρέθηκαν σύνδεσμοι στο αρχείο.")
    st.stop()

year = int(clips.list_date.dropna().dt.year.mode().iloc[0]) if clips.list_date.notna().any() else 2026
c1, c2 = st.sidebar.columns(2)
fest_start = c1.date_input("Έναρξη", date(year, 6, 22), format="DD/MM/YYYY")
fest_end = c2.date_input("Λήξη", date(year, 6, 28), format="DD/MM/YYYY")

date_pref = st.sidebar.radio(
    "Ημερομηνία δημοσιεύματος", ["Αποδελτίωσης", "Δημοσίευσης (όπου υπάρχει)"], horizontal=True,
    help="Αποδελτίωσης: η επικεφαλίδα του αρχείου. Δημοσίευσης: η ημερομηνία στο URL ή στα "
         "μεταδεδομένα της σελίδας (μετά τη λήψη κειμένων). Ό,τι λείπει συμπληρώνεται διαδοχικά.")

include_dups = st.sidebar.checkbox("Συμπερίληψη διπλοεγγραφών", value=False,
                                   help="Ο ίδιος σύνδεσμος καταχωρισμένος περισσότερες από μία φορές.")

# Συνένωση με μέσα, κείμενα, επισημειώσεις
outlets = pd.read_csv(OUTLETS_FILE)
articles = st.session_state.get("articles", csv_or_none(ARTICLES_FILE))
annots = st.session_state.get("annotations", csv_or_none(ANNOT_FILE))

df = clips.merge(outlets, on="host", how="left")
df["outlet"] = df["outlet"].fillna(df["print_outlet"]).fillna(df["host"])
df["category"] = df["category"].fillna(pd.Series(np.where(df.host == "έντυπο", "Ειδησεογραφικά", "Αταξινόμητο"), index=df.index))
df["reach"] = df["reach"].fillna(pd.Series(np.where(df.host == "έντυπο", "Εθνική", "Άγνωστη"), index=df.index))
if articles is not None:
    df = df.merge(articles.drop(columns=["url"], errors="ignore"), on="url_key", how="left")
df = resolve_dates(df, "pub" if date_pref.startswith("Δημοσίευσης") else "list")
df["period"] = df["date"].map(lambda d: assign_period(d, fest_start, fest_end))
df["wave_short"] = df["wave"].str.extract(r"^(\d+ο[^0-9]*?Τύπου)")[0].fillna("—")
has_text = articles is not None and "text" in df and df["text"].notna().any()
if annots is not None:
    df = df.merge(annots, on="url_key", how="left")
has_annot = annots is not None and "sentiment_label" in df and df["sentiment_label"].notna().any()

periods = sorted(df.period.unique())
sel_periods = st.sidebar.multiselect("Περίοδοι", periods, default=periods)
cats = sorted(df.category.unique())
sel_cats = st.sidebar.multiselect("Κατηγορίες μέσων", cats, default=cats)
reaches = sorted(df.reach.unique())
sel_reach = st.sidebar.multiselect("Εμβέλεια", reaches, default=reaches)
kinds = sorted(df.kind.unique())
sel_kind = st.sidebar.multiselect("Είδος", kinds, default=kinds)

st.sidebar.divider()
st.sidebar.markdown(
    "**Πηγή δεδομένων:** αποδελτίωση Τύπου (.docx)\n\n"
    f"**Κείμενα:** {'✅ ' + str(int((articles.status == 'ok').sum())) + ' άρθρα' if articles is not None else '— (βλ. Εμπλουτισμός)'}\n\n"
    f"**Επισημείωση LLM:** {'✅ ' + str(len(annots)) + ' άρθρα' if annots is not None else '—'}"
)
st.sidebar.markdown("**Απεικόνιση:** Καλλίνικος Κωνσταντίνος")

# ─────────────────────────────────────────────
# ΦΙΛΤΡΑΡΙΣΜΑ
# ─────────────────────────────────────────────
fdf = df[df.period.isin(sel_periods) & df.category.isin(sel_cats)
         & df.reach.isin(sel_reach) & df.kind.isin(sel_kind)]
if not include_dups:
    fdf = fdf[~fdf.is_duplicate]
if fdf.empty:
    st.warning("Κανένα δημοσίευμα με τα τρέχοντα φίλτρα.")
    st.stop()

waves = (df[df.wave != NO_WAVE].dropna(subset=["wave_date"])
         .drop_duplicates("wave")[["wave", "wave_date", "wave_short"]]
         .rename(columns={"wave_short": "short"}))

# ═════════════════════════════════════════════
# ΤΙΤΛΟΣ & KPIs
# ═════════════════════════════════════════════
st.title(f"📚 {title or 'Αποδελτίωση Τύπου'}")
st.caption(f"Ανάλυση {len(fdf)} δημοσιευμάτων · "
           f"{fdf.date.min():%d/%m/%Y} – {fdf.date.max():%d/%m/%Y} · "
           f"φεστιβάλ {fest_start:%d/%m}–{fest_end:%d/%m/%Y}")

st.subheader("📊 Βασικοί Δείκτες")
daily = fdf.groupby(fdf.date.dt.normalize()).size()
during = fdf.period.eq("3. Διάρκεια φεστιβάλ").mean() * 100
outlet_counts = fdf.outlet.value_counts()
k = st.columns(4)
k[0].metric("Δημοσιεύματα", f"{len(fdf):,}",
            help=f"Διπλοεγγραφές στο αρχείο: {int(df.is_duplicate.sum())}")
k[1].metric("Μοναδικά μέσα", f"{fdf.outlet.nunique()}")
k[2].metric("Ημέρες με κάλυψη", f"{daily.size}")
k[3].metric("Κορυφαία ημέρα", f"{daily.idxmax():%d/%m}", f"{daily.max()} δημοσιεύματα", delta_color="off")
k = st.columns(4)
k[0].metric("Κατά τη διάρκεια", f"{during:.0f}%")
k[1].metric("Εθνικής εμβέλειας", f"{fdf.reach.eq('Εθνική').mean() * 100:.0f}%")
k[2].metric("Συνεντεύξεις", f"{int(fdf.kind.eq('Συνέντευξη').sum())}")
k[3].metric("Gini μέσων", f"{gini(outlet_counts.values):.2f}",
            help="0 = ισοκατανομή δημοσιευμάτων στα μέσα · 1 = συγκέντρωση σε ελάχιστα μέσα")
approx = (fdf.date_source == "Δ.Τ. (κατά προσέγγιση)").sum()
if approx:
    st.caption(f"ℹ️ {approx} δημοσιεύματα δεν έχουν δική τους ημερομηνία στο αρχείο· "
               "χρονολογούνται με την ημερομηνία του αντίστοιχου Δ.Τ. Μετά τη λήψη κειμένων "
               "και την επιλογή «Δημοσίευσης» η χρονολόγηση γίνεται ακριβέστερη.")
st.divider()

# ═════════════════════════════════════════════
# 1. ΧΡΟΝΙΚΗ ΚΑΤΑΝΟΜΗ
# ═════════════════════════════════════════════
st.subheader("⏱️ Χρονική Κατανομή")
t_day, t_week, t_cum, t_wave, t_dow = st.tabs(
    ["📅 Ημερήσια", "🗓️ Εβδομαδιαία", "📈 Σωρευτική", "📣 Ανά Δελτίο Τύπου", "📆 Ημέρα εβδομάδας"])

with t_day:
    d = fdf.groupby([fdf.date.dt.normalize().rename("Ημερομηνία"), "period"]).size().reset_index(name="Δημοσιεύματα")
    fig = px.bar(d, x="Ημερομηνία", y="Δημοσιεύματα", color="period",
                 color_discrete_map=PERIOD_COLORS, labels={"period": "Περίοδος"})
    festival_band(fig, fest_start, fest_end, waves)
    fig.update_layout(bargap=0.1, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

with t_week:
    w = fdf.set_index("date").resample("W-MON", label="left", closed="left").size().reset_index(name="Δημοσιεύματα")
    w["Εβδομάδα"] = w["date"].dt.strftime("%d/%m")
    fig = px.bar(w, x="date", y="Δημοσιεύματα", text="Δημοσιεύματα", color_discrete_sequence=["#4A90D9"],
                 labels={"date": "Εβδομάδα (από Δευτέρα)"})
    festival_band(fig, fest_start, fest_end)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

with t_cum:
    c = daily.sort_index().cumsum().reset_index(name="Σωρευτικά")
    c.columns = ["Ημερομηνία", "Σωρευτικά"]
    fig = px.area(c, x="Ημερομηνία", y="Σωρευτικά", color_discrete_sequence=["#4A90D9"])
    festival_band(fig, fest_start, fest_end, waves)
    st.plotly_chart(fig, width="stretch")
    half = c[c.Σωρευτικά >= c.Σωρευτικά.max() / 2].iloc[0]
    st.caption(f"💡 Το 50% της κάλυψης είχε συγκεντρωθεί έως τις {half.Ημερομηνία:%d/%m}.")

with t_wave:
    wv = fdf.groupby("wave").agg(Δημοσιεύματα=("url_key", "size"), Μέσα=("outlet", "nunique"),
                                 Εθνικά=("reach", lambda s: (s == "Εθνική").sum())).reset_index()
    wv = wv.merge(waves[["wave", "wave_date"]], on="wave", how="left").sort_values("wave_date", na_position="last")
    fig = px.bar(wv, y="wave", x="Δημοσιεύματα", orientation="h", text="Δημοσιεύματα",
                 color="Μέσα", color_continuous_scale="Blues", labels={"wave": ""})
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=wv.wave.tolist()[::-1]))
    st.plotly_chart(fig, width="stretch")
    st.caption("Απήχηση κάθε δελτίου τύπου: δημοσιεύματα που χρονολογούνται έως "
               "20 ημέρες μετά την αποστολή του. Τα υπόλοιπα («Χωρίς συγκεκριμένο Δ.Τ.») "
               "είναι κυρίως η κάλυψη της περιόδου του φεστιβάλ.")

with t_dow:
    days_gr = ["Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή"]
    dw = fdf.date.dt.dayofweek.map(dict(enumerate(days_gr))).value_counts().reindex(days_gr, fill_value=0)
    fig = px.bar(x=dw.index, y=dw.values, labels={"x": "", "y": "Δημοσιεύματα"},
                 color_discrete_sequence=["#4A90D9"], text=dw.values)
    st.plotly_chart(fig, width="stretch")
    st.caption("Χρήσιμο για τον προγραμματισμό αποστολής δελτίων τύπου.")

st.divider()

# ═════════════════════════════════════════════
# 2. ΜΕΣΑ & ΕΜΒΕΛΕΙΑ
# ═════════════════════════════════════════════
st.subheader("📰 Μέσα & Εμβέλεια")
t_top, t_cat, t_reach, t_lor, t_mix = st.tabs(
    ["🏆 Top μέσα", "🗂️ Κατηγορίες", "🌍 Εμβέλεια", "⚖️ Συγκέντρωση (Lorenz)", "🔀 Κατηγορία × Περίοδος"])

with t_top:
    n_top = st.slider("Πλήθος μέσων", 5, 40, 20, key="ntop")
    top = (fdf.groupby(["outlet", "category"]).size().reset_index(name="Δημοσιεύματα")
           .sort_values("Δημοσιεύματα", ascending=False).head(n_top))
    fig = px.bar(top, x="Δημοσιεύματα", y="outlet", color="category", orientation="h",
                 text="Δημοσιεύματα", color_discrete_sequence=CAT_SEQ,
                 labels={"outlet": "", "category": "Κατηγορία"}, height=max(400, 24 * len(top)))
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, width="stretch")

with t_cat:
    cc = fdf.category.value_counts().reset_index()
    cc.columns = ["Κατηγορία", "Δημοσιεύματα"]
    col_a, col_b = st.columns(2)
    col_a.plotly_chart(px.pie(cc, names="Κατηγορία", values="Δημοσιεύματα", hole=0.45,
                              color_discrete_sequence=CAT_SEQ), width="stretch")
    co = fdf.groupby("category").outlet.nunique().reset_index(name="Μέσα")
    col_b.plotly_chart(px.bar(co.sort_values("Μέσα"), x="Μέσα", y="category", orientation="h",
                              text="Μέσα", labels={"category": ""},
                              color_discrete_sequence=["#4A90D9"], title="Μοναδικά μέσα ανά κατηγορία"),
                       width="stretch")

with t_reach:
    rr = fdf.groupby(["reach", "period"]).size().reset_index(name="Δημοσιεύματα")
    fig = px.bar(rr, x="reach", y="Δημοσιεύματα", color="period", color_discrete_map=PERIOD_COLORS,
                 labels={"reach": "Εμβέλεια", "period": "Περίοδος"})
    fig.update_layout(xaxis=dict(categoryorder="total descending"))
    st.plotly_chart(fig, width="stretch")
    unver = outlets[outlets.verify == 1].outlet.tolist()
    if unver:
        st.caption("⚠️ Η κατηγοριοποίηση ορισμένων μέσων χρειάζεται επιβεβαίωση "
                   f"(στήλη verify στο data/outlets.csv): {', '.join(unver)}.")

with t_lor:
    vals = np.sort(outlet_counts.values)
    lor = np.insert(vals.cumsum() / vals.sum(), 0, 0)
    xs = np.linspace(0, 1, len(lor))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=lor, fill="tozeroy", name="Καμπύλη Lorenz", line=dict(color="#4A90D9")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Ισοκατανομή", line=dict(dash="dash", color="#95a5a6")))
    fig.update_layout(xaxis_title="Αθροιστικό % μέσων", yaxis_title="Αθροιστικό % δημοσιευμάτων",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    share = outlet_counts.cumsum() / outlet_counts.sum()
    n80 = int((share <= 0.8).sum()) + 1
    one_off = int((outlet_counts == 1).sum())
    st.caption(f"💡 Gini = {gini(outlet_counts.values):.2f} · {n80} από {len(outlet_counts)} μέσα "
               f"καλύπτουν το 80% των δημοσιευμάτων · {one_off} μέσα με ένα μόνο δημοσίευμα.")

with t_mix:
    mx = pd.crosstab(fdf.category, fdf.period)
    fig = px.imshow(mx, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                    labels=dict(x="Περίοδος", y="", color="Δημοσιεύματα"))
    st.plotly_chart(fig, width="stretch")

st.divider()

# ═════════════════════════════════════════════
# 3. ΠΕΡΙΕΧΟΜΕΝΟ & ΕΚΤΑΣΗ
# ═════════════════════════════════════════════
st.subheader("📝 Περιεχόμενο & Έκταση")
if not has_text:
    st.info("Για αριθμό λέξεων, θεματικές από το κείμενο και ανίχνευση αναδημοσιεύσεων "
            "χρειάζεται πρώτα η λήψη των κειμένων — δείτε «⚙️ Εμπλουτισμός δεδομένων» στο τέλος.")
else:
    tx = fdf[fdf.status == "ok"].copy()
    k = st.columns(4)
    k[0].metric("Κείμενα που ανακτήθηκαν", f"{len(tx)} / {fdf.url.ne('').sum()}")
    k[1].metric("Σύνολο λέξεων", f"{int(tx.word_count.sum()):,}".replace(",", "."))
    k[2].metric("Διάμεσος λέξεων", f"{tx.word_count.median():.0f}")
    k[3].metric("Μέγιστο", f"{int(tx.word_count.max())}")

    t_hist, t_box, t_trend, t_orig, t_fail = st.tabs(
        ["📏 Κατανομή λέξεων", "🗂️ Ανά κατηγορία", "📈 Έκταση στον χρόνο", "🧬 Αναδημοσιεύσεις", "⚠️ Αποτυχίες λήψης"])
    with t_hist:
        fig = px.histogram(tx, x="word_count", nbins=40, color="period", color_discrete_map=PERIOD_COLORS,
                           labels={"word_count": "Λέξεις", "period": "Περίοδος"})
        st.plotly_chart(fig, width="stretch")
        bins = pd.cut(tx.word_count, [0, 150, 400, 800, 10**6],
                      labels=["Σύντομη είδηση (<150)", "Μεσαίο (150–400)", "Εκτενές (400–800)", "Αφιέρωμα (800+)"])
        st.dataframe(bins.value_counts().rename("Δημοσιεύματα"), width="stretch")
    with t_box:
        fig = px.box(tx, x="category", y="word_count", points="all", color="category",
                     color_discrete_sequence=CAT_SEQ, hover_data=["outlet", "title"],
                     labels={"word_count": "Λέξεις", "category": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with t_trend:
        fig = px.scatter(tx, x="date", y="word_count", color="reach", size="word_count",
                         hover_data=["outlet", "title"], labels={"word_count": "Λέξεις", "date": ""})
        festival_band(fig, fest_start, fest_end)
        st.plotly_chart(fig, width="stretch")
    with t_orig:
        thr = st.slider("Κατώφλι ομοιότητας", 0.4, 0.9, 0.6, 0.05,
                        help="Ομοιότητα συνημιτόνου (TF-IDF χαρακτήρων) με το πιο όμοιο άλλο κείμενο.")
        orig = compute_originality(articles, thr)
        tx2 = tx.merge(orig, on="url_key", how="left")
        oc = tx2.groupby(["wave_short", "reuse"]).size().reset_index(name="Δημοσιεύματα")
        fig = px.bar(oc, x="wave_short", y="Δημοσιεύματα", color="reuse", barmode="stack",
                     color_discrete_map={"Αναδημοσίευση/κοινό κείμενο": "#95a5a6", "Πρωτότυπο κείμενο": "#2ecc71"},
                     labels={"wave_short": "Δελτίο τύπου", "reuse": ""})
        st.plotly_chart(fig, width="stretch")
        rate = tx2.reuse.eq("Αναδημοσίευση/κοινό κείμενο").mean() * 100
        st.caption(f"💡 Περίπου {rate:.0f}% των κειμένων μοιράζονται σε μεγάλο βαθμό το ίδιο κείμενο "
                   "με άλλο δημοσίευμα — ένδειξη αυτούσιας αναπαραγωγής δελτίου τύπου. "
                   "Τα «πρωτότυπα» είναι η κερδισμένη (earned) δημοσιογραφική κάλυψη.")
        with st.expander("📋 Τα πιο «πρωτότυπα» δημοσιεύματα"):
            st.dataframe(tx2.sort_values("max_similarity")[["date", "outlet", "title", "word_count", "max_similarity", "url"]].head(30),
                         width="stretch", hide_index=True,
                         column_config={"url": st.column_config.LinkColumn("Σύνδεσμος"),
                                        "date": st.column_config.DateColumn("Ημ/νία", format="DD/MM")})
    with t_fail:
        fl = fdf[fdf.status.ne("ok") & fdf.status.notna()]
        st.dataframe(fl[["outlet", "status", "url"]], width="stretch", hide_index=True,
                     column_config={"url": st.column_config.LinkColumn("Σύνδεσμος")})
        st.caption("Σελίδες με paywall, βίντεο ή αποκλεισμό bot. Μπορείτε να ξαναδοκιμάσετε "
                   "με `python enrich.py fetch --retry-failed`.")

st.divider()

# ═════════════════════════════════════════════
# 4. ΘΕΜΑΤΙΚΕΣ
# ═════════════════════════════════════════════
st.subheader("🏷️ Θεματικές")
themes = load_themes(THEMES_FILE)
sources = ["Λέξεις-κλειδιά"] + (["Gemini (LLM)"] if has_annot else [])
th_src = st.radio("Μέθοδος", sources, horizontal=True,
                  help="Λέξεις-κλειδιά: λεξικό στο data/themes.csv, εφαρμόζεται στο κείμενο ή — όπου "
                       "δεν υπάρχει — στον τίτλο/URL. Gemini: θεματικές που απέδωσε το μοντέλο.")
tdf = fdf.copy()
if th_src.startswith("Gemini"):
    tdf["themes_list"] = tdf.themes_llm.fillna("").map(lambda s: [t for t in s.split("|") if t])
    tdf = tdf[tdf.sentiment_label.notna()]
else:
    def _theme_input(r):
        if has_text and isinstance(r.get("text"), str):
            return f"{r.get('title') or ''} {r['text']}"
        return f"{r.get('note') or ''} {slug_text(r['url'])}"
    tdf["themes_list"] = tdf.apply(lambda r: detect_themes(_theme_input(r), themes), axis=1)
    if not has_text:
        st.caption("ℹ️ Χωρίς κείμενα, οι θεματικές ανιχνεύονται μόνο από τα URL (greeklish) — "
                   "ενδεικτικά αποτελέσματα, κάλυψη περίπου μισών δημοσιευμάτων.")

ex = tdf.explode("themes_list").dropna(subset=["themes_list"]).reset_index(drop=True)
covered = tdf.themes_list.str.len().gt(0).mean() * 100
t_bar, t_heat, t_terms, t_co = st.tabs(["📊 Συχνότητα", "🔥 Θέμα × Περίοδος", "🔤 Χαρακτηριστικοί όροι", "🔗 Συνεμφανίσεις"])
with t_bar:
    tc = ex.themes_list.value_counts().reset_index()
    tc.columns = ["Θεματική", "Δημοσιεύματα"]
    tc["%"] = (tc["Δημοσιεύματα"] / len(tdf) * 100).round(1)
    fig = px.bar(tc, x="Δημοσιεύματα", y="Θεματική", orientation="h", text="%",
                 color_discrete_sequence=["#4A90D9"], labels={"Θεματική": ""})
    fig.update_traces(texttemplate="%{text}%")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Τουλάχιστον μία θεματική εντοπίστηκε στο {covered:.0f}% των δημοσιευμάτων. "
               "Ένα δημοσίευμα μπορεί να ανήκει σε περισσότερες θεματικές.")
with t_heat:
    if not ex.empty:
        hm = pd.crosstab(ex.themes_list, ex.period)
        st.plotly_chart(px.imshow(hm, text_auto=True, aspect="auto", color_continuous_scale="Oranges",
                                  labels=dict(x="", y="", color="Δημοσιεύματα")), width="stretch")
with t_terms:
    if not has_text:
        st.info("Απαιτούνται τα κείμενα των άρθρων.")
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        stop = load_stopwords()
        def tok(s):
            return [w for w in re.findall(r"[^\W\d_]{4,}", s.lower()) if normalize(w) not in stop]
        grp = fdf[fdf.status == "ok"].groupby("period").text.apply(lambda s: " ".join(s))
        if len(grp) >= 2:
            vec = TfidfVectorizer(tokenizer=tok, lowercase=False, token_pattern=None, sublinear_tf=True)
            X = vec.fit_transform(grp.values)
            terms = np.array(vec.get_feature_names_out())
            cols = st.columns(len(grp))
            for i, (per, col) in enumerate(zip(grp.index, cols)):
                row = X[i].toarray().ravel()
                top_i = row.argsort()[::-1][:15]
                col.markdown(f"**{per}**")
                col.dataframe(pd.DataFrame({"Όρος": terms[top_i], "TF-IDF": row[top_i].round(3)}),
                              hide_index=True, width="stretch")
            st.caption("Όροι που χαρακτηρίζουν κάθε περίοδο σε σχέση με τις άλλες (TF-IDF, "
                       "με λίστα stopwords στο data/stopwords_el.txt).")
        else:
            st.info("Χρειάζονται κείμενα από τουλάχιστον δύο περιόδους.")
with t_co:
    pairs = Counter()
    for lst in tdf.themes_list:
        s = sorted(set(lst))
        pairs.update((a, b) for i, a in enumerate(s) for b in s[i + 1:])
    if pairs:
        cp = pd.DataFrame([(a, b, n) for (a, b), n in pairs.most_common(15)],
                          columns=["Θεματική Α", "Θεματική Β", "Κοινά δημοσιεύματα"])
        st.dataframe(cp, width="stretch", hide_index=True)

st.divider()

# ═════════════════════════════════════════════
# 5. SENTIMENT
# ═════════════════════════════════════════════
st.subheader("💬 Sentiment & Είδος Κάλυψης")
if not has_annot:
    st.info("Η ανάλυση συναισθήματος γίνεται με Gemini πάνω στα κείμενα, με οδηγία να κρίνεται η "
            "στάση απέναντι στο φεστιβάλ και όχι το θέμα (το «Κόσμοι σε σύγκρουση» θα έδινε ψευδώς "
            "αρνητικά αποτελέσματα με απλό λεξικό). Δείτε «⚙️ Εμπλουτισμός δεδομένων».")
else:
    sd = fdf[fdf.sentiment_label.notna()].copy()
    k = st.columns(4)
    k[0].metric("Επισημειωμένα", f"{len(sd)}")
    k[1].metric("Μέσο σκορ", f"{sd.sentiment_score.mean():+.2f}", help="−1 αρνητικό · 0 ουδέτερο · +1 θετικό")
    k[2].metric("Θετικά", f"{sd.sentiment_label.eq('θετικό').mean() * 100:.0f}%")
    k[3].metric("Αρνητικά/Μικτά", f"{sd.sentiment_label.isin(['αρνητικό', 'μικτό']).mean() * 100:.0f}%")

    t_dist, t_time, t_by, t_type, t_pers, t_list = st.tabs(
        ["🥧 Κατανομή", "📈 Στον χρόνο", "🗂️ Ανά κατηγορία", "🧾 Είδος κειμένου", "👤 Πρόσωπα", "📋 Περιλήψεις"])
    with t_dist:
        sc = sd.sentiment_label.value_counts().reset_index()
        sc.columns = ["Sentiment", "Δημοσιεύματα"]
        st.plotly_chart(px.pie(sc, names="Sentiment", values="Δημοσιεύματα", hole=0.45,
                               color="Sentiment", color_discrete_map=SENT_COLORS), width="stretch")
    with t_time:
        wk = sd.set_index("date").resample("W-MON", label="left", closed="left").sentiment_score.agg(["mean", "size"]).reset_index()
        fig = px.scatter(sd, x="date", y="sentiment_score", color="sentiment_label",
                         color_discrete_map=SENT_COLORS, hover_data=["outlet", "summary"],
                         labels={"sentiment_score": "Σκορ", "date": "", "sentiment_label": ""})
        fig.add_trace(go.Scatter(x=wk["date"] + pd.Timedelta(days=3), y=wk["mean"], mode="lines+markers",
                                 name="Εβδομαδιαίος μ.ό.", line=dict(color="#2c3e50")))
        festival_band(fig, fest_start, fest_end)
        st.plotly_chart(fig, width="stretch")
    with t_by:
        sb = pd.crosstab(sd.category, sd.sentiment_label, normalize="index").mul(100).round(1).reset_index()
        sb = sb.melt(id_vars="category", var_name="Sentiment", value_name="%")
        fig = px.bar(sb, x="%", y="category", color="Sentiment", orientation="h",
                     color_discrete_map=SENT_COLORS, labels={"category": ""})
        st.plotly_chart(fig, width="stretch")
    with t_type:
        tt = sd.groupby(["article_type", "sentiment_label"]).size().reset_index(name="Δημοσιεύματα")
        fig = px.bar(tt, x="Δημοσιεύματα", y="article_type", color="sentiment_label", orientation="h",
                     color_discrete_map=SENT_COLORS, labels={"article_type": "", "sentiment_label": ""})
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, width="stretch")
    with t_pers:
        pp = sd.persons.fillna("").str.split("|").explode().str.strip()
        pp = pp[pp.ne("")].value_counts().head(25).reset_index()
        pp.columns = ["Πρόσωπο", "Αναφορές"]
        fig = px.bar(pp, x="Αναφορές", y="Πρόσωπο", orientation="h", color_discrete_sequence=["#8e44ad"],
                     labels={"Πρόσωπο": ""}, height=max(400, 22 * len(pp)))
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, width="stretch")
    with t_list:
        st.dataframe(sd.sort_values("sentiment_score")[["date", "outlet", "sentiment_label", "sentiment_score",
                                                        "article_type", "summary", "url"]],
                     width="stretch", hide_index=True,
                     column_config={"url": st.column_config.LinkColumn("Σύνδεσμος"),
                                    "date": st.column_config.DateColumn("Ημ/νία", format="DD/MM")})
    st.caption("⚠️ Οι επισημειώσεις LLM είναι αυτόματες· για δημοσίευση αποτελεσμάτων προτείνεται "
               "έλεγχος δείγματος από άνθρωπο (π.χ. 10–15%).")

st.divider()

# ═════════════════════════════════════════════
# 6. ΠΙΝΑΚΑΣ & ΑΝΑΦΟΡΕΣ
# ═════════════════════════════════════════════
st.subheader("📋 Αναλυτικά Στοιχεία & Αναφορές")
show_cols = [c for c in ["date", "date_source", "period", "wave", "outlet", "category", "reach", "kind",
                         "title", "word_count", "sentiment_label", "sentiment_score", "url"] if c in fdf]
st.dataframe(fdf.sort_values("date")[show_cols], width="stretch", hide_index=True, height=350,
             column_config={"url": st.column_config.LinkColumn("Σύνδεσμος"),
                            "date": st.column_config.DateColumn("Ημ/νία", format="DD/MM/YYYY")})


def build_report() -> str:
    L = [f"# Αναφορά δημοσιότητας — {title}", "",
         f"*Παραγωγή: {datetime.now():%d/%m/%Y %H:%M} · Φεστιβάλ: {fest_start:%d/%m}–{fest_end:%d/%m/%Y}*", "",
         "## Ποσοτικά", "",
         f"- **{len(fdf)}** δημοσιεύματα σε **{fdf.outlet.nunique()}** μέσα, "
         f"από {fdf.date.min():%d/%m} έως {fdf.date.max():%d/%m/%Y} ({daily.size} ημέρες με κάλυψη).",
         f"- Κορυφαία ημέρα: **{daily.idxmax():%d/%m}** με {daily.max()} δημοσιεύματα.",
         f"- Κατά τη διάρκεια του φεστιβάλ: **{during:.0f}%** της κάλυψης.",
         f"- Εθνικής εμβέλειας μέσα: **{fdf.reach.eq('Εθνική').mean() * 100:.0f}%** των δημοσιευμάτων.",
         f"- Συγκέντρωση (Gini): {gini(outlet_counts.values):.2f}.", "",
         "### Ανά περίοδο", ""]
    for p, n in fdf.period.value_counts().sort_index().items():
        L.append(f"- {p}: {n}")
    L += ["", "### Ανά δελτίο τύπου", ""]
    for _, r in wv.iterrows():
        L.append(f"- {r.wave}: {r.Δημοσιεύματα} δημοσιεύματα σε {r.Μέσα} μέσα")
    L += ["", "### Top 10 μέσα", ""]
    for o, n in outlet_counts.head(10).items():
        L.append(f"- {o}: {n}")
    L += ["", "## Ποιοτικά", ""]
    if has_text:
        txo = fdf[fdf.status == "ok"]
        L.append(f"- Κείμενα: {len(txo)} · διάμεσος {txo.word_count.median():.0f} λέξεις · "
                 f"σύνολο {int(txo.word_count.sum()):,} λέξεις.".replace(",", "."))
    L.append(f"- Θεματικές ({th_src}):")
    for t, n in ex.themes_list.value_counts().head(8).items():
        L.append(f"  - {t}: {n} ({n / len(tdf) * 100:.0f}%)")
    if has_annot:
        sdd = fdf[fdf.sentiment_label.notna()]
        L.append(f"- Sentiment (n={len(sdd)}): μέσο σκορ {sdd.sentiment_score.mean():+.2f}· " +
                 ", ".join(f"{k} {v:.0f}%" for k, v in sdd.sentiment_label.value_counts(normalize=True).mul(100).items()))
    L += ["", "---", "*Μεθοδολογία: αποδελτίωση .docx · θεματικές με λεξικό λέξεων-κλειδιών ή Gemini · "
          "sentiment ως στάση απέναντι στη διοργάνωση.*"]
    return "\n".join(L)


report_md = build_report()
c1, c2, c3 = st.columns(3)
xls_sheets = {
    "Δημοσιεύματα": fdf[show_cols],
    "Ανά ημέρα": daily.rename_axis("Ημερομηνία").reset_index(name="Δημοσιεύματα"),
    "Ανά μέσο": fdf.groupby(["outlet", "category", "reach"]).size().reset_index(name="Δημοσιεύματα")
                   .sort_values("Δημοσιεύματα", ascending=False),
    "Ανά Δ.Τ.": wv,
    "Θεματικές": ex.themes_list.value_counts().rename_axis("Θεματική").reset_index(name="Δημοσιεύματα"),
}
c1.download_button("📥 Excel (όλοι οι πίνακες)", to_excel(xls_sheets),
                   file_name="festival_press_report.xlsx", width="stretch")
c2.download_button("📝 Αναφορά (Markdown)", report_md.encode("utf-8"),
                   file_name="festival_press_report.md", width="stretch")
c3.download_button("📄 CSV δημοσιευμάτων", fdf[show_cols].to_csv(index=False).encode("utf-8-sig"),
                   file_name="festival_press.csv", width="stretch")
with st.expander("👁️ Προεπισκόπηση αναφοράς"):
    st.markdown(report_md)

st.divider()

# ═════════════════════════════════════════════
# 7. ΕΜΠΛΟΥΤΙΣΜΟΣ ΔΕΔΟΜΕΝΩΝ
# ═════════════════════════════════════════════
with st.expander("⚙️ Εμπλουτισμός δεδομένων (λήψη κειμένων & επισημείωση Gemini)"):
    st.markdown(
        "Προτεινόμενη ροή: εκτέλεση **τοπικά** με `python enrich.py fetch` και "
        "`python enrich.py annotate`, και commit των `data/articles.csv` / `data/annotations.csv`. "
        "Εναλλακτικά από εδώ — τα αποτελέσματα κρατιούνται μόνο για την τρέχουσα συνεδρία, "
        "οπότε κατεβάστε τα CSV και ανεβάστε τα στο repo.")
    from enrich import annotate_all, fetch_all

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**1. Λήψη κειμένων** (trafilatura)")
        retry = st.checkbox("Επανάληψη όσων απέτυχαν", value=False)
        if st.button("▶️ Λήψη", width="stretch"):
            bar = st.progress(0.0, "Έναρξη…")
            res = fetch_all(clips, existing=articles, retry_failed=retry,
                            progress=lambda i, n, u: bar.progress(i / max(n, 1), f"{i}/{n} · {u[:60]}"))
            st.session_state["articles"] = res
            st.rerun()
        if articles is not None:
            st.download_button("💾 articles.csv", articles.to_csv(index=False).encode("utf-8"),
                               file_name="articles.csv", width="stretch")
    with a2:
        st.markdown("**2. Επισημείωση** (Gemini)")
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:  # noqa: BLE001
            key = ""
        key = key or st.text_input("GEMINI_API_KEY", type="password")
        model = st.text_input("Μοντέλο", "gemini-2.5-flash-lite")
        delay = st.number_input("Καθυστέρηση ανά αίτημα (δευτ.)", 0.0, 30.0, 6.0, 0.5,
                                help="Free tier: ~10–15 αιτήματα/λεπτό.")
        if st.button("▶️ Επισημείωση", width="stretch", disabled=not (key and articles is not None)):
            bar = st.progress(0.0, "Έναρξη…")
            res = annotate_all(articles, title or "Φεστιβάλ", key, model=model, delay=delay, existing=annots,
                               progress=lambda i, n, u: bar.progress(i / max(n, 1), f"{i}/{n} · {u[:60]}"))
            st.session_state["annotations"] = res
            st.rerun()
        if annots is not None:
            st.download_button("💾 annotations.csv", annots.to_csv(index=False).encode("utf-8"),
                               file_name="annotations.csv", width="stretch")
