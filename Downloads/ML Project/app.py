"""
app.py  —  Streamlit Dashboard
Fake Product Hype Detection System
Run: streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Hype Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  .risk-high   { background:#ff4b4b; color:white; padding:6px 14px;
                 border-radius:20px; font-weight:700; font-size:1.1rem; }
  .risk-medium { background:#ffa500; color:white; padding:6px 14px;
                 border-radius:20px; font-weight:700; font-size:1.1rem; }
  .risk-low    { background:#21c354; color:white; padding:6px 14px;
                 border-radius:20px; font-weight:700; font-size:1.1rem; }
</style>
""", unsafe_allow_html=True)

RISK_CSS   = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}
RISK_EMOJI = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
COLOR_MAP  = {"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#21c354"}


# ════════════════════════════════════════════════════════════
# Chart helpers
# ════════════════════════════════════════════════════════════

def risk_badge(level):
    cls = RISK_CSS.get(level, "risk-low")
    return f'<span class="{cls}">{RISK_EMOJI.get(level,"")} {level} Risk</span>'


def gauge_chart(score):
    color = "#21c354" if score < 33 else "#ffa500" if score < 66 else "#ff4b4b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": "Hype Score", "font": {"size": 22}},
        number={"font": {"size": 44, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white",
                     "tickfont": {"color": "white"}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)", "bordercolor": "gray",
            "steps": [
                {"range": [0,  33], "color": "rgba(33,195,84,0.15)"},
                {"range": [33, 66], "color": "rgba(255,165,0,0.15)"},
                {"range": [66,100], "color": "rgba(255,75,75,0.15)"},
            ],
            "threshold": {"line": {"color": color, "width": 4},
                          "thickness": 0.85, "value": score},
        },
    ))
    fig.update_layout(height=280, margin=dict(l=30,r=30,t=40,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color":"white"})
    return fig


def trend_chart(reviews_df, product_id):
    df = reviews_df[reviews_df["product_id"] == product_id].copy()
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby(df["date"].dt.date).size().reset_index()
    daily.columns = ["date", "count"]
    daily["date"]     = pd.to_datetime(daily["date"])
    daily["rolling7"] = daily["count"].rolling(7, min_periods=1).mean()
    q75 = daily["count"].quantile(0.75)
    burst = daily[daily["count"] > q75 * 1.5]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["date"], y=daily["count"],
                         name="Daily Reviews", marker_color="rgba(99,110,250,0.7)"))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["rolling7"],
                             name="7-day Avg", line=dict(color="#ffa500", width=2.5)))
    if len(burst):
        fig.add_trace(go.Scatter(x=burst["date"], y=burst["count"], mode="markers",
                                 name="Burst Days",
                                 marker=dict(color="#ff4b4b", size=10, symbol="x")))
    fig.update_layout(
        title=f"Review Volume — {product_id}", height=320,
        xaxis_title="Date", yaxis_title="Reviews",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,48,0.5)",
        font={"color":"white"}, margin=dict(l=30,r=20,t=50,b=30),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def rating_chart(reviews_df, product_id):
    df = reviews_df[reviews_df["product_id"] == product_id]
    counts = df["rating"].value_counts().reindex([1,2,3,4,5], fill_value=0)
    colors = ["#ff4b4b","#ff8c42","#ffd166","#06d6a0","#118ab2"]
    fig = go.Figure(go.Bar(
        x=[f"{int(r)}★" for r in counts.index], y=counts.values,
        marker_color=colors, text=counts.values,
        textposition="outside", textfont={"color":"white"},
    ))
    fig.update_layout(
        title=f"Rating Distribution — {product_id}", height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,48,0.5)",
        font={"color":"white"}, margin=dict(l=20,r=20,t=45,b=20),
    )
    return fig


def signal_chart(row):
    signals = {
        "Text Fake Prob":   float(row.get("text_score", 0)),
        "Temporal Anomaly": float(row.get("temporal_score", 0)),
        "5-Star %":         float(row.get("five_star_pct", 0)),
        "Burst Ratio":      float(row.get("burst_ratio", 0)),
    }
    values = [min(v*100, 100) for v in signals.values()]
    colors = ["#ff4b4b" if v>60 else "#ffa500" if v>35 else "#21c354" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=list(signals.keys()), orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside", textfont={"color":"white"},
    ))
    fig.update_layout(
        title="Signal Breakdown", height=240,
        xaxis=dict(range=[0,110], title="Score (0–100)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,48,0.5)",
        font={"color":"white"}, margin=dict(l=20,r=60,t=45,b=20),
    )
    return fig


def explanation_text(row):
    score    = float(row["hype_score"])
    risk     = str(row["risk_level"])
    text_s   = float(row.get("text_score", 0))
    temp_s   = float(row.get("temporal_score", 0))
    five_s   = float(row.get("five_star_pct", 0))
    burst    = float(row.get("burst_ratio", 0))
    daily_mx = int(row.get("daily_max", 0))

    reasons = []
    if text_s > 0.45:
        reasons.append(
            f"🚩 **High language similarity** to known fake reviews "
            f"(text score: {text_s:.2f}).")
    if temp_s > 0.55:
        reasons.append(
            f"🚩 **Unusual review burst detected** — peak of {daily_mx} "
            f"reviews in a single day (temporal score: {temp_s:.2f}).")
    if five_s > 0.7:
        reasons.append(
            f"🚩 **{five_s*100:.0f}% five-star reviews** — statistically uncommon.")
    if burst > 0.25:
        reasons.append(
            f"🚩 **{burst*100:.0f}% of active days are burst days** — "
            f"reviews not arriving organically.")
    if not reasons:
        reasons.append("✅ No strong hype signals detected.")

    return (
        f"**Hype Score: {score:.1f}/100 — {risk} Risk**\n\n"
        f"**Key findings:**\n\n" + "\n\n".join(reasons)
    )


# ════════════════════════════════════════════════════════════
# Pipeline runner  (cached in session_state by filename+size)
# ════════════════════════════════════════════════════════════

def run_full_pipeline_with_progress(csv_bytes: bytes, filename: str,
                                    cache_key: str = ""):
    from src.data.loader           import load_and_clean
    from src.models.text_model     import TextModel
    from src.models.temporal_model import TemporalModel
    from src.models.fusion         import compute_hype_scores

    if not cache_key:
        import hashlib
        cache_key = f"pipeline__auto__{hashlib.md5(csv_bytes).hexdigest()}"
    metrics_key = cache_key.replace("pipeline__", "metrics__")

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    with st.status("Running pipeline...", expanded=True) as status:

        st.write("📂 Step 1/4 — Saving uploaded file...")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(csv_bytes)
            tmp_path = f.name
        st.write(f"   File size: {len(csv_bytes)/1_048_576:.1f} MB")

        st.write("🧹 Step 2/4 — Loading & cleaning data...")
        clean_df = load_and_clean(tmp_path,
                                  save_path="data/processed/clean_reviews.csv",
                                  verbose=False)
        st.write(f"   ✅ {len(clean_df):,} reviews · "
                 f"{clean_df['product_id'].nunique():,} products loaded")

        st.write("🤖 Step 3/4 — Training text model (TF-IDF + LR)...")
        tm = TextModel()
        metrics = tm.train("data/processed/clean_reviews.csv", verbose=False)
        text_df = tm.predict("data/processed/clean_reviews.csv",
                             save_path="data/features/text_probs.csv",
                             verbose=False)
        auc_str = f" · ROC-AUC: {metrics['roc_auc']}" if "roc_auc" in metrics else ""
        st.write(f"   ✅ Text model trained{auc_str}")
        if "report" in metrics:
            r = metrics["report"]
            fk = r.get("Fake", r.get("1", {}))
            st.write(
                f"   Precision: {fk.get('precision',0):.3f} · "
                f"Recall: {fk.get('recall',0):.3f} · "
                f"F1: {fk.get('f1-score',0):.3f}"
            )

        st.write("📈 Step 4/4 — Training temporal anomaly detector...")
        tp = TemporalModel()
        t_metrics = tp.train("data/processed/clean_reviews.csv", verbose=False)
        tp.predict("data/processed/clean_reviews.csv",
                   save_path="data/features/temporal_scores.csv",
                   verbose=False)
        st.write(
            f"   ✅ {t_metrics['n_anomalies']} anomalous products "
            f"({t_metrics['anomaly_pct']}%) [{t_metrics['n_estimators']} trees]"
        )

        st.write("⚡ Computing Hype Scores...")
        hype_df = compute_hype_scores(verbose=False)
        high = (hype_df["risk_level"] == "High").sum()
        st.write(f"   ✅ {len(hype_df)} products scored · {high} High Risk")

        status.update(label="✅ Pipeline complete!", state="complete", expanded=False)

    clean_with_prob = clean_df.merge(
        text_df[["review_id", "fake_prob"]], on="review_id", how="left"
    )
    result = (hype_df, clean_with_prob)
    st.session_state[cache_key]                  = result
    st.session_state[metrics_key + "__text"]     = metrics
    st.session_state[metrics_key + "__temporal"] = t_metrics
    return result


@st.cache_data(show_spinner=False)
def load_precomputed():
    hype_df  = pd.read_csv("data/features/hype_scores.csv")
    clean_df = pd.read_csv("data/processed/clean_reviews.csv",
                           parse_dates=["date"])
    text_df  = pd.read_csv("data/features/text_probs.csv")
    clean_with_prob = clean_df.merge(
        text_df[["review_id", "fake_prob"]], on="review_id", how="left"
    )
    return hype_df, clean_with_prob


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/detective.png", width=64)
    st.title("Hype Detection")
    st.caption("Fake Product Hype Detection System  v1.0")
    st.divider()

    st.subheader("📂 Data Source")
    data_mode = st.radio(
        "Choose data source:",
        ["Upload CSV file", "Use pre-computed results"],
        index=1,
    )

    uploaded = None
    if data_mode == "Upload CSV file":
        uploaded = st.file_uploader(
            "Upload review CSV", type=["csv"],
            help="Required columns: review_text, rating, date\nOptional: product_id, is_fake",
        )
        st.caption("First upload runs pipeline once.\nSame file = instant from cache.")
    else:
        if not Path("data/features/hype_scores.csv").exists():
            st.warning("No pre-computed results found.\nRun:\n```\npython run_pipeline.py\n```")

    st.divider()
    st.subheader("⚙️ Display Settings")
    show_reviews = st.checkbox("Show review table", value=True)
    max_reviews  = st.slider("Max reviews to show", 10, 200, 50)
    st.divider()
    st.markdown(
        "**Score Formula**\n\n"
        "```\nHype = 0.45 × text_prob\n"
        "     + 0.35 × temporal\n"
        "     + 0.10 × five_star%\n"
        "     + 0.10 × burst_ratio\n"
        "```\n\n"
        "**Risk Thresholds**\n"
        "- 🟢 Low:    0–32\n"
        "- 🟡 Medium: 33–65\n"
        "- 🔴 High:   66–100"
    )


# ════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════

st.title("🔍 Fake Product Hype Detection System")
st.caption("Detects artificially inflated product popularity using NLP + temporal analysis.")

hype_df           = None
reviews_df        = None
_text_metrics     = None
_temporal_metrics = None

if data_mode == "Upload CSV file" and uploaded is not None:
    _cache_key   = f"pipeline__{uploaded.name}__{uploaded.size}"
    _metrics_key = f"metrics__{uploaded.name}__{uploaded.size}"

    if _cache_key in st.session_state:
        hype_df, reviews_df = st.session_state[_cache_key]
        _text_metrics     = st.session_state.get(_metrics_key + "__text")
        _temporal_metrics = st.session_state.get(_metrics_key + "__temporal")
    else:
        file_bytes = uploaded.read()
        file_mb    = len(file_bytes) / 1_048_576
        if file_mb > 150:
            st.error(
                f"File is {file_mb:.0f} MB — too large. "
                "Use a CSV under 150 MB, or run:  python run_pipeline.py --data your_file.csv"
            )
            st.stop()
        try:
            hype_df, reviews_df = run_full_pipeline_with_progress(
                file_bytes, uploaded.name, _cache_key
            )
            _text_metrics     = st.session_state.get(_metrics_key + "__text")
            _temporal_metrics = st.session_state.get(_metrics_key + "__temporal")
        except Exception as e:
            st.error(f"**Pipeline error:** {e}")
            st.exception(e)
            st.stop()

elif data_mode == "Use pre-computed results":
    _text_metrics    = None
    _temporal_metrics = None
    if Path("data/features/hype_scores.csv").exists():
        try:
            hype_df, reviews_df = load_precomputed()
        except Exception as e:
            st.error(f"Error loading results: {e}")
            st.stop()
    else:
        st.info(
            "👈 No data loaded yet.\n\n"
            "Upload a CSV, or run:\n```\npython run_pipeline.py\n```"
        )
        st.stop()

if hype_df is None:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
    st.stop()


# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔎 Product Drill-Down",
    "📋 Full Leaderboard",
    "🧠 Model Metrics",
])


# ────────────────────────────────────────────────────────────
# TAB 1: Overview
# ────────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    high_count = (hype_df["risk_level"] == "High").sum()
    med_count  = (hype_df["risk_level"] == "Medium").sum()
    low_count  = (hype_df["risk_level"] == "Low").sum()
    with c1: st.metric("Total Products", len(hype_df))
    with c2: st.metric("🔴 High Risk",   high_count,
                        delta=f"{high_count/len(hype_df)*100:.0f}%",
                        delta_color="inverse")
    with c3: st.metric("🟡 Medium Risk", med_count)
    with c4: st.metric("🟢 Low Risk",   low_count)

    st.divider()

    hype_df["_color"] = hype_df["risk_level"].map(COLOR_MAP)
    fig_scatter = px.scatter(
        hype_df, x="text_score", y="temporal_score",
        color="risk_level", color_discrete_map=COLOR_MAP,
        size="hype_score", size_max=30,
        hover_name="product_id",
        hover_data={"hype_score": True, "risk_level": False,
                    "_color": False, "total_reviews": True},
        title="Risk Map: Text Score vs Temporal Score",
        labels={"text_score":"Text Fake Score","temporal_score":"Temporal Anomaly"},
        height=420,
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,48,0.5)",
        font={"color":"white"}, legend_title="Risk Level",
    )
    st.plotly_chart(fig_scatter, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(
            hype_df, x="hype_score", nbins=20,
            color="risk_level", color_discrete_map=COLOR_MAP,
            title="Hype Score Distribution",
            labels={"hype_score":"Hype Score (0–100)"}, height=300,
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(30,33,48,0.5)",
                               font={"color":"white"})
        st.plotly_chart(fig_hist, width='stretch')

    with col_b:
        rc = hype_df["risk_level"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=rc.index, values=rc.values,
            marker_colors=[COLOR_MAP[r] for r in rc.index],
            hole=0.42, textinfo="label+percent",
        ))
        fig_pie.update_layout(title="Risk Level Distribution", height=300,
                              paper_bgcolor="rgba(0,0,0,0)", font={"color":"white"})
        st.plotly_chart(fig_pie, width='stretch')

    st.subheader("⚠️ Top 5 Highest Hype Score Products")
    top5 = hype_df.nlargest(5, "hype_score")[
        ["product_id","hype_score","risk_level","total_reviews","daily_max"]
    ].reset_index(drop=True)
    top5.index += 1
    st.dataframe(top5, width='stretch')


# ────────────────────────────────────────────────────────────
# TAB 2: Product Drill-Down
# ────────────────────────────────────────────────────────────
with tab2:
    _sorted = hype_df.sort_values("hype_score", ascending=False)
    product_list = _sorted["product_id"].tolist()
    _label_map = {
        row["product_id"]: f"{row['product_id']}  —  {row['hype_score']:.1f} pts  {row['risk_level']} Risk"
        for _, row in _sorted[["product_id","hype_score","risk_level"]].iterrows()
    }

    selected = st.selectbox(
        "Select product to inspect:",
        product_list,
        format_func=lambda pid: _label_map.get(pid, pid),
    )

    row = hype_df[hype_df["product_id"] == selected].iloc[0]

    g_col, info_col = st.columns([1, 1.4])
    with g_col:
        st.plotly_chart(gauge_chart(float(row["hype_score"])), width='stretch')
        st.markdown(risk_badge(str(row["risk_level"])), unsafe_allow_html=True)
    with info_col:
        st.markdown("### Quick Stats")
        st.markdown(f"**Total Reviews:** {int(row['total_reviews'])}")
        st.markdown(f"**Peak Daily Reviews:** {int(row['daily_max'])}")
        st.markdown(f"**Avg Daily Reviews:** {float(row['daily_mean']):.1f}")
        st.markdown(f"**5-Star Rate:** {float(row['five_star_pct'])*100:.1f}%")
        st.markdown(f"**Burst Ratio:** {float(row['burst_ratio'])*100:.1f}%")
        st.markdown(f"**Text Fake Score:** {float(row.get('text_score',0)):.4f}")
        st.markdown(f"**Temporal Score:** {float(row.get('temporal_score',0)):.4f}")

    st.divider()

    with st.expander("📋 Explanation Report", expanded=True):
        st.markdown(explanation_text(row))

    c1, c2 = st.columns(2)
    with c1:
        if reviews_df is not None:
            st.plotly_chart(trend_chart(reviews_df, selected), width='stretch')
    with c2:
        if reviews_df is not None:
            st.plotly_chart(rating_chart(reviews_df, selected), width='stretch')

    st.plotly_chart(signal_chart(row), width='stretch')

    if show_reviews and reviews_df is not None:
        st.subheader("📝 Individual Reviews")
        prod_reviews = reviews_df[reviews_df["product_id"] == selected].copy()
        prod_reviews["date"] = pd.to_datetime(prod_reviews["date"], errors="coerce")
        if "fake_prob" in prod_reviews.columns:
            prod_reviews = prod_reviews.sort_values("fake_prob", ascending=False)
        display_cols = [c for c in
            ["review_text","rating","date","fake_prob","is_fake","reviewer_id"]
            if c in prod_reviews.columns]
        st.dataframe(prod_reviews[display_cols].head(max_reviews),
                     width='stretch', height=320)


# ────────────────────────────────────────────────────────────
# TAB 3: Leaderboard
# ────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📋 All Products — Ranked by Hype Score")

    f1, f2 = st.columns(2)
    with f1:
        risk_filter = st.multiselect("Filter by risk:",
            ["High","Medium","Low"], default=["High","Medium","Low"])
    with f2:
        score_range = st.slider("Hype score range:", 0, 100, (0, 100))

    filtered = hype_df[
        (hype_df["risk_level"].isin(risk_filter)) &
        (hype_df["hype_score"].between(*score_range))
    ].reset_index(drop=True)
    filtered.index += 1

    disp = filtered[[
        "product_id","hype_score","risk_level",
        "text_score","temporal_score",
        "five_star_pct","burst_ratio","total_reviews","daily_max",
    ]].copy()
    disp["five_star_pct"] = (disp["five_star_pct"] * 100).round(1).astype(str) + "%"

    st.dataframe(disp, width='stretch', height=500)
    st.download_button(
        label="⬇️ Download Results CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="hype_scores_export.csv",
        mime="text/csv",
    )


# ────────────────────────────────────────────────────────────
# TAB 4: Model Metrics
# ────────────────────────────────────────────────────────────
with tab4:
    st.header("🧠 Model Performance Metrics")

    if _text_metrics is None and _temporal_metrics is None:
        st.info(
            "Upload a CSV file to see model metrics.\n\n"
            "Pre-computed results do not include training metrics."
        )
    else:
        # ── Text Model ────────────────────────────────────────
        st.subheader("📝 Text Model — TF-IDF + Logistic Regression")
        tm = _text_metrics or {}

        if tm.get("mode") == "supervised" and "report" in tm:
            r  = tm["report"]
            fk = r.get("Fake", r.get("1", {}))
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1: st.metric("ROC-AUC",        f"{tm['roc_auc']:.4f}")
            with k2: st.metric("Fake Precision",  f"{fk.get('precision',0):.3f}")
            with k3: st.metric("Fake Recall",     f"{fk.get('recall',0):.3f}")
            with k4: st.metric("Fake F1",         f"{fk.get('f1-score',0):.3f}")
            with k5: st.metric("Accuracy",        f"{r.get('accuracy',0):.3f}")
            st.caption(f"Train: {tm.get('train_n',0):,}  ·  Test: {tm.get('test_n',0):,}")
            st.divider()

            col_roc, col_cm = st.columns(2)
            with col_roc:
                if tm.get("roc_fpr"):
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=tm["roc_fpr"], y=tm["roc_tpr"], mode="lines",
                        name=f"ROC (AUC={tm['roc_auc']:.4f})",
                        line=dict(color="#636efa", width=2.5),
                        fill="tozeroy", fillcolor="rgba(99,110,250,0.12)",
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0,1], y=[0,1], mode="lines",
                        name="Baseline", line=dict(color="gray", dash="dash"),
                    ))
                    fig_roc.update_layout(
                        title="ROC Curve", height=360,
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(30,33,48,0.5)",
                        font={"color":"white"}, legend=dict(x=0.55, y=0.15),
                    )
                    st.plotly_chart(fig_roc, width='stretch')

            with col_cm:
                cm = tm.get("confusion_matrix")
                if cm:
                    fig_cm = go.Figure(go.Heatmap(
                        z=cm, x=["Genuine","Fake"], y=["Genuine","Fake"],
                        colorscale=[[0,"rgba(30,33,48,0.8)"],[1,"rgba(99,110,250,0.9)"]],
                        text=[[str(v) for v in r_] for r_ in cm],
                        texttemplate="%{text}", textfont={"size":22,"color":"white"},
                        showscale=True,
                    ))
                    fig_cm.update_layout(
                        title="Confusion Matrix", height=360,
                        xaxis_title="Predicted", yaxis_title="Actual",
                        paper_bgcolor="rgba(0,0,0,0)", font={"color":"white"},
                    )
                    st.plotly_chart(fig_cm, width='stretch')

            # Classification report table
            st.subheader("📋 Classification Report")
            rows_data = []
            for cls_name in ["Genuine","Fake","macro avg","weighted avg"]:
                key = cls_name if cls_name in r else {"Genuine":"0","Fake":"1"}.get(cls_name, cls_name)
                if key in r and isinstance(r[key], dict):
                    rows_data.append({
                        "Class":     cls_name,
                        "Precision": round(r[key].get("precision",0), 4),
                        "Recall":    round(r[key].get("recall",0), 4),
                        "F1-Score":  round(r[key].get("f1-score",0), 4),
                        "Support":   int(r[key].get("support",0)),
                    })
            if rows_data:
                st.dataframe(pd.DataFrame(rows_data).set_index("Class"), width='stretch')

            # Top words
            fw = tm.get("top_fake_words", [])
            gw = tm.get("top_genuine_words", [])
            if fw or gw:
                st.subheader("🔤 Top Discriminative Words")
                wc1, wc2 = st.columns(2)
                with wc1:
                    if fw:
                        fw_df = pd.DataFrame(fw).head(20)
                        fig_fw = go.Figure(go.Bar(
                            x=fw_df["score"], y=fw_df["word"], orientation="h",
                            marker_color="rgba(255,75,75,0.8)",
                            text=fw_df["score"].round(3).astype(str),
                            textposition="outside", textfont={"color":"white"},
                        ))
                        fig_fw.update_layout(
                            title="Fake-Indicating Words", height=480,
                            yaxis=dict(autorange="reversed"),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(30,33,48,0.5)",
                            font={"color":"white"}, margin=dict(l=10,r=60,t=50,b=20),
                        )
                        st.plotly_chart(fig_fw, width='stretch')
                with wc2:
                    if gw:
                        gw_df = pd.DataFrame(gw).head(20)
                        gw_df["score"] = gw_df["score"].abs()
                        fig_gw = go.Figure(go.Bar(
                            x=gw_df["score"], y=gw_df["word"], orientation="h",
                            marker_color="rgba(33,195,84,0.8)",
                            text=gw_df["score"].round(3).astype(str),
                            textposition="outside", textfont={"color":"white"},
                        ))
                        fig_gw.update_layout(
                            title="Genuine-Indicating Words", height=480,
                            yaxis=dict(autorange="reversed"),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(30,33,48,0.5)",
                            font={"color":"white"}, margin=dict(l=10,r=60,t=50,b=20),
                        )
                        st.plotly_chart(fig_gw, width='stretch')
        else:
            st.warning(
                f"Text model ran in **{tm.get('mode','unknown')}** mode — "
                "no `is_fake` labels found. Add an `is_fake` column for full metrics."
            )

        # ── Temporal Model ────────────────────────────────────
        st.divider()
        st.subheader("📈 Temporal Model — Isolation Forest")
        tmet = _temporal_metrics or {}
        if tmet:
            t1, t2, t3, t4_kpi = st.columns(4)
            with t1: st.metric("Products",   f"{tmet.get('n_products',0):,}")
            with t2: st.metric("Anomalies",  f"{tmet.get('n_anomalies',0):,}")
            with t3: st.metric("Anomaly %",  f"{tmet.get('anomaly_pct',0):.1f}%")
            with t4_kpi: st.metric("Trees",  tmet.get("n_estimators","—"))

            fa = tmet.get("feat_means_anomaly", {})
            fn = tmet.get("feat_means_normal", {})
            fc = tmet.get("feature_cols", [])
            if fa and fn and fc:
                st.subheader("📊 Feature Comparison: Anomalous vs Normal")
                feat_comp = pd.DataFrame({
                    "Feature":   fc,
                    "Anomalous": [fa.get(c, 0) for c in fc],
                    "Normal":    [fn.get(c, 0) for c in fc],
                })
                fig_feat = go.Figure()
                fig_feat.add_trace(go.Bar(name="Anomalous", x=feat_comp["Feature"],
                                          y=feat_comp["Anomalous"],
                                          marker_color="rgba(255,75,75,0.85)"))
                fig_feat.add_trace(go.Bar(name="Normal", x=feat_comp["Feature"],
                                          y=feat_comp["Normal"],
                                          marker_color="rgba(33,195,84,0.85)"))
                fig_feat.update_layout(
                    barmode="group", title="Mean Feature Values", height=400,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,48,0.5)",
                    font={"color":"white"}, xaxis=dict(tickangle=-30),
                    legend=dict(orientation="h", y=1.12),
                )
                st.plotly_chart(fig_feat, width='stretch')

                feat_comp["Ratio"] = (feat_comp["Anomalous"] /
                                      (feat_comp["Normal"] + 1e-9)).round(2)
                feat_comp_s = feat_comp.sort_values("Ratio", ascending=False)
                fig_ratio = go.Figure(go.Bar(
                    x=feat_comp_s["Feature"], y=feat_comp_s["Ratio"],
                    marker_color=["rgba(255,75,75,0.85)" if v>1.2
                                  else "rgba(255,165,0,0.85)" if v>0.9
                                  else "rgba(33,195,84,0.85)"
                                  for v in feat_comp_s["Ratio"]],
                    text=feat_comp_s["Ratio"].astype(str) + "×",
                    textposition="outside", textfont={"color":"white"},
                ))
                fig_ratio.update_layout(
                    title="Feature Ratio: Anomalous ÷ Normal", height=360,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,48,0.5)",
                    font={"color":"white"}, xaxis=dict(tickangle=-30),
                )
                st.plotly_chart(fig_ratio, width='stretch')
                st.subheader("📋 Feature Stats Table")
                st.dataframe(
                    feat_comp[["Feature","Anomalous","Normal","Ratio"]].set_index("Feature"),
                    width='stretch',
                )

        # ── Hype Score Analysis ───────────────────────────────
        st.divider()
        st.subheader("📊 Hype Score Analysis")
        sc1, sc2 = st.columns(2)
        with sc1:
            fig_box = go.Figure()
            for risk in ["High","Medium","Low"]:
                grp = hype_df[hype_df["risk_level"] == risk]["hype_score"]
                fig_box.add_trace(go.Box(
                    y=grp, name=f"{risk} ({len(grp)})",
                    marker_color=COLOR_MAP[risk], boxmean=True,
                ))
            fig_box.update_layout(
                title="Hype Score by Risk Level", yaxis_title="Hype Score", height=360,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,33,48,0.5)",
                font={"color":"white"},
            )
            st.plotly_chart(fig_box, width='stretch')
        with sc2:
            fig_d = px.scatter(
                hype_df, x="five_star_pct", y="burst_ratio",
                color="hype_score", color_continuous_scale="RdYlGn_r",
                size="total_reviews", size_max=25,
                hover_name="product_id",
                hover_data={"hype_score": True, "risk_level": True},
                title="5-Star % vs Burst Ratio",
                height=360,
            )
            fig_d.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(30,33,48,0.5)",
                                font={"color":"white"})
            st.plotly_chart(fig_d, width='stretch')

        st.subheader("📋 Complete Product Stats")
        nc = ["hype_score","text_score","temporal_score",
              "five_star_pct","burst_ratio","total_reviews","daily_max"]
        disp_full = hype_df.sort_values("hype_score", ascending=False)[
            ["product_id","risk_level"] + [c for c in nc if c in hype_df.columns]
        ].reset_index(drop=True)
        disp_full.index += 1
        disp_full["five_star_pct"] = (disp_full["five_star_pct"] * 100).round(1)
        st.dataframe(disp_full, width='stretch', height=500)
        st.download_button(
            label="⬇️ Download Full Stats CSV",
            data=disp_full.to_csv(index=False).encode("utf-8"),
            file_name="product_stats_full.csv",
            mime="text/csv",
        )


# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Fake Product Hype Detection  ·  "
    "Models: TF-IDF + LR (text) · Isolation Forest (temporal)  ·  "
    "Built with Streamlit + Plotly"
)
