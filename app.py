# =============================================================================
#  NYC Airbnb ML Pipeline Dashboard
#  Tech Stack : Python · Pandas · Scikit-Learn · Matplotlib · Seaborn · Streamlit
#  Dataset    : AB_NYC_2019.csv
#  Run with   : streamlit run airbnb_ml_dashboard.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")          # Suppress all runtime warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # Non-interactive backend (thread-safe)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Airbnb ML Pipeline",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — dark accent card style
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 18px 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { font-size: 0.9rem; opacity: 0.85; margin: 4px 0 0; }
    .predict-box {
        background: linear-gradient(135deg, #1a4731 0%, #2e7d52 100%);
        border-radius: 14px;
        padding: 24px 32px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .predict-box h1 { font-size: 3rem; margin: 0; }
    .predict-box p  { opacity: 0.9; font-size: 1.1rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SIDEBAR — DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_Bélo.svg/320px-Airbnb_Logo_Bélo.svg.png",
    width=160,
)
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload AB_NYC_2019.csv",
    type=["csv"],
    help="Upload the NYC Airbnb 2019 dataset to begin.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Hyperparameters**")
n_estimators = st.sidebar.slider("n_estimators", 10, 200, 50, 10)
max_depth    = st.sidebar.slider("max_depth",     3,  20, 10,  1)
test_size    = st.sidebar.slider("Test split %",  10,  40, 20,  5) / 100
price_cap    = st.sidebar.slider("Price cap ($)", 500, 2000, 1000, 100)

st.sidebar.markdown("---")
st.sidebar.info(
    "**NYC Airbnb ML Pipeline**\n\n"
    "End-to-end pipeline:\n"
    "EDA → Cleaning → Encoding → "
    "Train/Test Split → K-Fold CV → "
    "Random Forest → Live Predictor"
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: close all matplotlib figures after rendering
# ─────────────────────────────────────────────────────────────────────────────
def close_figs():
    plt.close("all")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CACHED PIPELINE  (re-runs only when inputs change)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes, n_est, m_depth, t_size, p_cap):
    """
    Full end-to-end ML pipeline.
    Returns all artefacts needed by the 4 dashboard tabs.
    """

    # ── 2.1  LOAD ────────────────────────────────────────────────────────────
    import io
    raw_df = pd.read_csv(io.BytesIO(file_bytes))

    # ── 2.2  EDA ARTEFACTS ───────────────────────────────────────────────────
    shape       = raw_df.shape
    head_df     = raw_df.head(10)
    missing_ser = raw_df.isnull().sum()
    missing_ser = missing_ser[missing_ser > 0].sort_values(ascending=False)

    # Missing-values bar chart
    fig_miss, ax_miss = plt.subplots(figsize=(8, 3))
    if len(missing_ser):
        sns.barplot(x=missing_ser.values, y=missing_ser.index,
                    palette="Reds_r", ax=ax_miss)
        ax_miss.set_title("Missing Values per Column", fontsize=13, fontweight="bold")
        ax_miss.set_xlabel("Count")
        for bar in ax_miss.patches:
            ax_miss.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                         f"{int(bar.get_width()):,}", va="center", fontsize=9)
    else:
        ax_miss.text(0.5, 0.5, "No Missing Values  🎉",
                     ha="center", va="center", fontsize=14, transform=ax_miss.transAxes)
        ax_miss.axis("off")
    fig_miss.tight_layout()

    # Price distribution histogram (raw, before cap)
    fig_hist, axes = plt.subplots(1, 2, figsize=(12, 4))
    raw_prices = raw_df["price"].clip(upper=p_cap * 2)
    sns.histplot(raw_prices, bins=60, color="#e84393",
                 edgecolor="white", linewidth=0.4, ax=axes[0])
    axes[0].set_title("Price Distribution (raw)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Price ($)")
    axes[0].xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))

    capped = raw_df.loc[raw_df["price"] <= p_cap, "price"]
    sns.histplot(capped, bins=60, color="#3b82f6",
                 edgecolor="white", linewidth=0.4, ax=axes[1])
    axes[1].set_title(f"Price Distribution (capped ≤ ${p_cap:,})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Price ($)")
    axes[1].xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    fig_hist.tight_layout()

    # Borough price comparison
    fig_borough, ax_borough = plt.subplots(figsize=(9, 4))
    borough_order = (raw_df.groupby("neighbourhood_group")["price"]
                     .median().sort_values(ascending=False).index.tolist())
    sns.boxplot(data=raw_df[raw_df["price"] <= p_cap],
                x="neighbourhood_group", y="price",
                order=borough_order, palette="Set2", ax=ax_borough)
    ax_borough.set_title("Price by Borough (capped)", fontsize=12, fontweight="bold")
    ax_borough.set_xlabel("")
    ax_borough.set_ylabel("Price ($)")
    ax_borough.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    fig_borough.tight_layout()

    # Room-type count
    fig_room, ax_room = plt.subplots(figsize=(6, 3.5))
    room_counts = raw_df["room_type"].value_counts()
    colors = ["#6366f1", "#3b82f6", "#22d3ee"]
    ax_room.pie(room_counts.values, labels=room_counts.index,
                autopct="%1.1f%%", colors=colors,
                startangle=140, wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax_room.set_title("Room Type Distribution", fontsize=12, fontweight="bold")
    fig_room.tight_layout()

    # Correlation heatmap
    num_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
    corr = raw_df[num_cols].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, annot_kws={"size": 8}, ax=ax_corr)
    ax_corr.set_title("Correlation Matrix", fontsize=12, fontweight="bold")
    fig_corr.tight_layout()

    # ── 2.3  MAP DATA (pre-cap, for geo view) ────────────────────────────────
    map_df = raw_df[["latitude", "longitude", "price", "neighbourhood_group",
                     "room_type", "name"]].copy()
    map_df = map_df[(map_df["price"] > 0) & (map_df["price"] <= p_cap)].dropna()

    # ── 2.4  DATA ENGINEERING & CLEANING ────────────────────────────────────
    df = raw_df.copy()

    # Drop text / ID columns not useful for modelling
    drop_cols = ["id", "name", "host_id", "host_name",
                 "neighbourhood", "last_review"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Fill missing reviews_per_month with 0 (no reviews → 0)
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    # Drop any remaining NaN rows
    df.dropna(inplace=True)

    # Cap price at configured threshold to remove extreme outliers
    df = df[df["price"] > 0]        # Remove free/zero listings
    df = df[df["price"] <= p_cap]   # Cap at user-defined ceiling

    # ── 2.5  FEATURE SELECTION & LABEL ENCODING ──────────────────────────────
    le_borough   = LabelEncoder()
    le_room      = LabelEncoder()

    df["neighbourhood_group"] = le_borough.fit_transform(df["neighbourhood_group"])
    df["room_type"]           = le_room.fit_transform(df["room_type"])

    # Store encoder classes for the live predictor
    borough_classes = le_borough.classes_.tolist()   # e.g. ['Bronx','Brooklyn',...]
    room_classes    = le_room.classes_.tolist()       # e.g. ['Entire home/apt',...]

    # ── 2.6  TRAIN / TEST SPLIT ──────────────────────────────────────────────
    FEATURES = ["neighbourhood_group", "room_type", "minimum_nights",
                "number_of_reviews", "reviews_per_month",
                "calculated_host_listings_count", "availability_365",
                "latitude", "longitude"]

    X = df[FEATURES]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=t_size, random_state=42
    )

    # ── 2.7  MODEL TRAINING + K-FOLD CROSS VALIDATION ────────────────────────
    model = RandomForestRegressor(
        n_estimators=n_est,
        max_depth=m_depth,
        random_state=42,
        n_jobs=-1,
    )

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_neg_mse = cross_val_score(model, X_train, y_train,
                                 cv=kf, scoring="neg_mean_squared_error")
    cv_rmse_scores = np.sqrt(-cv_neg_mse)
    mean_cv_rmse   = float(cv_rmse_scores.mean())

    # Final fit on full training set
    model.fit(X_train, y_train)

    # ── 2.8  TEST SET PERFORMANCE ────────────────────────────────────────────
    y_pred    = model.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_r2   = float(r2_score(y_test, y_pred))

    # Actual vs Predicted scatter
    sample_idx = np.random.choice(len(y_test), size=min(800, len(y_test)), replace=False)
    fig_avp, ax_avp = plt.subplots(figsize=(7, 5))
    ax_avp.scatter(y_test.values[sample_idx], y_pred[sample_idx],
                   alpha=0.35, color="#6366f1", s=18, edgecolors="none")
    lims = [0, p_cap]
    ax_avp.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax_avp.set_xlabel("Actual Price ($)")
    ax_avp.set_ylabel("Predicted Price ($)")
    ax_avp.set_title("Actual vs Predicted (test set sample)", fontsize=12, fontweight="bold")
    ax_avp.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax_avp.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax_avp.legend()
    fig_avp.tight_layout()

    # Residual distribution
    residuals = y_test.values - y_pred
    fig_res, ax_res = plt.subplots(figsize=(7, 4))
    sns.histplot(residuals, bins=60, color="#f59e0b", edgecolor="white",
                 linewidth=0.4, ax=ax_res, kde=True)
    ax_res.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax_res.set_title("Residual Distribution", fontsize=12, fontweight="bold")
    ax_res.set_xlabel("Residual (Actual − Predicted)")
    fig_res.tight_layout()

    # Feature importance
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature":    FEATURES,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
    colors_feat = plt.cm.Blues(np.linspace(0.3, 0.9, len(feat_imp_df)))
    ax_feat.barh(feat_imp_df["Feature"], feat_imp_df["Importance"],
                 color=colors_feat, edgecolor="white", height=0.65)
    ax_feat.set_title("Feature Importances (Random Forest)", fontsize=12, fontweight="bold")
    ax_feat.set_xlabel("Importance Score")
    for bar in ax_feat.patches:
        ax_feat.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{bar.get_width():.3f}", va="center", fontsize=8)
    fig_feat.tight_layout()

    # ── 2.9  BUNDLE ALL ARTEFACTS ────────────────────────────────────────────
    return dict(
        # Dataset
        shape=shape,
        head_df=head_df,
        missing_ser=missing_ser,
        # EDA figures
        fig_miss=fig_miss,
        fig_hist=fig_hist,
        fig_borough=fig_borough,
        fig_room=fig_room,
        fig_corr=fig_corr,
        # Map
        map_df=map_df,
        # Metrics
        mean_cv_rmse=mean_cv_rmse,
        cv_rmse_scores=cv_rmse_scores,
        test_rmse=test_rmse,
        test_r2=test_r2,
        fig_avp=fig_avp,
        fig_res=fig_res,
        fig_feat=fig_feat,
        # Live predictor assets
        model=model,
        le_borough_classes=borough_classes,
        le_room_classes=room_classes,
        features=FEATURES,
        train_rows=len(X_train),
        test_rows=len(X_test),
        price_cap=p_cap,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.title("🗽 NYC Airbnb — End-to-End ML Pipeline Dashboard")
st.markdown(
    "An interactive machine-learning pipeline: **EDA → Cleaning → Encoding "
    "→ Train/Test Split → K-Fold CV → Random Forest → Live Predictor**"
)
st.markdown("---")

if uploaded_file is None:
    st.info(
        "👈  **Upload `AB_NYC_2019.csv`** from the sidebar to launch the pipeline.",
        icon="🚀",
    )
    st.stop()

# ── Run the pipeline (cached) ──────────────────────────────────────────────
with st.spinner("⚙️  Running ML pipeline…"):
    artefacts = run_pipeline(
        uploaded_file.getvalue(),
        n_estimators, max_depth, test_size, price_cap
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4.  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_eda, tab_map, tab_metrics, tab_predictor = st.tabs([
    "📊 Tab 1 — EDA",
    "🗺️ Tab 2 — NYC Map",
    "📈 Tab 3 — Metrics",
    "🤖 Tab 4 — Live Predictor",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("📊 Exploratory Data Analysis")

    # Dataset overview cards
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f"""<div class="metric-card">
            <h2>{artefacts['shape'][0]:,}</h2>
            <p>Total Rows</p></div>""", unsafe_allow_html=True
    )
    col2.markdown(
        f"""<div class="metric-card">
            <h2>{artefacts['shape'][1]}</h2>
            <p>Columns</p></div>""", unsafe_allow_html=True
    )
    col3.markdown(
        f"""<div class="metric-card">
            <h2>{artefacts['train_rows']:,}</h2>
            <p>Training Rows</p></div>""", unsafe_allow_html=True
    )
    col4.markdown(
        f"""<div class="metric-card">
            <h2>{artefacts['test_rows']:,}</h2>
            <p>Test Rows</p></div>""", unsafe_allow_html=True
    )

    st.markdown("#### 🔍 Raw Data Preview (first 10 rows)")
    st.dataframe(artefacts["head_df"], use_container_width=True)

    st.markdown("#### ❓ Missing Values")
    st.pyplot(artefacts["fig_miss"])
    close_figs()

    st.markdown("#### 💰 Price Distributions")
    st.pyplot(artefacts["fig_hist"])
    close_figs()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 🏙️ Price by Borough")
        st.pyplot(artefacts["fig_borough"])
        close_figs()
    with col_r:
        st.markdown("#### 🛏️ Room Type Mix")
        st.pyplot(artefacts["fig_room"])
        close_figs()

    st.markdown("#### 🔗 Correlation Matrix")
    st.pyplot(artefacts["fig_corr"])
    close_figs()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — NYC MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.header("🗺️ Geospatial Price Map")
    st.caption(
        f"Each dot is a listing (price ≤ ${artefacts['price_cap']:,}), "
        "colour-coded by price. Zoom and pan to explore."
    )

    map_df = artefacts["map_df"].copy()

    # Streamlit's built-in map needs lat/lon columns; for colour use PyDeck
    try:
        import pydeck as pdk

        # Normalise price → 0-255 for colour mapping (red channel)
        map_df["norm"] = (map_df["price"] / artefacts["price_cap"] * 255).clip(0, 255).astype(int)
        map_df["r"] = map_df["norm"]
        map_df["g"] = (255 - map_df["norm"])
        map_df["b"] = 80
        map_df["a"] = 160

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["longitude", "latitude"],
            get_fill_color=["r", "g", "b", "a"],
            get_radius=80,
            pickable=True,
        )
        view = pdk.ViewState(latitude=40.7128, longitude=-73.99, zoom=10, pitch=30)
        tooltip = {
            "html": "<b>{name}</b><br/>Price: ${price}<br/>{neighbourhood_group} · {room_type}",
            "style": {"backgroundColor": "#1e293b", "color": "white", "fontSize": "13px"},
        }
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                  tooltip=tooltip,
                                  map_style="mapbox://styles/mapbox/dark-v9"))

    except ImportError:
        # Fallback: Streamlit's native scatter_map (requires lat/lon columns)
        st.info("💡 Install `pydeck` for an interactive 3-D map. Showing basic map.")
        st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"}))

    # Price heat-map (matplotlib fallback – always works)
    st.markdown("#### 📍 Matplotlib Scatter Overview (colour = price)")
    fig_geo, ax_geo = plt.subplots(figsize=(11, 8))
    sc = ax_geo.scatter(
        map_df["longitude"], map_df["latitude"],
        c=map_df["price"], cmap="plasma", s=2, alpha=0.4
    )
    plt.colorbar(sc, ax=ax_geo, label="Price ($)")
    ax_geo.set_title("NYC Airbnb Listings — Price Heatmap", fontsize=13, fontweight="bold")
    ax_geo.set_xlabel("Longitude")
    ax_geo.set_ylabel("Latitude")
    fig_geo.tight_layout()
    st.pyplot(fig_geo)
    close_figs()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METRICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.header("📈 Model Performance Metrics")

    # Top KPI cards
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"""<div class="metric-card">
            <h2>${artefacts['mean_cv_rmse']:.2f}</h2>
            <p>Mean CV RMSE (3-Fold)</p></div>""", unsafe_allow_html=True
    )
    col2.markdown(
        f"""<div class="metric-card">
            <h2>${artefacts['test_rmse']:.2f}</h2>
            <p>Test RMSE</p></div>""", unsafe_allow_html=True
    )
    col3.markdown(
        f"""<div class="metric-card">
            <h2>{artefacts['test_r2']:.4f}</h2>
            <p>R² Score (Test Set)</p></div>""", unsafe_allow_html=True
    )

    st.markdown("---")

    # CV fold detail
    st.markdown("#### 🔄 3-Fold Cross-Validation Detail")
    cv_df = pd.DataFrame({
        "Fold":  [f"Fold {i+1}" for i in range(len(artefacts["cv_rmse_scores"]))],
        "RMSE ($)": artefacts["cv_rmse_scores"].round(2),
    })
    st.dataframe(cv_df, use_container_width=False)

    st.markdown("---")
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown("#### 🎯 Actual vs Predicted")
        st.pyplot(artefacts["fig_avp"])
        close_figs()
    with col_r2:
        st.markdown("#### 📉 Residual Distribution")
        st.pyplot(artefacts["fig_res"])
        close_figs()

    st.markdown("#### 🏆 Feature Importances")
    st.pyplot(artefacts["fig_feat"])
    close_figs()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predictor:
    st.header("🤖 Live Price Predictor")
    st.markdown(
        "Adjust the inputs below and click **Predict Price** to get a real-time "
        "estimate from the trained Random Forest model."
    )

    col_inp1, col_inp2 = st.columns(2)

    with col_inp1:
        st.markdown("#### 🏙️ Location & Room")
        borough   = st.selectbox("Borough", artefacts["le_borough_classes"])
        room_type = st.selectbox("Room Type", artefacts["le_room_classes"])

        st.markdown("#### 📅 Availability")
        availability_365 = st.slider("Availability (days/year)", 0, 365, 180)

    with col_inp2:
        st.markdown("#### 🛏️ Stay Requirements")
        min_nights = st.slider("Minimum Nights", 1, 30, 3)
        num_reviews = st.slider("Number of Reviews", 0, 600, 50)
        reviews_pm  = st.slider("Reviews per Month", 0.0, 20.0, 1.5, 0.1)

        st.markdown("#### 🏠 Host Details")
        host_listings = st.slider("Host Listings Count", 1, 100, 1)

    st.markdown("---")

    # Centre the predict button
    _, btn_col, _ = st.columns([2, 1, 2])
    predict_clicked = btn_col.button("🔮 Predict Price", use_container_width=True)

    if predict_clicked:
        # Encode categorical inputs using the stored label-encoder classes
        borough_enc   = artefacts["le_borough_classes"].index(borough)
        room_enc      = artefacts["le_room_classes"].index(room_type)

        # Build input vector matching the training FEATURES order
        # ["neighbourhood_group","room_type","minimum_nights","number_of_reviews",
        #  "reviews_per_month","calculated_host_listings_count","availability_365",
        #  "latitude","longitude"]
        # For lat/lon we use the median of the selected borough from the map_df
        borough_geo = artefacts["map_df"][
            artefacts["map_df"]["neighbourhood_group"] == borough
        ][["latitude", "longitude"]].median()
        lat = float(borough_geo.get("latitude", 40.7128))
        lon = float(borough_geo.get("longitude", -73.9857))

        X_input = np.array([[
            borough_enc, room_enc, min_nights, num_reviews,
            reviews_pm, host_listings, availability_365, lat, lon
        ]])

        predicted_price = float(artefacts["model"].predict(X_input)[0])
        predicted_price = max(10.0, predicted_price)   # Sanity floor

        st.markdown(
            f"""<div class="predict-box">
                <p>Estimated nightly price for a <strong>{room_type}</strong>
                   in <strong>{borough}</strong></p>
                <h1>${predicted_price:,.0f} / night</h1>
                <p style="font-size:0.85rem;opacity:0.75;">
                    Model: Random Forest · n_estimators={n_estimators} · max_depth={max_depth}
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

        # Breakdown expander
        with st.expander("🔍 View input features passed to model"):
            debug_df = pd.DataFrame({
                "Feature": artefacts["features"],
                "Value":   X_input[0],
            })
            st.dataframe(debug_df, use_container_width=False)