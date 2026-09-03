"""
Notebook 06 — Held-out Validation
E-Commerce Product Selection Framework
MSc Data Analytics Dissertation — Syed Aqeel (24060071)
London Metropolitan University — Supervisor: Dr. Subeksha
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Niche Discovery Framework",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour scheme ─────────────────────────────────────────────────────
NAVY   = '#1B2A4A'
ORANGE = '#E8833A'
TEAL   = '#2E86AB'
PURPLE = '#6B4C9A'
GREEN  = '#2E7D32'
LIGHT  = '#F0F4F8'

NICHE_COLOURS = {
    'Kitchen Appliances' : '#1B8A3C',  # Strong green
    'Bedding'            : '#E8833A',  # Orange
    'Home Appliances'    : '#1565C0',  # Bright blue
    'Bathroom'           : '#7B1FA2',  # Bright purple
}

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8FAFC; }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1B2A4A 0%, #2E4A7A 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: white;
    }
    .app-header p {
        color: #C5D5E8;
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
    }
    .app-header .badge {
        display: inline-block;
        background: #E8833A;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* Section headers */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1B2A4A;
        border-left: 4px solid #E8833A;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
        color: #1B2A4A;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
    .metric-card .delta {
        font-size: 0.85rem;
        color: #2E7D32;
        font-weight: 600;
    }

    /* Rank badge */
    .rank-badge {
        display: inline-block;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: #1B2A4A;
        color: white;
        font-weight: 800;
        font-size: 0.9rem;
        line-height: 32px;
        text-align: center;
    }

    /* Verdict badges */
    .badge-validated {
        background: #E6F4EA;
        color: #2E7D32;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #2E7D32;
    }
    .badge-partial {
        background: #FFF8E1;
        color: #F57F17;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        border: 1px solid #F57F17;
    }

    /* Info box */
    .info-box {
        background: #EFF6FF;
        border-left: 4px solid #2E86AB;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1E3A5F;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #E2E8F0;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# DATA — hardcoded from actual notebook outputs
# ════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():

    # ── Original composite scores (Notebook 05) ──────────────────────
    # Exact values from Notebook 05 Step 16
    original_scores = pd.DataFrame({
        'niche'              : ['Kitchen Appliances', 'Bathroom',
                                'Bedding', 'Home Appliances'],
        'composite_score'    : [0.5453, 0.4560, 0.3607, 0.3168],
        'demand_score'       : [0.3004, 0.3000, 0.7906, 0.2508],
        'competition_score'  : [0.6000, 0.7299, 0.0281, 0.5734],
        'sentiment_gap_score': [1.0000, 0.2540, 0.0000, 0.3634],
        'pricing_score'      : [0.2757, 0.6000, 0.5825, 0.0366],
        'review_count'       : [6503, 6497, 12033, 6723],
        'rank'               : [1, 2, 3, 4],
    })

    # Exact values from Notebook 06 Step 12 (corrected)
    validation_scores = pd.DataFrame({
        'niche'              : ['Kitchen Appliances', 'Bathroom',
                                'Home Appliances', 'Bedding'],
        'composite_score'    : [0.5747, 0.4654, 0.4561, 0.4028],
        'demand_score'       : [0.3307, 0.2500, 0.4168, 0.8415],
        'competition_score'  : [0.6000, 0.6985, 0.5826, 0.0615],
        'sentiment_gap_score': [1.0000, 0.3830, 0.6951, 0.0000],
        'pricing_score'      : [0.3775, 0.6000, 0.0581, 0.6749],
        'rank'               : [1, 2, 3, 4],
    })

    # ── Aspect sentiment gaps (Notebook 06 Step 10, corrected, held-out 200k) ──
    aspects = pd.DataFrame({
        'aspect'         : ['smell', 'noise', 'durability', 'quality',
                            'price', 'performance', 'size', 'cleaning',
                            'comfort', 'appearance', 'ease_of_use'],
        'Kitchen Appliances': [0.2218, 0.1365, 0.0317, 0.0421,
                               0.0098, 0.0237, -0.0054, -0.0256,
                               0.0826, 0.0388, -0.0153],
        'Bedding'           : [0.0971, -0.1009, -0.0388, -0.0102,
                               -0.0449, -0.0020, -0.0230, 0.0628,
                               -0.0628, 0.0026, -0.0129],
        'Home Appliances'   : [0.0500, -0.0396, 0.0645, 0.0829,
                               0.0207, 0.0326, 0.0088, 0.0056,
                               0.0350, 0.0588, 0.0156],
        'Bathroom'          : [0.0659, -0.0535, -0.0135, 0.0233,
                               0.0319, 0.0197, 0.0004, 0.0448,
                               0.0090, -0.0107, 0.0077],
    })

    # ── Model comparison (Notebook 04) ───────────────────────────────
    models = pd.DataFrame({
        'Model'               : ['Logistic Regression', 'Random Forest',
                                 'XGBoost', 'LSTM'],
        'Accuracy (%)'        : [93.52, 92.44, 89.66, 92.31],
        'Weighted F1'         : [0.9379, 0.9185, 0.8786, 0.9259],
        'ROC-AUC'             : [0.9784, 0.9633, 0.9596, 0.9627],
        'Negative Recall'     : [0.9252, 0.6160, 0.4040, 0.8769],
        'Training Time (s)'   : [1.2, 222.6, 285.7, 88.5],
    })

    # ── Sentiment model generalisation ────────────────────────────────
    # Exact values from Notebooks 04 and 06
    generalisation = pd.DataFrame({
        'Scale'       : ['Original (93k)', 'Held-out 200k'],
        'Accuracy'    : [93.52, 92.84],
        'F1'          : [0.9379, 0.9315],
        'ROC-AUC'     : [0.9784, 0.9772],
        'Neg. Recall' : [92.52, 92.30],
    })

    # ── N-gram comparison ─────────────────────────────────────────────
    ngrams = pd.DataFrame({
        'Gram Level'          : ['Bigram (2)', 'Trigram (3)', 'Four-gram (4)'],
        'Top Phrase Frequency': [5648, 625, 109],
        'Niche Hits (Top 20)' : [4, 9, 9],
        'Niche Hit Rate (%)'  : [20.0, 45.0, 45.0],
        'Selected'            : ['No', 'Yes — Optimal', 'No'],
    })

    return original_scores, validation_scores, aspects, models, generalisation, ngrams

original_scores, validation_scores, aspects, models, generalisation, ngrams = load_data()

# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='background:{NAVY};padding:1rem;border-radius:8px;margin-bottom:1rem;'>
        <div style='color:{ORANGE};font-weight:800;font-size:1.1rem;'>📦 Niche Discovery</div>
        <div style='color:#C5D5E8;font-size:0.8rem;margin-top:0.3rem;'>
        Amazon Home & Kitchen<br>MSc Dissertation Tool
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚖️ Adjust Component Weights")
    st.markdown(
        "<div class='info-box'>Drag the sliders and pick your niches, then press <b>Apply</b> to update the charts below.</div>",
        unsafe_allow_html=True
    )

    # Defaults live in session_state so the charts keep showing the last
    # *applied* configuration, not whatever the sliders happen to say
    # mid-drag — this is what the Apply button below controls.
    if "applied_weights" not in st.session_state:
        st.session_state.applied_weights = {
            "demand": 30, "competition": 25, "sentiment": 25, "pricing": 20
        }
    if "applied_niches" not in st.session_state:
        st.session_state.applied_niches = [
            'Kitchen Appliances', 'Bedding', 'Home Appliances', 'Bathroom'
        ]

    w_demand = st.slider(
        "📈 Demand", 0, 100, st.session_state.applied_weights["demand"], 5,
        help="Review volume, recency, helpful votes and Google Trends"
    )
    w_competition = st.slider(
        "🏆 Competition", 0, 100, st.session_state.applied_weights["competition"], 5,
        help="Unique products, Gini coefficient and unique sellers"
    )
    w_sentiment = st.slider(
        "💬 Sentiment Gap", 0, 100, st.session_state.applied_weights["sentiment"], 5,
        help="Aspect-level negativity vs category average"
    )
    w_pricing = st.slider(
        "💰 Pricing Stability", 0, 100, st.session_state.applied_weights["pricing"], 5,
        help="Price variance and price sentiment negativity"
    )

    raw_total = w_demand + w_competition + w_sentiment + w_pricing
    if raw_total > 100:
        st.error(
            f"❌ Total weight is {raw_total}%. The combined weight cannot exceed 100%. "
            "Please reduce the weights before applying."
        )
    elif raw_total < 100:
        st.warning(
            f"⚠️ Total weight is {raw_total}%. The combined weight must equal 100%. "
            "Please adjust the weights before applying."
        )
    else:
        st.success("✅ Total weight: 100% — ready to apply.")

    st.markdown("---")
    st.markdown("### 🔍 Filter Niches")
    niche_selection = st.multiselect(
        "Select niches to display",
        options=['Kitchen Appliances', 'Bedding',
                 'Home Appliances', 'Bathroom'],
        default=st.session_state.applied_niches
    )

    apply_clicked = st.button("✅ Apply weights & filters", use_container_width=True)
    if apply_clicked:
        if raw_total > 100:
            st.error(
                f"Can't apply — total weight is {raw_total}%. "
                "The combined weight cannot exceed 100%."
            )
        elif raw_total < 100:
            st.error(
                f"Can't apply — total weight is {raw_total}%. "
                "The combined weight must equal 100%."
            )
        elif not niche_selection:
            st.error("Can't apply — select at least one niche.")
        else:
            st.session_state.applied_weights = {
                "demand": w_demand, "competition": w_competition,
                "sentiment": w_sentiment, "pricing": w_pricing,
            }
            st.session_state.applied_niches = niche_selection
            st.toast("Weights and filters applied ✅")

    # Everything below the sidebar uses the *applied* state, not the
    # live slider positions — so nothing updates until Apply is pressed.
    aw = st.session_state.applied_weights
    w_demand, w_competition = aw["demand"], aw["competition"]
    w_sentiment, w_pricing = aw["sentiment"], aw["pricing"]
    selected_niches = st.session_state.applied_niches
    weights_valid = (w_demand + w_competition + w_sentiment + w_pricing == 100)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem;color:#94A3B8;'>
    <b>Data source:</b> McAuley Amazon Reviews 2023<br>
    <b>Category:</b> Home & Kitchen<br>
    <b>Corpus:</b> 93,052 reviews<br>
    <b>Validation:</b> 200,000 reviews<br>
    <b>Models:</b> VADER → RoBERTa → PyABSA → LR<br>
    <b>GitHub:</b> aqeel-gardezi/MSc-Project
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='app-header'>
    <h1>📦 Amazon Product Selection Framework</h1>
    <p>A data-driven tool for identifying high-demand, low-competition niches using NLP and Machine Learning</p>
    <span class='badge'>Home & Kitchen · 67M Reviews · MSc Dissertation</span>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SECTION 1 — CORPUS OVERVIEW
# ════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-title'>📊 Corpus Overview</div>",
            unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "67.4M", "Total Reviews", "Home & Kitchen 2023"),
    (c2, "93,052", "Corpus Sample", "Reservoir sampled"),
    (c3, "66,179", "Unique Products", "Avg 1.4 reviews/product"),
    (c4, "4 Niches", "Discovered", "From corpus data"),
    (c5, "200k", "Validation Set", "Full-pipeline held-out"),
]
for col, val, label, sub in metrics:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='value'>{val}</div>
            <div class='label'>{label}</div>
            <div class='delta'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SECTION 2 — LIVE COMPOSITE SCORING
# ════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-title'>🎯 Live Composite Scoring</div>",
            unsafe_allow_html=True)

# Recompute scores only when the last applied weights total exactly 100%.
_applied_total = w_demand + w_competition + w_sentiment + w_pricing
if weights_valid and selected_niches:
    wD = w_demand / 100
    wC = w_competition / 100
    wS = w_sentiment / 100
    wP = w_pricing / 100

    live_scores = original_scores[
        original_scores['niche'].isin(selected_niches)
    ].copy()

    live_scores['live_composite'] = (
        live_scores['demand_score']        * wD +
        live_scores['competition_score']   * wC +
        live_scores['sentiment_gap_score'] * wS +
        live_scores['pricing_score']       * wP
    )
    live_scores = live_scores.sort_values(
        'live_composite', ascending=False
    ).reset_index(drop=True)
    live_scores['live_rank'] = live_scores.index + 1

    # Bar chart
    fig_live = go.Figure()
    for _, row in live_scores.iterrows():
        fig_live.add_trace(go.Bar(
            x=[row['niche']],
            y=[row['live_composite']],
            name=row['niche'],
            marker_color=NICHE_COLOURS.get(row['niche'], NAVY),
            text=[f"{row['live_composite']:.4f}"],
            textposition='outside',
            textfont=dict(size=13, color=NAVY),
        ))

    fig_live.update_layout(
        title=dict(
            text=f"Live Composite Scores (Demand {w_demand}% · Competition {w_competition}% · Sentiment {w_sentiment}% · Pricing {w_pricing}%)",
            font=dict(size=14, color=NAVY)
        ),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(range=[0, 0.85], gridcolor='#F0F4F8',
                   title='Composite Score'),
        xaxis=dict(title='Niche'),
        height=380,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    fig_live.add_hline(y=0.50, line_dash='dash',
                       line_color='#CCCCCC', annotation_text='0.50 reference')

    col_chart, col_rank = st.columns([2, 1])

    with col_chart:
        st.plotly_chart(fig_live, use_container_width=True, theme=None)

    with col_rank:
        st.markdown("**Live Rankings**")
        for _, row in live_scores.iterrows():
            colour = NICHE_COLOURS.get(row['niche'], NAVY)
            orig_rank = int(original_scores.loc[
                original_scores['niche'] == row['niche'], 'rank'
            ].values[0])
            live_rank = int(row['live_rank'])
            change = orig_rank - live_rank
            arrow = "↑" if change > 0 else ("↓" if change < 0 else "—")
            arrow_col = GREEN if change > 0 else ("#CC0000" if change < 0 else "#666")
            st.markdown(f"""
            <div style='background:white;border-left:4px solid {colour};
                        padding:0.6rem 0.8rem;margin:0.4rem 0;
                        border-radius:0 8px 8px 0;
                        box-shadow:0 1px 4px rgba(0,0,0,0.08);'>
                <span style='font-weight:800;color:{colour};'>#{live_rank}</span>
                <span style='margin-left:0.5rem;font-weight:600;color:{NAVY};'>
                    {row['niche']}</span><br>
                <span style='font-size:0.85rem;color:#64748B;'>
                    Score: {row['live_composite']:.4f} &nbsp;
                    <span style='color:{arrow_col};font-weight:700;'>
                        {arrow} vs original #{orig_rank}
                    </span>
                </span>
            </div>
            """, unsafe_allow_html=True)

else:
    st.warning("Adjust weights to sum to 100% and select at least one niche.")

# ════════════════════════════════════════════════════════════════════
# SECTION 3 — COMPONENT BREAKDOWN
# ════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-title'>🔍 Component Score Breakdown</div>",
            unsafe_allow_html=True)

col_radar, col_table = st.columns([1, 1])

with col_radar:
    # Radar chart
    categories = ['Demand', 'Competition', 'Sent. Gap', 'Pricing']
    fig_radar = go.Figure()

    for _, row in original_scores[
        original_scores['niche'].isin(selected_niches)
    ].iterrows():
        values = [
            row['demand_score'],
            row['competition_score'],
            row['sentiment_gap_score'],
            row['pricing_score'],
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=row['niche'],
            line=dict(color=NICHE_COLOURS.get(row['niche'], NAVY),
                      width=2.5),
            fillcolor=NICHE_COLOURS.get(row['niche'], NAVY),
            opacity=0.55,
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor='#E2E8F0'),
            angularaxis=dict(gridcolor='#E2E8F0'),
            bgcolor='white',
        ),
        showlegend=True,
        legend=dict(orientation='h', y=-0.15),
        paper_bgcolor='white',
        title=dict(text='Component Profiles — Original Corpus',
                   font=dict(size=13, color=NAVY)),
        height=380,
        margin=dict(t=50, b=60, l=40, r=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True, theme=None)

with col_table:
    st.markdown("**Original Corpus — Component Scores**")
    display_cols = ['niche', 'demand_score', 'competition_score',
                    'sentiment_gap_score', 'pricing_score',
                    'composite_score', 'rank']
    display_df = original_scores[
        original_scores['niche'].isin(selected_niches)
    ][display_cols].copy()
    display_df.columns = ['Niche', 'Demand', 'Competition',
                          'Sent. Gap', 'Pricing', 'Composite', 'Rank']
    display_df = display_df.sort_values('Rank')

    st.dataframe(
        display_df.style
        .format({'Demand': '{:.4f}', 'Competition': '{:.4f}',
                 'Sent. Gap': '{:.4f}', 'Pricing': '{:.4f}',
                 'Composite': '{:.4f}'})
        .highlight_max(subset=['Composite'], color='#C5E1C5')
        .highlight_min(subset=['Composite'], color='#F5C6C6')
        .set_properties(**{'text-align': 'center'}),
        use_container_width=True,
        height=200
    )

    st.markdown("**Weight Applied (Current — total 100%)**")
    weight_data = {
        'Component' : ['Demand', 'Competition', 'Sentiment Gap', 'Pricing'],
        'Weight'    : [f"{wD*100:.0f}%", f"{wC*100:.0f}%",
                       f"{wS*100:.0f}%", f"{wP*100:.0f}%"],
        'Signal Count': ['4 signals', '3 signals', '11 aspects', '2 signals'],
    }
    st.dataframe(pd.DataFrame(weight_data), use_container_width=True,
                 hide_index=True)

# ════════════════════════════════════════════════════════════════════
# SECTION 4 — ASPECT SENTIMENT GAPS
# ════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-title'>💬 Aspect Sentiment Gaps — Unmet Needs</div>",
            unsafe_allow_html=True)

st.markdown(
    "<div class='info-box'>Positive values = niche has more complaints than category average on that aspect → signals an unmet customer need and product opportunity.</div>",
    unsafe_allow_html=True
)

selected_niche_asp = st.selectbox(
    "Select niche to inspect",
    options=[n for n in ['Kitchen Appliances', 'Home Appliances',
                          'Bathroom', 'Bedding']
             if n in selected_niches],
    index=0
)

if selected_niche_asp:
    asp_data = aspects[['aspect', selected_niche_asp]].copy()
    asp_data.columns = ['Aspect', 'Gap']
    asp_data = asp_data.sort_values('Gap', ascending=False)

    colours_asp = [GREEN if v > 0 else '#CC0000' for v in asp_data['Gap']]

    fig_asp = go.Figure(go.Bar(
        x=asp_data['Gap'],
        y=asp_data['Aspect'],
        orientation='h',
        marker_color=colours_asp,
        text=[f"{v:+.4f}" for v in asp_data['Gap']],
        textposition='outside',
    ))
    fig_asp.add_vline(x=0, line_color=NAVY, line_width=1.5)
    fig_asp.update_layout(
        title=dict(
            text=f"Sentiment Gap by Aspect — {selected_niche_asp}  (green = unmet need, red = satisfied)",
            font=dict(size=13, color=NAVY)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(title='Gap vs Category Average', gridcolor='#F0F4F8',
                   zeroline=False),
        yaxis=dict(title=''),
        height=380,
        margin=dict(t=50, b=40, l=120, r=80),
    )
    st.plotly_chart(fig_asp, use_container_width=True, theme=None)

    # Top unmet needs
    top_gaps = asp_data[asp_data['Gap'] > 0].head(3)
    if not top_gaps.empty:
        st.markdown(f"**Top unmet needs in {selected_niche_asp}:**")
        cols_gap = st.columns(len(top_gaps))
        for i, (_, row) in enumerate(top_gaps.iterrows()):
            with cols_gap[i]:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='value' style='font-size:1.5rem;color:{ORANGE};'>
                        {row['Gap']:+.1%}</div>
                    <div class='label'>{row['Aspect'].replace('_', ' ').title()}</div>
                    <div class='delta'>Above category avg</div>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SECTION 5 — VALIDATION RESULTS
# ════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-title'>✅ Held-out Validation Results (200k)</div>",
            unsafe_allow_html=True)

col_v1, col_v2 = st.columns([1, 1])

with col_v1:
    # Side by side composite scores
    niches_order = ['Kitchen Appliances', 'Bedding',
                    'Home Appliances', 'Bathroom']
    orig_vals = [float(original_scores.loc[
        original_scores['niche'] == n, 'composite_score'
    ].values[0]) for n in niches_order]
    ho_vals = [float(validation_scores.loc[
        validation_scores['niche'] == n, 'composite_score'
    ].values[0]) for n in niches_order]

    fig_val = go.Figure()
    fig_val.add_trace(go.Bar(
        name='Original (93k)',
        x=niches_order, y=orig_vals,
        marker_color=NAVY, opacity=0.85,
        text=[f'{v:.4f}' for v in orig_vals],
        textposition='outside', textfont=dict(size=10)
    ))
    fig_val.add_trace(go.Bar(
        name='Held-out (200k)',
        x=niches_order, y=ho_vals,
        marker_color=ORANGE, opacity=0.85,
        text=[f'{v:.4f}' for v in ho_vals],
        textposition='outside', textfont=dict(size=10)
    ))
    fig_val.update_layout(
        barmode='group',
        title=dict(text='Composite Scores — Original vs Held-out (200k)',
                   font=dict(size=13, color=NAVY)),
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(range=[0, 0.85], gridcolor='#F0F4F8',
                   title='Composite Score'),
        legend=dict(orientation='h', y=-0.15),
        height=360,
        margin=dict(t=50, b=60, l=40, r=40),
    )
    fig_val.add_hline(y=0.50, line_dash='dash', line_color='#CCCCCC')
    st.plotly_chart(fig_val, use_container_width=True, theme=None)

with col_v2:
    # Sentiment model generalisation
    fig_gen = go.Figure()
    metrics_gen = ['Accuracy', 'F1', 'ROC-AUC']
    orig_metrics = [93.52/100, 0.9379, 0.9784]
    ho_metrics   = [92.84/100, 0.9315, 0.9772]

    fig_gen.add_trace(go.Bar(
        name='Original (93k)',
        x=metrics_gen, y=orig_metrics,
        marker_color=NAVY, opacity=0.85,
        text=[f'{v:.4f}' for v in orig_metrics],
        textposition='outside', textfont=dict(size=11)
    ))
    fig_gen.add_trace(go.Bar(
        name='Held-out (200k)',
        x=metrics_gen, y=ho_metrics,
        marker_color=ORANGE, opacity=0.85,
        text=[f'{v:.4f}' for v in ho_metrics],
        textposition='outside', textfont=dict(size=11)
    ))
    fig_gen.update_layout(
        barmode='group',
        title=dict(text='Sentiment Model Generalisation',
                   font=dict(size=13, color=NAVY)),
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(range=[0.88, 1.0], gridcolor='#F0F4F8',
                   title='Score'),
        legend=dict(orientation='h', y=-0.15),
        height=360,
        margin=dict(t=50, b=60, l=40, r=40),
    )
    st.plotly_chart(fig_gen, use_container_width=True, theme=None)

# Validation verdict cards
st.markdown("**Validation Verdict**")
v_cols = st.columns(3)
verdicts = [
    ('Sentiment Model', 'VALIDATED', GREEN,
     'Accuracy drop < 0.7pp · ROC-AUC 0.9772 vs 0.9784'),
    ('Niche Discovery', 'VALIDATED', GREEN,
     '4/4 niches replicated · Share deviation < 2%'),
    ('Composite Ranking', 'VALIDATED', GREEN,
     'Kitchen #1 & Bathroom #2 unchanged · Home Appliances moved to #3'),
]
for col, (comp, verdict, colour, evidence) in zip(v_cols, verdicts):
    with col:
        st.markdown(f"""
        <div style='background:white;border:1px solid #E2E8F0;
                    border-top:4px solid {colour};
                    border-radius:8px;padding:1rem;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
            <div style='font-weight:700;color:{NAVY};font-size:0.95rem;'>
                {comp}</div>
            <div style='margin:0.4rem 0;'>
                <span style='background:{colour}1A;color:{colour};
                             font-weight:800;font-size:0.85rem;
                             padding:0.2rem 0.7rem;border-radius:20px;
                             border:1px solid {colour};'>
                    {verdict}
                </span>
            </div>
            <div style='font-size:0.82rem;color:#64748B;'>{evidence}</div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SECTION 6 — ML MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-title'>🤖 ML Model Comparison</div>",
            unsafe_allow_html=True)

col_m1, col_m2 = st.columns([1, 1])

with col_m1:
    fig_models = go.Figure()
    colours_m = [NAVY, TEAL, ORANGE, PURPLE]
    for i, row in models.iterrows():
        fig_models.add_trace(go.Bar(
            name=row['Model'],
            x=[row['Model']],
            y=[row['Accuracy (%)']],
            marker_color=colours_m[i],
            text=[f"{row['Accuracy (%)']:.2f}%"],
            textposition='outside',
        ))
    fig_models.update_layout(
        title=dict(text='Model Accuracy Comparison',
                   font=dict(size=13, color=NAVY)),
        showlegend=False,
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(range=[85, 97], gridcolor='#F0F4F8',
                   title='Accuracy (%)'),
        height=320,
        margin=dict(t=50, b=40, l=40, r=40),
    )
    st.plotly_chart(fig_models, use_container_width=True, theme=None)

with col_m2:
    st.markdown("**Model Performance Summary**")
    st.dataframe(
        models.style
        .format({'Accuracy (%)': '{:.2f}',
                 'Weighted F1': '{:.4f}',
                 'ROC-AUC': '{:.4f}',
                 'Negative Recall': '{:.4f}',
                 'Training Time (s)': '{:.1f}'})
        .highlight_max(subset=['Accuracy (%)', 'Weighted F1',
                                'ROC-AUC', 'Negative Recall'],
                       color='#E6F4EA')
        .highlight_min(subset=['Training Time (s)'], color='#E6F4EA')
        .set_properties(**{'text-align': 'center'}),
        use_container_width=True,
        hide_index=True,
        height=200
    )
    st.markdown(f"""
    <div style='background:#E6F4EA;border-left:4px solid {GREEN};
                padding:0.7rem 1rem;border-radius:0 8px 8px 0;
                margin-top:0.5rem;'>
        <b style='color:{GREEN};'>✓ Best Model: Logistic Regression</b><br>
        <span style='font-size:0.85rem;color:#333;'>
        Highest accuracy (93.52%), F1 (0.9379), ROC-AUC (0.9784)
        and negative recall (92.52%) — at fastest training time (1.2s).
        </span>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SECTION 7 — N-GRAM ANALYSIS
# ════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-title'>📝 N-gram Analysis</div>",
            unsafe_allow_html=True)

col_ng1, col_ng2 = st.columns([1, 1])

with col_ng1:
    fig_ng = go.Figure()
    colours_ng = [NAVY, ORANGE, TEAL]
    for i, row in ngrams.iterrows():
        fig_ng.add_trace(go.Bar(
            name=row['Gram Level'],
            x=[row['Gram Level']],
            y=[row['Top Phrase Frequency']],
            marker_color=colours_ng[i],
            text=[f"{row['Top Phrase Frequency']:,}"],
            textposition='outside',
        ))
    fig_ng.update_layout(
        title=dict(text='Top Phrase Frequency by Gram Level',
                   font=dict(size=13, color=NAVY)),
        showlegend=False,
        plot_bgcolor='white', paper_bgcolor='white',
        yaxis=dict(gridcolor='#F0F4F8', title='Top Phrase Frequency'),
        height=300,
        margin=dict(t=50, b=40, l=40, r=40),
    )
    st.plotly_chart(fig_ng, use_container_width=True, theme=None)

with col_ng2:
    st.markdown("**N-gram Comparison Results**")
    st.dataframe(
        ngrams.style
        .apply(lambda x: [
            f'background-color: #E6F4EA; font-weight: bold'
            if v == 'Yes — Optimal' else '' for v in x
        ], subset=['Selected'])
        .set_properties(**{'text-align': 'center'}),
        use_container_width=True,
        hide_index=True,
        height=160
    )
    st.markdown(f"""
    <div class='info-box'>
    <b>Finding:</b> Trigrams and four-grams achieve equivalent niche
    relevance (45%) but trigrams do so at 5.7× higher frequency
    (625 vs 109). Trigrams selected as optimal phrase length.
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='app-footer'>
    <b>A Data-Driven Framework for Product Selection in E-Commerce</b><br>
    Syed Aqeel · MSc Data Analytics · London Metropolitan University · ID: 24060071<br>
    Supervisor: Dr. Subeksha · Data: McAuley Amazon Reviews 2023 · GitHub: aqeel-gardezi/MSc-Project
</div>
""", unsafe_allow_html=True)
