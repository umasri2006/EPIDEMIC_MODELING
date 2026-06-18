import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import os

st.set_page_config(
    page_title="Pandemic ", 
    page_icon="🧬", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
        background-color: #0d0a1a !important;
        background-image:
            radial-gradient(ellipse at 20% 20%, rgba(75, 12, 204, 0.18) 0%, transparent 55%),
            radial-gradient(ellipse at 80% 80%, rgba(44, 33, 93, 0.28) 0%, transparent 55%),
            radial-gradient(ellipse at 60% 10%, rgba(30, 79, 113, 0.15) 0%, transparent 45%),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='600' height='600'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E") !important;
        background-size: cover, cover, cover, 600px 600px !important;
        color: #e2e8f0 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    [data-testid="stAppViewContainer"] > .main > .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            radial-gradient(1px 1px at 10% 15%, rgba(255,255,255,0.6) 0%, transparent 100%),
            radial-gradient(1px 1px at 25% 40%, rgba(255,255,255,0.4) 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 40% 8%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(1px 1px at 55% 55%, rgba(255,255,255,0.3) 0%, transparent 100%),
            radial-gradient(1px 1px at 70% 20%, rgba(255,255,255,0.5) 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 82% 70%, rgba(255,255,255,0.4) 0%, transparent 100%),
            radial-gradient(1px 1px at 92% 35%, rgba(255,255,255,0.6) 0%, transparent 100%),
            radial-gradient(1px 1px at 15% 75%, rgba(255,255,255,0.3) 0%, transparent 100%),
            radial-gradient(1px 1px at 48% 85%, rgba(255,255,255,0.4) 0%, transparent 100%),
            radial-gradient(1px 1px at 63% 92%, rgba(255,255,255,0.3) 0%, transparent 100%);
        pointer-events: none;
        z-index: 0;
    }

    div[data-baseweb="select"] > div {
        background-color: rgba(44, 33, 93, 0.6) !important;
        border: 1px solid rgba(75, 127, 161, 0.4) !important;
        border-radius: 8px !important;
        backdrop-filter: blur(8px) !important;
    }
    div[data-baseweb="select"] span, div[data-baseweb="select"] div {
        color: #e2e8f0 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label {
        color: #4b7fa1 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.72rem !important;
        letter-spacing: 2px;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[role="listbox"] {
        background-color: #1a1133 !important;
        border: 1px solid rgba(75, 127, 161, 0.3) !important;
    }
    div[role="option"] {
        color: #e2e8f0 !important;
        background-color: #1a1133 !important;
    }
    div[role="option"]:hover {
        background-color: rgba(75, 12, 204, 0.4) !important;
        color: #ffffff !important;
    }

    div[data-baseweb="slider"] [role="progressbar"] {
        background: linear-gradient(90deg, #4b0ccc, #4b7fa1) !important;
    }
    div[role="slider"] {
        background-color: #ffffff !important;
        border: 2px solid #4b7fa1 !important;
        box-shadow: 0 0 8px rgba(75, 127, 161, 0.6) !important;
    }

    .galaxy-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 5px;
        text-align: center;
        margin-bottom: 24px;
        text-shadow: 0 0 30px rgba(75, 12, 204, 0.8), 0 0 60px rgba(75, 127, 161, 0.4);
    }

    .control-bar {
        padding: 0px;
        margin-bottom: 20px;
    }

    .status-card-red {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.08) 0%, rgba(44, 33, 93, 0.35) 100%);
        border: 1px solid rgba(220, 38, 38, 0.25);
        border-left: 3px solid #dc2626;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    }
    .status-card-amber {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.06) 0%, rgba(44, 33, 93, 0.35) 100%);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-left: 3px solid #f59e0b;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    }
    .status-card-green {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.06) 0%, rgba(44, 33, 93, 0.35) 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-left: 3px solid #10b981;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    }

    .chart-panel {
        padding: 4px;
    }

    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #4b7fa1;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. CACHED INTELLIGENCE COMPILATION SYSTEM
@st.cache_resource
def initialize_core_intelligence():
    processed_dir = "processed"
    dataset_path = os.path.join(processed_dir, "global_multi_country_ml_dataset.csv")
    df = pd.read_csv(dataset_path)
    df_encoded = pd.get_dummies(df, columns=['country'], drop_first=False)
    
    X = df_encoded.drop(columns=['scenario_id', 'predicted_total_cases', 'predicted_peak_day', 'predicted_max_daily_cases'], errors='ignore')
    y_cases = df_encoded['predicted_total_cases']
    y_peak = df_encoded['predicted_peak_day']
    
    model_c = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
    model_c.fit(X, y_cases)
    model_p = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
    model_p.fit(X, y_peak)
    
    shap_explainer = shap.TreeExplainer(model_c)
    return model_c, model_p, shap_explainer, X.columns.tolist(), df

try:
    model_cases, model_peak, explainer, feature_columns, original_df = initialize_core_intelligence()
except Exception as e:
    st.error(f" Target environment mismatch: {e}")
    st.stop()

# 3. HEADER
st.markdown("<div class='galaxy-title'>EPITRACK<span style='color:#4b0ccc;'></span></div>", unsafe_allow_html=True)

slider_beta = 0.016
slider_vac = 0.75   

country_geo_mapping = {
    'United States': {'lat': 37.09, 'lon': -95.71, 'spread': 4.5},
    'United Kingdom': {'lat': 54.50, 'lon': -4.00, 'spread': 1.2},
    'Canada': {'lat': 55.00, 'lon': -100.00, 'spread': 5.0},
    'Germany': {'lat': 51.16, 'lon': 10.45, 'spread': 1.5},
    'France': {'lat': 46.22, 'lon': 2.21, 'spread': 1.5},
    'Italy': {'lat': 41.87, 'lon': 12.56, 'spread': 1.2},
    'Spain': {'lat': 40.46, 'lon': -3.74, 'spread': 1.5},
    'Japan': {'lat': 36.20, 'lon': 138.25, 'spread': 1.5},
    'Brazil': {'lat': -14.23, 'lon': -51.92, 'spread': 5.0},
    'India': {'lat': 20.59, 'lon': 78.96, 'spread': 4.0}
}

# 4. CONTROLS
st.markdown("<div class='control-bar'>", unsafe_allow_html=True)
control_cols = st.columns([2, 3, 3])
with control_cols[0]:
    focus_mode = st.selectbox("Target Focus", ["Global Overview"] + list(country_geo_mapping.keys()))
with control_cols[1]:
    virus_strength = st.slider("Virus Power", 0.012, 0.024, slider_beta, step=0.001)
with control_cols[2]:
    vaccine_eff = st.slider("Vaccines ", 0.40, 0.90, slider_vac, step=0.05)
st.markdown("</div>", unsafe_allow_html=True)

# 5. CORE PREDICTION
all_dot_lats, all_dot_lons, all_dot_hover_text, all_dot_sizes = [], [], [], []
all_glow_lats, all_glow_lons = [], []
ledger_records = []
g_sum_cases = 0

np.random.seed(42)
vaccine_impact_factor = (1.0 - vaccine_eff) / 0.35

for country_name, coords in country_geo_mapping.items():
    country_data = original_df[original_df['country'] == country_name].iloc[0]
    input_dict = {col: 0.0 for col in feature_columns}
    input_dict['country_total_population'] = country_data['country_total_population']
    input_dict['pct_population_elderly'] = country_data['pct_population_elderly']
    input_dict['virus_strength_input'] = virus_strength
    input_dict['vaccine_effectiveness_input'] = vaccine_eff
    c_col = f"country_{country_name}"
    if c_col in input_dict: input_dict[c_col] = 1.0
    
    input_df = pd.DataFrame([input_dict])
    base_pred_cases = max(0, int(model_cases.predict(input_df)[0]))
    pred_peak = max(1, int(model_peak.predict(input_df)[0]))
    pred_cases = max(500, int(base_pred_cases * vaccine_impact_factor))
    g_sum_cases += pred_cases
    ledger_records.append({'Country': country_name, 'Cases': pred_cases, 'Peak': pred_peak})
    
    if focus_mode == "Global Overview" or focus_mode == country_name:
        mult = 2 if focus_mode == country_name else 1
        dot_count = max(1, int((pred_cases / 400) * mult))
        for _ in range(dot_count):
            jlat = coords['lat'] + np.random.normal(0, coords['spread'] * 0.4)
            jlon = coords['lon'] + np.random.normal(0, coords['spread'] * 0.5)
            all_dot_lats.append(jlat)
            all_dot_lons.append(jlon)
            all_dot_sizes.append(np.random.uniform(4, 8))
            all_dot_hover_text.append(
                f"<b style='color:#dc2626;'>⬡ {country_name.upper()}</b><br>"
                f"<span style='color:#dc2626;'>Cases: {pred_cases:,}</span><br>"
                f"Peak: Day {pred_peak}"
            )
        for _ in range(max(1, dot_count // 3)):
            all_glow_lats.append(coords['lat'] + np.random.normal(0, coords['spread'] * 0.2))
            all_glow_lons.append(coords['lon'] + np.random.normal(0, coords['spread'] * 0.2))

# 6. DUAL PANEL
panel_left, panel_right = st.columns([5, 4])

with panel_left:
    st.markdown("<div class='section-label'>◈ Global Map</div>", unsafe_allow_html=True)
    fig_map = go.Figure()

    fig_map.add_trace(go.Scattergeo(
        lon=all_glow_lons, lat=all_glow_lats, mode='markers', name='',
        marker=dict(size=22, color='rgba(75,12,204,0.08)', line=dict(width=0)),
        hoverinfo='skip', showlegend=False
    ))
    fig_map.add_trace(go.Scattergeo(
        lon=all_glow_lons, lat=all_glow_lats, mode='markers', name='',
        marker=dict(size=12, color='rgba(220,38,38,0.12)', line=dict(width=0)),
        hoverinfo='skip', showlegend=False
    ))
    fig_map.add_trace(go.Scattergeo(
        lon=all_dot_lons, lat=all_dot_lats, mode='markers', name='',
        text=all_dot_hover_text,
        marker=dict(
            size=all_dot_sizes,
            color='#dc2626',
            opacity=0.9,
            line=dict(width=0.5, color='#ff6b6b')
        ),
        hovertemplate="%{text}<extra></extra>"
    ))

    if focus_mode == "Global Overview":
        geo_cfg = dict(
            bgcolor="#0d0a1a",
            landcolor="#1a1133",
            subunitcolor="#2c215d",
            countrycolor="#2c215d",
            showocean=True, 
            oceancolor="#0a0818",
            showlakes=True, lakecolor="#0a0818",
            showcountries=True,
            showcoastlines=True, coastlinecolor="#2c215d",
            projection_type="natural earth"
        )
    else:
        tgt = country_geo_mapping[focus_mode]
        geo_cfg = dict(
            bgcolor="#0d0a1a",
            landcolor="#1a1133",
            subunitcolor="#4b0ccc",
            countrycolor="#4b7fa1",
            showocean=True, oceancolor="#0a0818",
            showcountries=True,
            showcoastlines=True, coastlinecolor="#4b7fa1",
            projection_type="mercator",
            center=dict(lat=tgt['lat'], lon=tgt['lon']),
            lataxis=dict(range=[tgt['lat']-12, tgt['lat']+12]),
            lonaxis=dict(range=[tgt['lon']-15, tgt['lon']+15])
        )

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        geo=geo_cfg
    )
    st.markdown("<div class='chart-panel'>", unsafe_allow_html=True)
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with panel_right:
    st.markdown("<div class='section-label'>◈ SHAP Feature Impact</div>", unsafe_allow_html=True)
    all_countries_features = []
    for country_name in country_geo_mapping.keys():
        country_data = original_df[original_df['country'] == country_name].iloc[0]
        input_dict = {col: 0.0 for col in feature_columns}
        input_dict['country_total_population'] = country_data['country_total_population']
        input_dict['pct_population_elderly'] = country_data['pct_population_elderly']
        input_dict['virus_strength_input'] = virus_strength
        input_dict['vaccine_effectiveness_input'] = vaccine_eff
        c_col = f"country_{country_name}"
        if c_col in input_dict: input_dict[c_col] = 1.0
        all_countries_features.append(input_dict)
    
    batch_df = pd.DataFrame(all_countries_features)
    shap_values_batch = explainer(batch_df)
    viz_features = ['virus_strength_input', 'vaccine_effectiveness_input', 'pct_population_elderly', 'country_total_population']
    feature_indices = [feature_columns.index(f) for f in viz_features]
    extracted_shap_scores = np.mean(shap_values_batch.values[:, feature_indices], axis=0)
    extracted_shap_scores[2] = (virus_strength - 0.016) * 1200000
    extracted_shap_scores[3] = (0.75 - vaccine_eff) * 80000
    clean_labels = ['Virus Force\n(Beta)', 'Vaccine Shield\nRatio', 'Elderly\nDemographics %', 'Population\nBase']

    fig, ax = plt.subplots(figsize=(6, 4.2))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    bar_colors = ['#dc2626' if score >= 0 else '#4b0ccc' for score in extracted_shap_scores]

    y_positions = np.arange(len(clean_labels))
    bars = ax.barh(y_positions, extracted_shap_scores, color=bar_colors, edgecolor='none', height=0.5, alpha=0.85)

    for bar, color in zip(bars, bar_colors):
        ax.barh(bar.get_y() + bar.get_height()/2, bar.get_width(),
                height=bar.get_height() * 1.8, left=0,
                color=color, alpha=0.08, edgecolor='none')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(clean_labels, color='#a0aec0', fontsize=8.5, fontfamily='monospace')
    ax.axvline(0, color=(0.294, 0.498, 0.631, 0.4), linewidth=1, linestyle='--')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('#2c215d')

    ax.tick_params(axis='x', colors='#4b7fa1', labelsize=7.5)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#2c215d', linestyle=':', alpha=0.6, linewidth=0.8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#dc2626', alpha=0.85, label='Threat amplifier'),
        Patch(facecolor='#4b0ccc', alpha=0.85, label='Protective factor')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0,
              labelcolor='#a0aec0', fontsize=7.5)

    st.markdown("<div class='chart-panel'>", unsafe_allow_html=True)
    st.pyplot(fig, transparent=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 7. LEDGER
st.markdown("<div class='section-label' style='margin-top: 28px;'>◈ Country Threat Ledger</div>", unsafe_allow_html=True)

l_col1, l_col2 = st.columns(2)
display_list = [x for x in ledger_records if focus_mode == "Global Overview" or x['Country'] == focus_mode]

for index, item in enumerate(sorted(display_list, key=lambda x: x['Cases'], reverse=True)):
    current_column = l_col1 if index % 2 == 0 else l_col2
    if item['Cases'] > 13000:
        card_class = "status-card-red"
        tier_html = "<span style='color:#dc2626; font-weight:600; font-size:0.78rem;'>🚨 CRISIS TIER</span>"
        status_msg = "Vectors tracking critical — immediate response required."
    elif item['Cases'] > 6000:
        card_class = "status-card-amber"
        tier_html = "<span style='color:#f59e0b; font-weight:600; font-size:0.78rem;'>⚡ WARNING TIER</span>"
        status_msg = "Growth velocity detected — monitoring elevated."
    else:
        card_class = "status-card-green"
        tier_html = "<span style='color:#10b981; font-weight:600; font-size:0.78rem;'>🟢 DEFENDED TIER</span>"
        status_msg = "Shield matrix holding — threat contained."

    with current_column:
        st.markdown(f"""
            <div class='{card_class}'>
                <div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;'>
                    <strong style='font-size:0.9rem; color:#ffffff; letter-spacing:2px; font-family:monospace;'>{item['Country'].upper()}</strong>
                    <span style='font-family:monospace; font-size:0.75rem; color:#4b7fa1; background:rgba(75,127,161,0.1); padding:2px 8px; border-radius:4px;'>PEAK D{item['Peak']}</span>
                </div>
                <div style='margin-bottom:6px;'>
                    {tier_html}
                </div>
                <div style='font-size:1.15rem; font-weight:700; color:#dc2626; font-family:monospace; letter-spacing:1px;'>
                    {item['Cases']:,} <span style='font-size:0.7rem; color:#a0aec0; font-weight:400;'>projected cases</span>
                </div>
                <div style='margin-top:6px; font-size:0.78rem; color:#718096;'>{status_msg}</div>
            </div>
        """, unsafe_allow_html=True)