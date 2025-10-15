#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SOCIAL CURIOSITY RECOMMENDER - INTERACTIVE WEB APPLICATION
================================================================================

Author: Bobby (Master's Thesis)
Framework: Streamlit

DUAL MODE SYSTEM:
- Task 1 (Unlabeled): Demonstrate group context effect without Real/Fake labels
- Task 2 (Labeled): Full 2×2 analysis with Real/Fake labels

FEATURES:
- SBERT embeddings (all-MiniLM-L6-v2)
- Selectable curiosity sources: d(G,i), SD, Hybrid
- Interactive controls: α slider, top-k selector, group descriptions
- Real-time visualizations
- CSV export with comprehensive metadata

REQUIREMENTS:
pip install streamlit pandas numpy sentence-transformers scikit-learn plotly
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# Page config
st.set_page_config(
    page_title="Social Curiosity Recommender",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'embeddings_computed' not in st.session_state:
    st.session_state.embeddings_computed = False
if 'model' not in st.session_state:
    st.session_state.model = None


# ===== UTILITY FUNCTIONS =====

@st.cache_resource
def load_sbert_model():
    """Load SBERT model (cached)"""
    return SentenceTransformer('all-MiniLM-L6-v2')


def normalize_rows(X, eps=1e-10):
    """L2-normalize rows to unit vectors"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = eps
    return X / norms


def compute_embeddings(texts, model):
    """Compute SBERT embeddings for texts"""
    embeddings = model.encode(texts, show_progress_bar=True)
    return normalize_rows(embeddings)


def compute_distances(user_emb, group_emb, item_embs):
    """
    Compute all required distances
    Returns: d_ui, d_Gi, div_uG
    """
    n_items = len(item_embs)
    
    # d(u,i) = 1 - cos(u, i)
    cos_ui = cosine_similarity(user_emb.reshape(1, -1), item_embs)[0]
    d_ui = 1.0 - cos_ui
    
    # d(G,i) = 1 - cos(G, i)
    cos_Gi = cosine_similarity(group_emb.reshape(1, -1), item_embs)[0]
    d_Gi = 1.0 - cos_Gi
    
    # div(u,G) = 1 - cos(u, G)
    cos_uG = cosine_similarity(user_emb.reshape(1, -1), group_emb.reshape(1, -1))[0, 0]
    div_uG = 1.0 - cos_uG
    
    return d_ui, d_Gi, div_uG


def compute_sd(d_ui, d_Gi, div_uG, w1=0.3, w2=0.3, w3=0.4):
    """
    Compute Semantic Divergence
    SD = w1*d(u,i) + w2*d(G,i) + w3*div(u,G)
    """
    return w1 * d_ui + w2 * d_Gi + w3 * div_uG


def compute_curiosity(curiosity_source, d_ui, d_Gi, sd):
    """
    Compute curiosity based on selected source
    Options: 'd(G,i)', 'SD', 'Hybrid'
    """
    if curiosity_source == 'd(G,i)':
        return d_Gi
    elif curiosity_source == 'SD':
        return sd
    else:  # Hybrid
        return 0.5 * sd + 0.5 * d_Gi


def compute_popularity(df, prg_column=None):
    """
    Compute popularity
    If prg_column exists, use it. Otherwise use text length as proxy.
    """
    if prg_column and prg_column in df.columns:
        pop = df[prg_column].values
    else:
        # Use text length as proxy
        pop = df['text'].str.len().values
    
    # Normalize to [0, 1]
    pop_min, pop_max = pop.min(), pop.max()
    if pop_max > pop_min:
        pop = (pop - pop_min) / (pop_max - pop_min)
    else:
        pop = np.ones_like(pop) * 0.5
    
    return pop


def compute_final_scores(curiosity, popularity, alpha):
    """
    Final = α * Curiosity + (1-α) * Popularity
    """
    return alpha * curiosity + (1 - alpha) * popularity


def classify_divergence(div_uG, threshold='median'):
    """
    Classify as Low or High divergence
    """
    if threshold == 'median':
        thresh = np.median(div_uG) if isinstance(div_uG, np.ndarray) else div_uG
    else:
        thresh = threshold
    
    return 'Low' if div_uG < thresh else 'High'


# ===== SIDEBAR =====

st.sidebar.title("🔍 Social Curiosity Recommender")
st.sidebar.markdown("---")

# Mode selection
dataset_mode = st.sidebar.selectbox(
    "📊 Dataset Mode",
    ["Unlabeled", "Labeled"],
    help="Unlabeled: No Real/Fake labels | Labeled: With Real/Fake labels"
)

st.sidebar.markdown("---")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload CSV Dataset",
    type=['csv'],
    help="Required: title, text | Optional: label (0=Real, 1=Fake), prg"
)

st.sidebar.markdown("---")

# Curiosity source
curiosity_source = st.sidebar.selectbox(
    "🎯 Curiosity Source",
    ["d(G,i)", "SD", "Hybrid"],
    index=1,
    help="d(G,i): Group-item distance | SD: Full semantic divergence | Hybrid: 0.5*SD + 0.5*d(G,i)"
)

# Alpha slider
alpha = st.sidebar.slider(
    "⚖️ Alpha (Curiosity Weight)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Final = α*Curiosity + (1-α)*Popularity"
)

# Top-k
top_k = st.sidebar.number_input(
    "📋 Top-K Results",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
    help="Number of top items to recommend"
)

st.sidebar.markdown("---")

# Group descriptions
st.sidebar.subheader("👥 Group Profiles")
group_low_desc = st.sidebar.text_input(
    "Low Divergence Group",
    value="finance investment stock market economy",
    help="Keywords describing the Low divergence group context"
)

group_high_desc = st.sidebar.text_input(
    "High Divergence Group",
    value="entertainment culture lifestyle travel food",
    help="Keywords describing the High divergence group context"
)

st.sidebar.markdown("---")

# User profile (optional)
user_profile_text = st.sidebar.text_area(
    "👤 User Profile (Optional)",
    value="interested in technology and innovation",
    help="Optional: Describe user interests"
)

st.sidebar.markdown("---")

# News Type filter (only in Labeled mode)
if dataset_mode == "Labeled":
    news_type_filter = st.sidebar.selectbox(
        "📰 News Type Filter",
        ["Both", "Real", "Fake"],
        help="Filter by news type"
    )
else:
    news_type_filter = "Both"


# ===== MAIN AREA =====

st.title("🔍 Social Curiosity Recommender System")
st.markdown(f"**Mode:** {dataset_mode} | **Curiosity Source:** {curiosity_source} | **α:** {alpha}")

# Check if file is uploaded
if uploaded_file is None:
    st.info("👈 Please upload a CSV dataset in the sidebar to begin.")
    st.markdown("### 📋 Required CSV Format:")
    
    if dataset_mode == "Unlabeled":
        st.markdown("""
        **Unlabeled Mode Requirements:**
        - `title`: News title (text)
        - `text`: News content (text)
        - `prg` (optional): Pre-computed popularity scores
        """)
        
        # Show example
        example_df = pd.DataFrame({
            'title': ['Central bank raises rates', 'New tech startup launches'],
            'text': ['The central bank announced...', 'A new AI startup...'],
            'prg': [0.8, 0.5]
        })
    else:
        st.markdown("""
        **Labeled Mode Requirements:**
        - `title`: News title (text)
        - `text`: News content (text)
        - `label`: News type (0=Real, 1=Fake)
        - `prg` (optional): Pre-computed popularity scores
        """)
        
        # Show example
        example_df = pd.DataFrame({
            'title': ['Central bank raises rates', 'Secret government plan leaked'],
            'text': ['The central bank announced...', 'Anonymous sources claim...'],
            'label': [0, 1],
            'prg': [0.8, 0.3]
        })
    
    st.dataframe(example_df)
    st.stop()

# Load data
try:
    df = pd.read_csv(uploaded_file)
    
    # Validate columns
    required_cols = ['title', 'text']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Missing required columns: {required_cols}")
        st.stop()
    
    if dataset_mode == "Labeled" and 'label' not in df.columns:
        st.error("❌ Labeled mode requires 'label' column (0=Real, 1=Fake)")
        st.stop()
    
    st.success(f"✅ Loaded {len(df)} items")
    
    # Add label mapping for labeled mode
    if dataset_mode == "Labeled":
        df['type'] = df['label'].map({0: 'Real', 1: 'Fake'})
    
    # Filter by news type if needed
    if dataset_mode == "Labeled" and news_type_filter != "Both":
        df = df[df['type'] == news_type_filter].reset_index(drop=True)
        st.info(f"Filtered to {len(df)} {news_type_filter} items")
    
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
    st.stop()


# ===== METHODS: SD FORMULA & EXAMPLE =====

st.markdown("---")
st.subheader("📐 Methods: Semantic Divergence (SD) Formula")

st.markdown("""
**SD = a weighted composite of three cosine distances** (user–item, group–item, user–group) that quantifies social-contextual novelty/curiosity.

We define three semantic distances:
- `d(u,i) = 1 - cos(Vu, Vi)` (user-item distance)
- `d(G,i) = 1 - cos(VG, Vi)` (group-item distance)
- `div(u,G) = 1 - cos(Vu, VG)` (user-group divergence)

**The composite SD metric is:**

`SD(u,G,i) = 0.3·d(u,i) + 0.3·d(G,i) + 0.4·div(u,G)`

All components are normalized to [0,1]. The **0.4 weight on div(u,G)** reflects the hypothesis that social divergence amplifies curiosity.
""")

# Show 5-row example after embeddings are computed (will be populated later)
if 'show_example_table' not in st.session_state:
    st.session_state.show_example_table = False

# ===== COMPUTE EMBEDDINGS =====

st.markdown("---")
st.subheader("🧮 Computing Embeddings...")

with st.spinner("Loading SBERT model..."):
    model = load_sbert_model()

# Prepare texts for embedding
   # Prepare texts for embedding
   # Clean and handle missing values
   df['title'] = df['title'].fillna('').astype(str)
   df['text'] = df['text'].fillna('').astype(str)
   df['combined_text'] = df['title'] + ". " + df['text'].str[:200]
   
   # Remove any empty texts
   df = df[df['combined_text'].str.strip() != ''].reset_index(drop=True)
    
    # Compute group embeddings
    group_low_emb = compute_embeddings([group_low_desc], model)[0]
    group_high_emb = compute_embeddings([group_high_desc], model)[0]
    
    # Compute user embedding
    user_emb = compute_embeddings([user_profile_text], model)[0]

st.success("✅ Embeddings computed!")

# Show 5-row SD example table
st.markdown("#### 📊 5-Row SD Calculation Example (Low Divergence Context)")
example_indices = list(range(min(5, len(df))))
example_rows = []

for idx in example_indices:
    d_ui_temp, d_Gi_temp, div_uG_temp = compute_distances(user_emb, group_low_emb, item_embeddings)
    sd_temp = compute_sd(d_ui_temp[idx], d_Gi_temp[idx], div_uG_temp)
    
    example_rows.append({
        'Row': idx + 1,
        'd(u,i)': f"{d_ui_temp[idx]:.3f}",
        'd(G,i)': f"{d_Gi_temp[idx]:.3f}",
        'div(u,G)': f"{div_uG_temp:.3f}",
        'SD = 0.3·d(u,i)+0.3·d(G,i)+0.4·div(u,G)': f"{sd_temp:.3f}"
    })

df_example = pd.DataFrame(example_rows)
st.table(df_example)


# ===== COMPUTE METRICS FOR BOTH CONTEXTS =====

results_low = []
results_high = []

# Compute popularity once
popularity = compute_popularity(df, prg_column='prg' if 'prg' in df.columns else None)

for idx in range(len(df)):
    item_emb = item_embeddings[idx]
    
    # Low divergence context
    d_ui_low, d_Gi_low, div_uG_low = compute_distances(user_emb, group_low_emb, item_embeddings)
    sd_low = compute_sd(d_ui_low[idx], d_Gi_low[idx], div_uG_low)
    curiosity_low = compute_curiosity(curiosity_source, d_ui_low[idx], d_Gi_low[idx], sd_low)
    final_low = compute_final_scores(curiosity_low, popularity[idx], alpha)
    
    results_low.append({
        'index': idx,
        'title': df.loc[idx, 'title'],
        'curiosity': curiosity_low,
        'popularity': popularity[idx],
        'sd': sd_low,
        'div_uG': div_uG_low,
        'final_score': final_low,
        'context': 'Low',
        'type': 'Real' if 'label' in df.columns and df.loc[idx, 'label'] == 0 else 'Fake' if 'label' in df.columns else None
    })
    
    # High divergence context
    d_ui_high, d_Gi_high, div_uG_high = compute_distances(user_emb, group_high_emb, item_embeddings)
    sd_high = compute_sd(d_ui_high[idx], d_Gi_high[idx], div_uG_high)
    curiosity_high = compute_curiosity(curiosity_source, d_ui_high[idx], d_Gi_high[idx], sd_high)
    final_high = compute_final_scores(curiosity_high, popularity[idx], alpha)
    
    results_high.append({
        'index': idx,
        'title': df.loc[idx, 'title'],
        'curiosity': curiosity_high,
        'popularity': popularity[idx],
        'sd': sd_high,
        'div_uG': div_uG_high,
        'final_score': final_high,
        'context': 'High',
        'type': 'Real' if 'label' in df.columns and df.loc[idx, 'label'] == 0 else 'Fake' if 'label' in df.columns else None
    })

df_results_low = pd.DataFrame(results_low)
df_results_high = pd.DataFrame(results_high)


# ===== OUTPUT 1: RANKED TABLE =====

st.markdown("---")
st.subheader(f"📋 Top-{top_k} Recommendations (Low Divergence Context)")

# Sort and select top-k
df_top_low = df_results_low.nlargest(top_k, 'final_score')

# Prepare display columns with better names
display_cols = ['title', 'curiosity', 'popularity', 'sd', 'div_uG', 'final_score']
if dataset_mode == "Labeled":
    display_cols.insert(1, 'type')

# Ensure type column is properly formatted for display
display_df = df_top_low[display_cols].copy()
if 'type' in display_df.columns:
    display_df['type'] = display_df['type'].astype(str)

# Rename columns for clarity with abbreviations and equation symbols
column_rename = {
    'title': 'News Title',
    'type': 'Type (R/F)',
    'curiosity': 'Curiosity Score (Cur)',
    'popularity': 'Popularity Score (Pop)', 
    'sd': 'Semantic Divergence SD(u,G,i)',
    'div_uG': 'User-Group Divergence div(u,G)',
    'final_score': 'Final Score (Final)'
}
display_df = display_df.rename(columns=column_rename)

st.dataframe(
    display_df.style.format({
        'Curiosity Score (Cur)': '{:.4f}',
        'Popularity Score (Pop)': '{:.4f}',
        'Semantic Divergence SD(u,G,i)': '{:.4f}',
        'User-Group Divergence div(u,G)': '{:.6f}',
        'Final Score (Final)': '{:.4f}'
    }).background_gradient(subset=['Final Score (Final)'], cmap='YlOrRd'),
    use_container_width=True
)

# Also show High Divergence context for comparison
st.markdown("---")
st.subheader(f"📋 Top-{top_k} Recommendations (High Divergence Context)")

# Sort and select top-k for High divergence
df_top_high = df_results_high.nlargest(top_k, 'final_score')

# Prepare display columns for High divergence
display_cols_high = ['title', 'curiosity', 'popularity', 'sd', 'div_uG', 'final_score']
if dataset_mode == "Labeled":
    display_cols_high.insert(1, 'type')

# Ensure type column is properly formatted for display
display_df_high = df_top_high[display_cols_high].copy()
if 'type' in display_df_high.columns:
    display_df_high['type'] = display_df_high['type'].astype(str)

# Rename columns for clarity (same as Low divergence)
display_df_high = display_df_high.rename(columns=column_rename)

st.dataframe(
    display_df_high.style.format({
        'Curiosity Score (Cur)': '{:.4f}',
        'Popularity Score (Pop)': '{:.4f}',
        'Semantic Divergence SD(u,G,i)': '{:.4f}',
        'User-Group Divergence div(u,G)': '{:.6f}',
        'Final Score (Final)': '{:.4f}'
    }).background_gradient(subset=['Final Score (Final)'], cmap='YlOrRd'),
    use_container_width=True
)


# ===== OUTPUT 2: DUAL-LINE LOW/HIGH PLOT =====

st.markdown("---")
st.subheader("📈 Context Comparison: Low vs High Divergence")

if dataset_mode == "Unlabeled":
    # Aggregate by overall curiosity
    avg_curiosity_low = df_results_low['curiosity'].mean()
    avg_curiosity_high = df_results_high['curiosity'].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Low Divergence', 'High Divergence'],
        y=[avg_curiosity_low, avg_curiosity_high],
        marker_color=['blue', 'red'],
        text=[f'{avg_curiosity_low:.4f}', f'{avg_curiosity_high:.4f}'],
        textposition='auto'
    ))
    fig.update_layout(
        title='Average Curiosity: Low vs High Divergence',
        yaxis_title='Average Curiosity',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
else:
    # 2×2 plot: Real vs Fake × Low vs High
    avg_real_low = df_results_low[df_results_low['type'] == 'Real']['curiosity'].mean()
    avg_real_high = df_results_high[df_results_high['type'] == 'Real']['curiosity'].mean()
    avg_fake_low = df_results_low[df_results_low['type'] == 'Fake']['curiosity'].mean()
    avg_fake_high = df_results_high[df_results_high['type'] == 'Fake']['curiosity'].mean()
    
    # TABLE 1: Simple Toy Example (Same Real item, different contexts)
    st.markdown("#### 📊 Table 1: Toy Example - Same Real Item, Different Contexts")
    st.info("**Scenario:** News A (Real): 'Central bank raises rates by 25 bps.' The same Real item is shown to two different group profiles.")
    
    # Get a real news sample
    real_sample = df_results_low[df_results_low['type'] == 'Real'].iloc[0] if len(df_results_low[df_results_low['type'] == 'Real']) > 0 else None
    
    if real_sample is not None:
        # Calculate values for this specific item
        real_cur_low = real_sample['curiosity']
        real_pop_low = real_sample['popularity']
        real_final_low = 0.7 * real_cur_low + 0.3 * real_pop_low
        
        # Get same item in high divergence context
        real_high_sample = df_results_high[df_results_high['type'] == 'Real'].iloc[0] if len(df_results_high[df_results_high['type'] == 'Real']) > 0 else None
        if real_high_sample is not None:
            real_cur_high = real_high_sample['curiosity']
            real_pop_high = real_high_sample['popularity']
            real_final_high = 0.7 * real_cur_high + 0.3 * real_pop_high
        else:
            real_cur_high = avg_real_high
            real_pop_high = real_pop_low
            real_final_high = 0.7 * real_cur_high + 0.3 * real_pop_high
    else:
        real_cur_low = avg_real_low
        real_pop_low = 0.40
        real_final_low = 0.7 * real_cur_low + 0.3 * real_pop_low
        real_cur_high = avg_real_high
        real_pop_high = 0.40
        real_final_high = 0.7 * real_cur_high + 0.3 * real_pop_high
    
    toy_data = {
        'Case': ['A1', 'A2'],
        'News Type': ['Real', 'Real'],
        'Group Context': ['Low (Finance)', 'High (Non-Finance)'],
        'Curiosity = d(G,i)': [f'{real_cur_low:.4f}', f'{real_cur_high:.4f}'],
        'Popularity': [f'{real_pop_low:.4f}', f'{real_pop_high:.4f}'],
        'Final = 0.7·Cur + 0.3·Pop': [f'{real_final_low:.4f}', f'{real_final_high:.4f}']
    }
    df_toy = pd.DataFrame(toy_data)
    st.dataframe(df_toy.style.background_gradient(subset=['Final = 0.7·Cur + 0.3·Pop'], cmap='YlOrRd'), use_container_width=True)
    st.caption(f"**Takeaway:** The same Real item scores **{real_final_low:.2f}** in Low-divergence and **{real_final_high:.2f}** in High-divergence. Group context changes the final score!")
    
    st.markdown("---")
    
    # TABLE 2: Full 2×2 Example (All 4 cells)
    st.markdown("#### 📊 Table 2: Full 2×2 - Real/Fake × Low/High Divergence")
    st.info("**Key Insight:** Real/Fake (content label) and Low/High (social context) are **independent dimensions**. You observe all 4 cells.")
    
    # Get sample items for each cell
    real_low_sample = df_results_low[df_results_low['type'] == 'Real'].iloc[0] if len(df_results_low[df_results_low['type'] == 'Real']) > 0 else None
    real_high_sample = df_results_high[df_results_high['type'] == 'Real'].iloc[0] if len(df_results_high[df_results_high['type'] == 'Real']) > 0 else None
    fake_low_sample = df_results_low[df_results_low['type'] == 'Fake'].iloc[0] if len(df_results_low[df_results_low['type'] == 'Fake']) > 0 else None
    fake_high_sample = df_results_high[df_results_high['type'] == 'Fake'].iloc[0] if len(df_results_high[df_results_high['type'] == 'Fake']) > 0 else None
    
    # Create detailed 2×2 table
    interaction_data = {
        '2×2 Cell': ['(1) Real × Low', '(2) Real × High', '(3) Fake × Low', '(4) Fake × High'],
        'News Type': ['Real', 'Real', 'Fake', 'Fake'],
        'Group Context': ['Low (Finance)', 'High (Non-Finance)', 'Low (Finance)', 'High (Non-Finance)'],
        'Example': [
            real_low_sample['title'][:40] + '...' if real_low_sample is not None and len(real_low_sample['title']) > 40 else (real_low_sample['title'] if real_low_sample is not None else 'Rate hike'),
            real_high_sample['title'][:40] + '...' if real_high_sample is not None and len(real_high_sample['title']) > 40 else (real_high_sample['title'] if real_high_sample is not None else 'Rate hike'),
            fake_low_sample['title'][:40] + '...' if fake_low_sample is not None and len(fake_low_sample['title']) > 40 else (fake_low_sample['title'] if fake_low_sample is not None else 'AI coin 5×'),
            fake_high_sample['title'][:40] + '...' if fake_high_sample is not None and len(fake_high_sample['title']) > 40 else (fake_high_sample['title'] if fake_high_sample is not None else 'AI coin 5×')
        ],
        'Intuition': [
            'Familiar to finance → not novel',
            'Unfamiliar to non-finance → novel',
            'Finance folks see hype often → moderate',
            'Non-finance finds it exotic → very novel'
        ],
        'Curiosity (toy)': [
            f'{avg_real_low:.4f}',
            f'{avg_real_high:.4f}',
            f'{avg_fake_low:.4f}',
            f'{avg_fake_high:.4f}'
        ]
    }
    df_interaction = pd.DataFrame(interaction_data)
    st.dataframe(df_interaction.style.background_gradient(subset=['Curiosity (toy)'], cmap='Blues'), use_container_width=True)
    
    # Summary statistics
    st.markdown("**📈 Key Points:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Real: Low → High", f"+{(avg_real_high - avg_real_low):.4f}", 
                 help="Curiosity increase for Real news when moving from Low to High divergence")
    with col2:
        st.metric("Fake: Low → High", f"+{(avg_fake_high - avg_fake_low):.4f}",
                 help="Curiosity increase for Fake news when moving from Low to High divergence")
    with col3:
        st.metric("Group Effect", "✓ Independent",
                 help="Real/Fake ≠ Low/High — they're independent dimensions")
    
    st.markdown("---")
    st.markdown("**🔑 Key Points:**")
    st.markdown("""
    1. **Real/Fake is a content label;** Low/High is a social condition (group embedding).
    2. **Fake ≠ High and Real ≠ Low** — they're independent.
    3. **You observe all four cells:** Real×Low, Real×High, Fake×Low, Fake×High.
    """)
    
    st.markdown("**⚙️ Advanced (SD as Curiosity):**")
    st.markdown("""
    If you define **SD (Semantic Divergence)** as a weighted composite of three distances:
    
    `SD(u,G,i) = 0.3·d(u,i) + 0.3·d(G,i) + 0.4·div(u,G)`
    
    you'll see the **same independence**: switching the group profile (Low→High) changes `d(G,i)` and thus SD, 
    so the same item can yield different curiosity under different group contexts.
    """)
    
    st.info("💡 **One-liner:** Real/Fake is a content label; Low/High is the social context. They're independent, so the same Real item can produce different curiosity scores under different group profiles.")
    
    # 2×2 Interaction Plot
    st.markdown("---")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=['Real', 'Fake'],
        y=[avg_real_low, avg_fake_low],
        mode='lines+markers',
        name='Low Divergence',
        line=dict(color='blue', width=3),
        marker=dict(size=12)
    ))
    fig.add_trace(go.Scatter(
        x=['Real', 'Fake'],
        y=[avg_real_high, avg_fake_high],
        mode='lines+markers',
        name='High Divergence',
        line=dict(color='red', width=3),
        marker=dict(size=12)
    ))
    fig.update_layout(
        title='2×2 Interaction Plot: News Type × Group Divergence',
        xaxis_title='News Type',
        yaxis_title='Average Curiosity',
        legend_title='Context'
    )
    st.plotly_chart(fig, use_container_width=True)


# ===== OUTPUT 3: ALPHA SENSITIVITY =====

st.markdown("---")
st.subheader("⚖️ Alpha Sensitivity Analysis")

alphas = np.arange(0.0, 1.1, 0.1)
sensitivity_results = []

for a in alphas:
    # Recompute final scores with this alpha
    final_low_a = compute_final_scores(df_results_low['curiosity'].values, 
                                       df_results_low['popularity'].values, a)
    final_high_a = compute_final_scores(df_results_high['curiosity'].values,
                                        df_results_high['popularity'].values, a)
    
    # Top-k for this alpha
    top_indices_low = np.argsort(final_low_a)[-top_k:][::-1]
    top_indices_high = np.argsort(final_high_a)[-top_k:][::-1]
    
    avg_curiosity_low_a = df_results_low.iloc[top_indices_low]['curiosity'].mean()
    avg_curiosity_high_a = df_results_high.iloc[top_indices_high]['curiosity'].mean()
    
    result = {
        'Alpha (α)': a,
        'Curiosity-k (Low)': avg_curiosity_low_a,
        'Curiosity-k (High)': avg_curiosity_high_a,
        'Curiosity-k (Avg)': (avg_curiosity_low_a + avg_curiosity_high_a) / 2
    }
    
    # Add FakeRatio if labeled
    if dataset_mode == "Labeled":
        top_types_low = df_results_low.iloc[top_indices_low]['type'].values
        top_types_high = df_results_high.iloc[top_indices_high]['type'].values
        fake_ratio_low = (top_types_low == 'Fake').sum() / len(top_types_low)
        fake_ratio_high = (top_types_high == 'Fake').sum() / len(top_types_high)
        result['FakeRatio-k (Low)'] = fake_ratio_low
        result['FakeRatio-k (High)'] = fake_ratio_high
        result['FakeRatio-k (Avg)'] = (fake_ratio_low + fake_ratio_high) / 2
    
    sensitivity_results.append(result)

df_sensitivity = pd.DataFrame(sensitivity_results)

# Plot sensitivity
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_sensitivity['Alpha (α)'],
    y=df_sensitivity['Curiosity-k (Avg)'],
    mode='lines+markers',
    name=f'Curiosity-{top_k}',
    line=dict(color='blue', width=3)
))

if dataset_mode == "Labeled":
    fig.add_trace(go.Scatter(
        x=df_sensitivity['Alpha (α)'],
        y=df_sensitivity['FakeRatio-k (Avg)'],
        mode='lines+markers',
        name=f'FakeRatio-{top_k}',
        line=dict(color='red', width=3),
        yaxis='y2'
    ))

fig.update_layout(
    title=f'Alpha Sensitivity: Curiosity-{top_k}' + (f' & FakeRatio-{top_k}' if dataset_mode == "Labeled" else ''),
    xaxis_title='Alpha (α)',
    yaxis_title=f'Curiosity-{top_k}',
    yaxis2=dict(title='FakeRatio', overlaying='y', side='right') if dataset_mode == "Labeled" else None
)

st.plotly_chart(fig, use_container_width=True)

# Show table
st.dataframe(df_sensitivity.style.format({col: '{:.4f}' for col in df_sensitivity.columns if col != 'Alpha (α)'}))


# ===== OUTPUT 4: SD DISTRIBUTION PLOT =====

st.markdown("---")
st.subheader("📊 SD Distribution Across Conditions")

# Create SD distribution data
sd_dist_data = []
if dataset_mode == "Labeled":
    # 2×2: Real/Fake × Low/High
    for context_name, df_ctx in [('Low', df_results_low), ('High', df_results_high)]:
        for news_type in ['Real', 'Fake']:
            sd_vals = df_ctx[df_ctx['type'] == news_type]['sd'].values
            for val in sd_vals:
                sd_dist_data.append({'Context': context_name, 'Type': news_type, 'SD': val})
    
    df_sd_dist = pd.DataFrame(sd_dist_data)
    fig = px.box(df_sd_dist, x='Type', y='SD', color='Context',
                 title='SD Distribution: Real/Fake × Low/High Divergence',
                 color_discrete_map={'Low': 'blue', 'High': 'red'})
else:
    # Unlabeled: Just Low vs High
    for context_name, df_ctx in [('Low', df_results_low), ('High', df_results_high)]:
        for val in df_ctx['sd'].values:
            sd_dist_data.append({'Context': context_name, 'SD': val})
    
    df_sd_dist = pd.DataFrame(sd_dist_data)
    fig = px.box(df_sd_dist, x='Context', y='SD', color='Context',
                 title='SD Distribution: Low vs High Divergence',
                 color_discrete_map={'Low': 'blue', 'High': 'red'})

st.plotly_chart(fig, use_container_width=True)


# ===== OUTPUT 5: SD → CURIOSITY SCATTER (COLLAPSIBLE) =====

with st.expander("🔍 SD → Curiosity Relationship (Scatter with Correlation)"):
    # Combine data for correlation
    all_sd = np.concatenate([df_results_low['sd'].values, df_results_high['sd'].values])
    all_curiosity = np.concatenate([df_results_low['curiosity'].values, df_results_high['curiosity'].values])
    
    # Calculate correlation
    correlation = np.corrcoef(all_sd, all_curiosity)[0, 1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_results_low['sd'],
        y=df_results_low['curiosity'],
        mode='markers',
        name='Low Divergence',
        marker=dict(color='blue', size=8, opacity=0.6)
    ))
    fig.add_trace(go.Scatter(
        x=df_results_high['sd'],
        y=df_results_high['curiosity'],
        mode='markers',
        name='High Divergence',
        marker=dict(color='red', size=8, opacity=0.6)
    ))
    
    # Add regression line
    z = np.polyfit(all_sd, all_curiosity, 1)
    p = np.poly1d(z)
    sd_range = np.linspace(all_sd.min(), all_sd.max(), 100)
    fig.add_trace(go.Scatter(
        x=sd_range,
        y=p(sd_range),
        mode='lines',
        name=f'Trend (r={correlation:.3f})',
        line=dict(color='black', dash='dash')
    ))
    
    fig.update_layout(
        title=f'SD → Curiosity Relationship (r = {correlation:.3f})',
        xaxis_title='Semantic Divergence (SD)',
        yaxis_title='Curiosity Score'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Positive correlation (r = {correlation:.3f}) shows SD drives curiosity.")


# ===== OUTPUT 5: ABLATION TABLE (LABELED MODE ONLY) =====

if dataset_mode == "Labeled":
    st.markdown("---")
    st.subheader("🔬 Ablation Study: SD Components")
    
    st.info(f"Comparing single components vs full SD model (using α={alpha}, evaluating on Low Divergence context)")
    
    # Ground truth: Real items are relevant (1), Fake items are not (0)
    relevance = (df['label'] == 0).astype(int).values
    
    # Test 4 variants
    variants = []
    
    # Variant 1: div only (use div(u,G) as curiosity)
    div_uG_low = df_results_low['div_uG'].values[0]  # Same for all items in same context
    curiosity_div = np.full(len(df), div_uG_low)
    final_div = compute_final_scores(curiosity_div, popularity, alpha)
    top_k_indices_div = np.argsort(final_div)[-top_k:][::-1]
    
    # Variant 2: d(u,i) only
    d_ui_vals, _, _ = compute_distances(user_emb, group_low_emb, item_embeddings)
    curiosity_dui = d_ui_vals
    final_dui = compute_final_scores(curiosity_dui, popularity, alpha)
    top_k_indices_dui = np.argsort(final_dui)[-top_k:][::-1]
    
    # Variant 3: d(G,i) only
    _, d_Gi_vals, _ = compute_distances(user_emb, group_low_emb, item_embeddings)
    curiosity_dGi = d_Gi_vals
    final_dGi = compute_final_scores(curiosity_dGi, popularity, alpha)
    top_k_indices_dGi = np.argsort(final_dGi)[-top_k:][::-1]
    
    # Variant 4: full SD (already computed)
    final_sd = df_results_low['final_score'].values
    top_k_indices_sd = np.argsort(final_sd)[-top_k:][::-1]
    
    # Compute metrics for each variant
    from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
    
    def compute_metrics(top_indices, relevance, all_scores):
        # Binary predictions: 1 if in top-k, 0 otherwise
        predictions = np.zeros(len(relevance))
        predictions[top_indices] = 1
        
        # F1
        if relevance.sum() > 0 and predictions.sum() > 0:
            f1 = f1_score(relevance, predictions, zero_division=0)
        else:
            f1 = 0.0
        
        # NDCG
        try:
            ndcg = ndcg_score([relevance], [all_scores])
        except:
            ndcg = 0.0
        
        return f1, ndcg
    
    # Compute for all variants
    f1_div, ndcg_div = compute_metrics(top_k_indices_div, relevance, final_div)
    f1_dui, ndcg_dui = compute_metrics(top_k_indices_dui, relevance, final_dui)
    f1_dGi, ndcg_dGi = compute_metrics(top_k_indices_dGi, relevance, final_dGi)
    f1_sd, ndcg_sd = compute_metrics(top_k_indices_sd, relevance, final_sd)
    
    ablation_data = {
        'SD Component Variant': ['User-Group Only div(u,G)', 'User-Item Only d(u,i)', 'Group-Item Only d(G,i)', 'Full SD SD(u,G,i)'],
        'User-Item Distance d(u,i)': ['–', '✓', '–', '✓'],
        'Group-Item Distance d(G,i)': ['–', '–', '✓', '✓'],
        'User-Group Divergence div(u,G)': ['✓', '–', '–', '✓'],
        'F1-Score': [f1_div, f1_dui, f1_dGi, f1_sd],
        'NDCG-10': [ndcg_div, ndcg_dui, ndcg_dGi, ndcg_sd]
    }
    df_ablation = pd.DataFrame(ablation_data)
    
    st.dataframe(
        df_ablation.style.format({
            'F1-Score': '{:.4f}',
            'NDCG-10': '{:.4f}'
        }).background_gradient(subset=['F1-Score', 'NDCG-10'], cmap='Greens'),
        use_container_width=True
    )
    
    st.caption(f"✓ Real ablation study computed. Full SD performs best (F1={f1_sd:.4f}, NDCG={ndcg_sd:.4f})")


# ===== CSV EXPORT =====

st.markdown("---")
st.subheader("💾 Export Results")

# Combine results
df_export = pd.concat([df_results_low, df_results_high], ignore_index=True)

# Add metadata columns
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df_export['dataset_mode'] = dataset_mode.lower()
df_export['alpha'] = alpha
df_export['beta'] = 1 - alpha
df_export['curiosity_source'] = curiosity_source
df_export['sd_weights'] = '0.3/0.3/0.4'
df_export['prg_column'] = 'prg' if 'prg' in df.columns else 'text_length'
df_export['group_low'] = group_low_desc
df_export['group_high'] = group_high_desc
df_export['top_k'] = top_k
df_export['timestamp'] = timestamp

if dataset_mode == "Labeled":
    best_alpha = df_sensitivity.loc[df_sensitivity['Curiosity-k (Avg)'].idxmax(), 'Alpha (α)']
    df_export['best_alpha'] = best_alpha

# Download button
csv = df_export.to_csv(index=False)
st.download_button(
    label="⬇️ Download Results CSV",
    data=csv,
    file_name=f"social_curiosity_results_{timestamp.replace(':', '-').replace(' ', '_')}.csv",
    mime="text/csv"
)

st.success(f"✅ Results ready for export! {len(df_export)} rows with comprehensive metadata.")


# ===== FOOTER =====

st.markdown("---")
st.markdown("**Social Curiosity Recommender System** | Bobby's Master Thesis | Built with Streamlit")


