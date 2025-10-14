# 🔍 Social Curiosity Recommender System

Master's Thesis Project - A novel recommendation system that integrates social context with semantic divergence to drive curiosity-based recommendations.

## 🎯 Overview

This system demonstrates how **group context** (Low vs High divergence) influences **curiosity** in news recommendations, independent of content veracity (Real vs Fake).

### Key Features

- **Semantic Divergence (SD)**: Composite metric combining user-item, group-item, and user-group distances
- **Dual-Mode System**: Unlabeled (exploratory) and Labeled (evaluation) modes
- **2×2 Analysis**: Demonstrates independence of Real/Fake × Low/High dimensions
- **Alpha Sensitivity**: Identifies optimal curiosity-popularity balance
- **Ablation Study**: Validates full SD model outperforms single components

## 🚀 Quick Start

### Online Demo
Visit the live demo: [Your Streamlit Cloud URL]

### Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Usage

1. **Select Mode**: Choose "Unlabeled" or "Labeled" in the sidebar
2. **Upload Data**: 
   - Unlabeled: CSV with `title`, `text`
   - Labeled: CSV with `title`, `text`, `label` (0=Real, 1=Fake)
3. **Configure**: Adjust α (curiosity weight), top-k, group descriptions
4. **Explore**: View rankings, plots, sensitivity analysis, ablation study
5. **Export**: Download results with comprehensive metadata

## 📊 Sample Dataset

Included: `synthetic_fake_news_demo.csv` (20 articles: 10 Real + 10 Fake)

## 🔬 Technical Details

### Semantic Divergence Formula
```
SD(u,G,i) = 0.3·d(u,i) + 0.3·d(G,i) + 0.4·div(u,G)
```

Where:
- `d(u,i) = 1 - cos(Vu, Vi)` (user-item distance)
- `d(G,i) = 1 - cos(VG, Vi)` (group-item distance)
- `div(u,G) = 1 - cos(Vu, VG)` (user-group divergence)

### Final Recommendation Score
```
Final = α·Curiosity + (1-α)·Popularity
```

### Embeddings
- Model: SBERT `all-MiniLM-L6-v2`
- Normalization: L2-normalized to unit vectors
- Input: `title + ". " + text[:200]`

## 📈 Outputs

### Unlabeled Mode
- Ranked recommendations (Low/High divergence contexts)
- Low vs High comparison plot
- Alpha sensitivity analysis
- SD distribution
- SD → Curiosity correlation

### Labeled Mode (All above, plus)
- Type (Real/Fake) column in rankings
- Table 1: Toy example (same Real item, different contexts)
- Table 2: Full 2×2 analysis (all 4 cells)
- 2×2 Interaction plot
- FakeRatio monitoring
- Ablation study (F1-Score & NDCG-10)

## 🎓 Citation

If you use this system in your research, please cite:

```
[Your Name]. (2024). Social Curiosity Recommender System: 
Integrating Group Context with Semantic Divergence for 
Curiosity-Driven News Recommendations. Master's Thesis.
```

## 📄 License

[Your License Choice - e.g., MIT]

## 👤 Author

**[Your Name]**
- Master's Thesis Project
- Supervisor: Professor Bobby
- Institution: [Your University]

## 🙏 Acknowledgments

- Professor Bobby for guidance and feedback
- [Your University] for supporting this research

