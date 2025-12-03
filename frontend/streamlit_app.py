"""
Streamlit frontend for news article semantic similarity search
"""
import streamlit as st
import requests
import json
from typing import List, Dict
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="News Article Semantic Similarity",
    page_icon="📰",
    layout="wide"
)

# API endpoint
API_URL = "http://127.0.0.1:8000"

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .article-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .similarity-score {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def search_articles(article: str, top_k: int = 5) -> Dict:
    """Search for similar articles."""
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"article": article, "top_k": top_k},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return None

def main():
    """Main Streamlit app."""
    st.markdown('<div class="main-header">📰 News Article Semantic Similarity Search</div>', 
                unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API server is not running. Please start the backend server:")
        st.code("uvicorn backend.api:app --reload", language="bash")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of results", 1, 20, 5)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application uses contrastive learning to find 
        semantically similar news articles.
        
        **Features:**
        - Semantic similarity search
        - Top-K article retrieval
        - Similarity scores
        """)
    
    # Main content
    tab1, tab2 = st.tabs(["🔍 Search", "📊 About"])
    
    with tab1:
        st.markdown("### Enter a News Article")
        
        # Input methods
        input_method = st.radio(
            "Input method:",
            ["Text Input", "URL (coming soon)"],
            horizontal=True
        )
        
        if input_method == "Text Input":
            article_text = st.text_area(
                "Paste your news article here:",
                height=200,
                placeholder="Enter a news article to find similar articles..."
            )
        else:
            article_url = st.text_input("Enter article URL:")
            article_text = None
            if article_url:
                st.info("URL fetching not implemented yet. Please use text input.")
        
        if st.button("🔍 Search Similar Articles", type="primary"):
            if not article_text or len(article_text.strip()) < 50:
                st.warning("Please enter a longer article (at least 50 characters).")
            else:
                with st.spinner("Searching for similar articles..."):
                    results = search_articles(article_text, top_k=top_k)
                
                if results:
                    st.success(f"Found {len(results['similar_articles'])} similar articles!")
                    
                    # Display query article
                    st.markdown("### Your Article")
                    st.markdown(f'<div class="article-box">{article_text[:500]}...</div>', 
                              unsafe_allow_html=True)
                    
                    # Display results
                    st.markdown("### Similar Articles")
                    
                    for i, (article, score, idx) in enumerate(zip(
                        results['similar_articles'],
                        results['scores'],
                        results['indices']
                    ), 1):
                        with st.expander(f"Article {i} (Similarity: {score:.4f})", expanded=(i <= 2)):
                            st.markdown(f'<div class="article-box">{article}</div>', 
                                      unsafe_allow_html=True)
                            st.markdown(f'<p class="similarity-score">Index: {idx} | Score: {score:.4f}</p>', 
                                      unsafe_allow_html=True)
                    
                    # Visualization
                    if len(results['scores']) > 1:
                        st.markdown("### Similarity Scores")
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[f"Article {i+1}" for i in range(len(results['scores']))],
                                y=results['scores'],
                                marker_color='lightblue',
                                text=[f"{s:.4f}" for s in results['scores']],
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            title="Similarity Scores",
                            xaxis_title="Article",
                            yaxis_title="Similarity Score",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("""
        ## About This Project
        
        This is a **News Article Semantic Similarity & Topic Retrieval** system 
        built using **Contrastive Learning**.
        
        ### How It Works
        
        1. **Encoding**: News articles are encoded into fixed-length embeddings 
           using a fine-tuned transformer model.
        
        2. **Similarity Search**: Given a query article, the system finds the 
           most semantically similar articles using cosine similarity in the 
           embedding space.
        
        3. **Contrastive Learning**: The model was trained using contrastive 
           learning (InfoNCE/Triplet Loss) to map similar articles closer 
           together in the embedding space.
        
        ### Features
        
        - ✅ Semantic similarity search (beyond keyword matching)
        - ✅ Top-K article retrieval
        - ✅ Fine-tuned embeddings using contrastive learning
        - ✅ Fast retrieval using FAISS vector index
        
        ### Technical Details
        
        - **Model**: Fine-tuned SentenceTransformer (MiniLM-based)
        - **Loss Function**: InfoNCE / Triplet Loss
        - **Vector Search**: FAISS (Facebook AI Similarity Search)
        - **Backend**: FastAPI
        - **Frontend**: Streamlit
        
        ### API Endpoints
        
        - `POST /search`: Search for similar articles
        - `GET /health`: Health check
        - `POST /encode`: Encode article to embedding
        
        ### Usage
        
        1. Start the backend server:
           ```bash
           uvicorn backend.api:app --reload
           ```
        
        2. Run this Streamlit app:
           ```bash
           streamlit run frontend/streamlit_app.py
           ```
        
        3. Enter a news article and search for similar articles!
        """)

if __name__ == "__main__":
    main()

