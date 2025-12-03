# News Article Semantic Similarity & Topic Retrieval Using Contrastive Learning

A semantic retrieval system for news articles that uses contrastive learning to map articles with similar content closer together in the embedding space, enabling highly accurate retrieval of related news.

## Project Overview

This project implements a contrastive learning-based encoder that:
- Maps news articles to fixed-length embeddings
- Retrieves top-K semantically similar articles
- Identifies underlying topic clusters
- Provides an interactive web interface for semantic search

## Features

- **Contrastive Learning**: Fine-tuned embeddings using InfoNCE/Triplet loss
- **Hard Negative Mining**: Improves model discriminative power
- **Baseline Comparison**: Evaluates performance before and after fine-tuning
- **Multi-Task Learning**: Optional topic classification extension
- **Interactive Frontend**: Streamlit interface for real-time semantic search
- **FastAPI Backend**: RESTful API for embedding and retrieval

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the environment:
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add environment to Jupyter (optional):
```bash
python -m ipykernel install --user --name=news-contrastive
```

## Project Structure

```
contrastive_learning/
├── requirements.txt
├── README.md
├── data/
│   └── (dataset will be downloaded here)
├── models/
│   └── (saved models will be stored here)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading and preprocessing
│   ├── triplets.py             # Triplet generation
│   ├── baseline.py             # Baseline evaluation
│   ├── training.py             # Contrastive learning training
│   ├── hard_negatives.py       # Hard negative mining
│   ├── evaluation.py           # Metrics and visualization
│   └── multitask.py            # Multi-task extension
├── backend/
│   └── api.py                  # FastAPI backend
├── frontend/
│   └── streamlit_app.py        # Streamlit frontend
├── notebooks/
│   └── main_notebook.ipynb     # Main Jupyter notebook
└── docs/
    └── loss_explanation.md     # InfoNCE loss mathematics
```

## Usage

### 1. Training the Model

Run the main training script:
```bash
python -m src.training
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/main_notebook.ipynb
```

### 2. Running the Backend API

```bash
uvicorn backend.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### 3. Running the Frontend

```bash
streamlit run frontend/streamlit_app.py
```

The frontend will be available at `http://localhost:8501`

## API Endpoints

- `POST /search`: Search for similar articles
  - Input: `{"article": "your news article text"}`
  - Output: `{"similar_articles": [...], "scores": [...]}`

- `GET /health`: Health check endpoint

## Evaluation Metrics

- **Recall@K**: Proportion of relevant items found in top-K results
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of first relevant result
- **Cosine Similarity**: Semantic similarity between embeddings

## Mathematical Background

The project uses InfoNCE (Information Noise Contrastive Estimation) loss:

$$\mathcal{L}_{i} = - \log \frac{\exp(\text{sim}(x_i, x_i^+)/\tau)}{\sum_{j=0}^{N} \exp(\text{sim}(x_i, x_j)/\tau)}$$

Where:
- $x_i$ is the anchor
- $x_i^+$ is the positive sample
- $\tau$ is the temperature hyperparameter
- $\text{sim}$ is the cosine similarity function

See `docs/loss_explanation.md` for detailed explanation.

## Results

The fine-tuned model shows:
- Improved clustering of similar articles
- Higher Recall@K scores compared to baseline
- Better semantic understanding of news content

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

