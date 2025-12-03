# Quick Start Guide

This guide will help you get started with the News Article Semantic Similarity project.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd contrastive_learning
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Add environment to Jupyter (optional):**
   ```bash
   python -m ipykernel install --user --name=news-contrastive
   ```

## Quick Start Options

### Option 1: Run the Main Training Script

Train the model using the command-line interface:

```bash
python main.py --max_triplets 5000 --epochs 3
```

For more options:
```bash
python main.py --help
```

### Option 2: Use Jupyter Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `notebooks/main_notebook.ipynb`

3. Run cells sequentially to:
   - Load and preprocess data
   - Create triplets
   - Evaluate baseline
   - Train the model
   - Evaluate fine-tuned model

### Option 3: Use Individual Modules

You can also use the modules individually:

```python
from src.data_loader import load_news_dataset, preprocess_dataset
from src.triplets import create_triplets_from_dataset
from src.training import ContrastiveTrainer

# Load data
dataset = load_news_dataset("ag_news")
dataset = preprocess_dataset(dataset)

# Create triplets
triplets = create_triplets_from_dataset(dataset['train'], max_triplets=2000)

# Train
trainer = ContrastiveTrainer()
train_dataloader = trainer.prepare_dataloader(triplets, batch_size=32)
trainer.train(train_dataloader, num_epochs=2)
```

## Running the Web Application

### 1. Create FAISS Index

First, create the FAISS index for fast retrieval:

```bash
python backend/create_index.py
```

### 2. Start the Backend API

```bash
uvicorn backend.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### 3. Start the Frontend

In a new terminal:

```bash
streamlit run frontend/streamlit_app.py
```

The frontend will be available at `http://localhost:8501`

## Project Structure

```
contrastive_learning/
├── src/                    # Source code modules
│   ├── data_loader.py      # Dataset loading
│   ├── triplets.py         # Triplet generation
│   ├── baseline.py         # Baseline evaluation
│   ├── training.py         # Model training
│   ├── evaluation.py       # Evaluation metrics
│   ├── hard_negatives.py   # Hard negative mining
│   └── multitask.py        # Multi-task extension
├── backend/                # FastAPI backend
│   ├── api.py             # API endpoints
│   └── create_index.py    # Index creation script
├── frontend/               # Streamlit frontend
│   └── streamlit_app.py   # Web interface
├── notebooks/              # Jupyter notebooks
│   └── main_notebook.ipynb
├── docs/                   # Documentation
│   └── loss_explanation.md
├── models/                 # Saved models (created after training)
├── data/                   # Dataset storage
├── main.py                 # Main training script
├── requirements.txt        # Dependencies
└── README.md              # Project documentation
```

## Example Usage

### Training with InfoNCE Loss

```bash
python main.py --loss infonce --temperature 0.05 --epochs 3
```

### Training with Hard Negatives

```bash
python main.py --use_hard_negatives --max_triplets 10000
```

### Evaluation Only

```bash
python main.py --skip_training --skip_baseline
```

## API Usage Examples

### Search for Similar Articles

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/search",
    json={
        "article": "Breaking news: Stock market crashes today",
        "top_k": 5
    }
)

results = response.json()
print(results)
```

### Encode an Article

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/encode",
    json={"article": "Your news article text here"}
)

embedding = response.json()["embedding"]
print(f"Embedding dimension: {len(embedding)}")
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Make sure you're in the project root directory and the virtual environment is activated.

### Issue: CUDA out of memory

**Solution:** Reduce batch size:
```bash
python main.py --batch_size 16
```

### Issue: API not responding

**Solution:** 
1. Check if the backend is running: `uvicorn backend.api:app --reload`
2. Verify the index exists: `ls models/faiss_index.bin`
3. If index doesn't exist, run: `python backend/create_index.py`

### Issue: Dataset download fails

**Solution:** The dataset will be downloaded automatically on first use. If it fails, check your internet connection or try a different dataset.

## Next Steps

1. Experiment with different loss functions (triplet, infonce, cosine)
2. Try different base models (BERT, RoBERTa, etc.)
3. Adjust hyperparameters (temperature, batch size, learning rate)
4. Implement hard negative mining for better results
5. Extend with multi-task learning

## Resources

- [Documentation](README.md)
- [Loss Function Explanation](docs/loss_explanation.md)
- [Functional Requirements](functional_requirment.md)
- [Step-by-Step Guide](step_by_step_guide.md)

## Support

For issues or questions, please refer to the documentation or create an issue in the repository.

