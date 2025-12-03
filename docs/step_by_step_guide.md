# **Step-by-Step Implementation Guide**

# **News Article Semantic Similarity & Topic Retrieval – VS Code + Jupyter Implementation Guide**

---

## **1. Environment Setup**

1. Create a Python virtual environment:

```bash
python -m venv venv

```

1. Activate the environment:
- Windows: `venv\Scripts\activate`

1. Install required packages:

```bash
pip install jupyter notebook ipykernel transformers datasets sentence-transformers torch faiss-cpu scikit-learn umap-learn matplotlib plotly
pip install fastapi uvicorn requests streamlit

```

1. Add your environment to Jupyter:

```bash
python -m ipykernel install --user --name=news-contrastive

```

---

## **2. Load Dataset (Jupyter Cell)**

```python
from datasets import load_dataset

# Example: AG News dataset
dataset = load_dataset("ag_news")

# Inspect sample
dataset['train'][0]

```

---

## **3. Preprocess Text (Jupyter Cell)**

```python
def preprocess(text):
    text = text.lower().strip()
    return text

dataset = dataset.map(lambda x: {"text": preprocess(x["text"])})

```

---

## **4. Build Anchor-Positive-Negative Triplets (Jupyter Cell)**

```python
import random

def create_triplets(dataset):
    triplets = []
    topic_to_articles = {}
    for item in dataset:
        topic_to_articles.setdefault(item['label'], []).append(item['text'])
    for item in dataset:
        anchor = item['text']
        positive = random.choice([t for t in topic_to_articles[item['label']] if t != anchor])
        negative_label = random.choice([l for l in topic_to_articles if l != item['label']])
        negative = random.choice(topic_to_articles[negative_label])
        triplets.append((anchor, positive, negative))
    return triplets

triplets = create_triplets(dataset['train'])

```

---

## **5. Load Pre-Trained Encoder (Jupyter Cell)**

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('all-MiniLM-L6-v2')

```

---

## **6. Baseline Evaluation (Jupyter Cell)**

```python
from sklearn.metrics.pairwise import cosine_similarity
import umap
import matplotlib.pyplot as plt

texts = [item['text'] for item in dataset['test'][:500]]  # small sample
embeddings = model.encode(texts)

# Cosine similarity example
cos_sim = cosine_similarity([embeddings[0]], embeddings[1:])

# Visualize with UMAP
reducer = umap.UMAP()
reduced = reducer.fit_transform(embeddings)
plt.scatter(reduced[:,0], reduced[:,1])
plt.show()

```

---

## **7. Prepare DataLoader for Contrastive Learning (Jupyter Cell)**

```python
train_examples = [InputExample(texts=[a, p, n]) for a, p, n in triplets[:2000]]  # small sample for demo
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.TripletLoss(model=model)

```

---

## **8. Train Model with Contrastive Learning (Jupyter Cell)**

```python
num_epochs = 2
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps
)

# Save fine-tuned model
model.save("news_contrastive_model")

```

---

## **9. Evaluate Fine-Tuned Model (Jupyter Cell)**

```python
embeddings_finetuned = model.encode([item['text'] for item in dataset['test'][:500]])

# Cosine similarity / Recall@K can be computed here
cos_sim_ft = cosine_similarity([embeddings_finetuned[0]], embeddings_finetuned[1:])

# Visualize improved embedding clusters
reduced_ft = reducer.fit_transform(embeddings_finetuned)
plt.scatter(reduced_ft[:,0], reduced_ft[:,1])
plt.show()

```

---

## **10. Hard Negative Mining (Optional, Jupyter Cell)**

```python
from rank_bm25 import BM25Okapi

tokenized_corpus = [doc.split(" ") for doc in [item['text'] for item in dataset['train']]]
bm25 = BM25Okapi(tokenized_corpus)
query = "breaking news in politics".split(" ")
hard_negatives_idx = bm25.get_top_n(query, tokenized_corpus, n=5)
hard_negatives = [dataset['train'][i]['text'] for i in hard_negatives_idx]

```

---

## **11. Multi-Task Extension (Optional / Bonus, Jupyter Cell)**

```python
import torch
from torch import nn

class MultiTaskModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.get_sentence_embedding_dimension(), num_classes)

    def forward(self, input_texts):
        embeddings = self.base_model.encode(input_texts)
        logits = self.classifier(torch.tensor(embeddings))
        return embeddings, logits

```

---

## **12. Frontend Interface (Bonus)**

### **Backend (FastAPI)**

```python
from fastapi import FastAPI
import faiss
import numpy as np

app = FastAPI()

# Load model
model = SentenceTransformer('news_contrastive_model')

# Create FAISS index
embeddings = model.encode([item['text'] for item in dataset['train']])
index = faiss.IndexFlatL2(384)
index.add(np.array(embeddings).astype('float32'))

@app.post("/search")
def search_article(article: str):
    emb = model.encode([article])
    D, I = index.search(np.array(emb).astype('float32'), k=5)
    return {"similar_articles": [dataset['train'][i]['text'] for i in I[0]]}

```

Run FastAPI:

```bash
uvicorn app:app --reload

```

### **Frontend (Streamlit)**

```python
import streamlit as st
import requests

st.title("News Article Semantic Similarity Search")

article = st.text_area("Paste a news article:")
if st.button("Search"):
    response = requests.post("http://127.0.0.1:8000/search", json={"article": article})
    st.write(response.json())

```

Run Streamlit:

```bash
streamlit run streamlit_app.py

```

---

## **13. Math Behind InfoNCE Loss (Documentation / Markdown Cell)**

[

\mathcal{L}*{i} = - \log \frac{\exp(\text{sim}(x_i, x_i^+)/\tau)}{\sum*{j=0}^{N} \exp(\text{sim}(x_i, x_j)/\tau)}

]

- Pushes **anchor-positive** closer and **anchor-negative** farther apart
- Temperature (\tau) controls sharpness
- Helps embeddings form **semantically meaningful clusters**

---

✅ This notebook-based guide can be implemented **cell by cell in VS Code**, and later extended with **frontend FastAPI + Streamlit integration**.

---