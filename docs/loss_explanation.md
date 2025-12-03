# Mathematical Explanation of InfoNCE Loss

## Overview

InfoNCE (Information Noise Contrastive Estimation) is a contrastive learning loss function that learns to distinguish between similar and dissimilar data points by maximizing the mutual information between positive pairs while minimizing it for negative pairs.

## Mathematical Formulation

The InfoNCE loss for a batch of samples is defined as:

$$\mathcal{L}_{InfoNCE} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(x_i, x_i^+)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(x_i, x_j)/\tau)}$$

Where:
- $N$ is the batch size
- $x_i$ is the anchor sample
- $x_i^+$ is the positive sample (similar to anchor)
- $x_j$ are all samples in the batch (including positive and negatives)
- $\text{sim}(x_i, x_j)$ is the similarity function (typically cosine similarity)
- $\tau$ is the temperature hyperparameter

## Components Explained

### 1. Similarity Function

The similarity function $\text{sim}(x_i, x_j)$ measures how similar two embeddings are. Commonly used:

**Cosine Similarity:**
$$\text{sim}(x_i, x_j) = \frac{x_i \cdot x_j}{||x_i|| \cdot ||x_j||} = \cos(\theta)$$

Where $\theta$ is the angle between the two vectors.

### 2. Temperature Parameter ($\tau$)

The temperature $\tau$ controls the sharpness of the probability distribution:

- **Small $\tau$ (e.g., 0.05)**: Sharp distribution, model is more confident about positive pairs
- **Large $\tau$ (e.g., 1.0)**: Smooth distribution, model is less confident

The temperature acts as a scaling factor that affects the gradient magnitude during training.

### 3. Softmax Normalization

The denominator $\sum_{j=1}^{N} \exp(\text{sim}(x_i, x_j)/\tau)$ normalizes the similarity scores into a probability distribution using softmax. This means:

$$P(x_j | x_i) = \frac{\exp(\text{sim}(x_i, x_j)/\tau)}{\sum_{k=1}^{N} \exp(\text{sim}(x_i, x_k)/\tau)}$$

The model learns to assign high probability to positive pairs and low probability to negative pairs.

## Intuition

### What InfoNCE Does

1. **Maximizes similarity** between anchor-positive pairs: The numerator $\exp(\text{sim}(x_i, x_i^+)/\tau)$ should be large.

2. **Minimizes similarity** between anchor-negative pairs: The negative samples in the denominator should have low similarity scores.

3. **Creates a contrastive effect**: By comparing positive pairs against all other samples in the batch, the model learns to distinguish between similar and dissimilar content.

### Example

Consider a batch with:
- Anchor: "Breaking news: Stock market crashes"
- Positive: "Financial markets experience major downturn"
- Negative 1: "Sports team wins championship"
- Negative 2: "New movie releases this weekend"

The loss encourages:
- High similarity between anchor and positive (both about financial news)
- Low similarity between anchor and negatives (different topics)

## Relationship to Other Loss Functions

### Triplet Loss

Triplet loss is a simpler form:

$$\mathcal{L}_{triplet} = \max(0, \text{sim}(x_a, x_n) - \text{sim}(x_a, x_p) + \text{margin})$$

InfoNCE is more sophisticated because:
- It considers multiple negatives simultaneously
- It uses softmax normalization
- It has better gradient properties

### Contrastive Loss

Contrastive loss considers pairs:

$$\mathcal{L}_{contrastive} = y \cdot d^2 + (1-y) \cdot \max(0, \text{margin} - d)^2$$

Where $y$ is 1 for positive pairs and 0 for negative pairs, and $d$ is the distance.

InfoNCE generalizes this to multiple negatives and uses a probabilistic formulation.

## Gradient Analysis

The gradient of InfoNCE with respect to the anchor embedding $x_i$ is:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\tau} \left[ \frac{\exp(\text{sim}(x_i, x_j)/\tau)}{\sum_k \exp(\text{sim}(x_i, x_k)/\tau)} \cdot \frac{\partial \text{sim}(x_i, x_j)}{\partial x_i} \right]$$

This shows that:
- The gradient is weighted by the softmax probability
- Hard negatives (high similarity but wrong) receive more gradient signal
- The temperature $\tau$ scales the gradient magnitude

## Practical Considerations

### Choosing Temperature

- **Too small ($\tau < 0.01$)**: Training becomes unstable, gradients explode
- **Too large ($\tau > 1.0$)**: Model doesn't learn to distinguish well
- **Optimal range**: Typically $0.05$ to $0.2$ for text embeddings

### Batch Size

- Larger batches provide more negative samples
- More negatives = better contrastive learning
- Typical batch sizes: 32-128 for contrastive learning

### Hard Negative Mining

Hard negatives (samples that are similar but semantically different) improve learning:
- They provide more informative gradients
- They force the model to learn finer distinctions
- Can be mined using BM25 or initial embeddings

## Implementation in This Project

In our implementation, we use the `MultipleNegativesRankingLoss` from sentence-transformers, which implements InfoNCE:

```python
train_loss = losses.MultipleNegativesRankingLoss(
    model=model, 
    scale=1.0/temperature  # scale = 1/τ
)
```

The `scale` parameter is the inverse of temperature, so:
- `scale=20` corresponds to `τ=0.05`
- `scale=1` corresponds to `τ=1.0`

## References

1. Oord, A. v. d., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748.

2. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. ICML.

3. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP.

