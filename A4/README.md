# Files:
- `a4.ipynb` contains Task-1
- `a4_t2_t3.ipynb` contains Task-2 and Task-3

## Evaluation of custom model

| Model Type  | Training Loss with SNLI and MNLI | Cosine Similarity(SNLI and MNLI) | Cosine Similarity (Similar Sentences) | Cosine Similarity (Dissimilar Sentences) |
|------------|------------------------------|----------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| Custom model   | 10.21                        | 0.992                                         | 0.992                                             | 0.999                                             |

<hr>

## Classification Report

| Class           | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Entailment     | 0.12      | 0.00   | 0.01     | 338     |
| Neutral        | 0.50      | 0.01   | 0.02     | 328     |
| Contradiction  | 0.33      | 0.99   | 0.50     | 334     |
| **Accuracy**   |           |        | 0.33     | 1000    |
| **Macro Avg**  | 0.32      | 0.33   | 0.18     | 1000    |
| **Weighted Avg** | 0.32    | 0.33   | 0.18     | 1000    |

<hr>

## Comparison of custom model with pre-trained model

| Model Type | Cosine Similarity (Similar sentence) | Cosine Similarity (Dissisimilar sentence) |
|----------|----------|----------|
| Our Model    | 0.9992    | 0.999    |
| Pre-trained    | 0.731     | 0.483     |

<hr>

# Demo
![](https://github.com/Bidhan-Bajracharya/NLP/blob/main/A4/demo.gif)
