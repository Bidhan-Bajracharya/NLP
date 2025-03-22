# Dataset
The dataset was taken from [HuggingFace](https://huggingface.co/nicholasKluge/ToxicityModel), named `nicholasKluge/ToxicityModel`.

# Evaluation and Analysis

## Even layered distilled BERT
For the even layered model scores were as such:
- Accuracy: 0.9380
- Precision: 0.9383
- Recall: 0.9380
- F1: 0.9380

It shows good overall performance. The classification report indicates a balanced performance, with precision and recall both close to 94%. The model performed slightly better on the negative class (precision of 0.95), but still maintained strong recall for both classes. We further see from confusion matrix that it achieved:
- TN: 471
- TP: 467
- FP: 37
- FN: 25

## Odd layered distilled BERT
Moving on to odd layered model, the scores were as such:
- Accuracy: 0.9340
- Precision: 0.9342
- Recall: 0.9340
- F1: 0.9340

It also shows good overall performance, but even layer out-performed it. The classification report shows that the model had a lower precision (0.92 for positive samples), meaning it made more false positives compared to the even-layered model. However, its recall for the positive class (0.94) was better than its precision, showing it was able to capture more positive instances. We further see from confusion matrix that it achieved:
- TN: 470
- TP: 464
- FP: 38
- FN: 28

## LoRA (Low-Rank Adaptation)
Finally, after implementing LoRA to the 12-layer student model, the best model's scores achieved were as such:
- Accuracy: 0.8990
- Precision: 0.8992
- Recall: 0.8990
- F1: 0.8990

It showed good performance, but is less than both the even and odd layered models. The report showed that the precision for the negative class was 0.91, and recall was 0.89, meaning the model had a higher chance of making false negatives. We further see from confusion matrix that it achieved:
- TN: 453
- TP: 446
- FP: 55
- FN: 46

So, the best model among all three was the even-layered distilled BERT model as it out-performed all the models in terms of accuracy, precision, recall, and F1 score.

## Observation

| Model Type   | Training Loss (Final Epoch) | Accuracy | Precision | Recall | F1 Score |
|--------------|---------------|----------|-----------|--------|----------|
| Odd Layer    | 0.1661 | 93.40%   | 93.42%    | 93.40% | 93.40%   |
| Even Layer   | 0.1657 | 93.80%   | 93.83%    | 93.80% | 93.80%   |
| LoRA         | 0.6971 | 89.90%   | 89.92%    | 89.90% | 89.90%   |



# Demo
![]()
