# Dataset
The dataset was taken from [HuggingFace](https://huggingface.co/nicholasKluge/ToxicityModel), named `nicholasKluge/ToxicityModel`.

# Metric Visualizations

## Training metrics

<p align="center">
  <img src="https://github.com/Bidhan-Bajracharya/NLP/blob/main/A7/static/even.png" width="30%" />
  <img src="https://github.com/Bidhan-Bajracharya/NLP/blob/main/A7/static/odd.png" width="30%" />
  <img src="https://github.com/Bidhan-Bajracharya/NLP/blob/main/A7/static/lora.png" width="30%" />
</p>

## Testing metrics

<p align="center">
  <div>
    <h3>Even Layer Confusion Matrix</h3>
    <img src="https://github.com/Bidhan-Bajracharya/NLP/blob/main/A7/static/even_confusion.png" />
  </div>
  <div>
    <h3>Odd Layer Confusion Matrix</h3>
    <img src="https://github.com/Bidhan-Bajracharya/NLP/blob/main/A7/static/odd_confusion.png" />
  </div>
  <div>
    <h3>LoRA Confusion Matrix</h3>
    <img src="https://github.com/Bidhan-Bajracharya/NLP/blob/main/A7/static/lora_confusion.png" />
  </div>
</p>

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
| LoRA         | 0.1252 | 89.90%   | 89.92%    | 89.90% | 89.90%   |

## Discussion of Challenges

### Implementation

In terms of implementation, the odd and even layer distillation required careful layer-wise mapping from the teacher model. But this challenge was tackled easily due to the reference code provided.

### Training

In terms of training and resource usage, LoRA was more memory efficient as only adapter parameters were updated but for distillation it's slower because multiple layers have to be distilled seperately. So, to improve distillation efficiency - distillation of only the key layers can be done and also gradient checkpoint can also be applied to reduce the memory consumption and speed up the training.

For both the odd and even layered distillation, following training and validation performance were observed:
- Total Train Loss is Decreasing: This indicates that the model is learning and optimizing during training.

- Classification Loss `(Train Loss_cls)` is Steadily Decreasing: This is a positive sign, as the model is improving its classification ability. However, it flattens in later epochs, indicating that learning is slowing down.

- Divergence Loss `(Train Loss_div)` is Increasing: This could indicate that the student model is struggling to match the teacher model’s representations, which might affect generalization.

- Cosine Similarity Loss `(Train Loss_cos)` is Decreasing: This suggests that the model is gradually aligning its representation with the teacher model.

- Validation Loss is Stagnating and Slightly Increasing: This is a critical. Even though the training loss is decreasing, the validation loss is not improving and is instead slightly increasing, which is a sign of overfitting.

For improvement following techniques can be applied:
- Apply regularization techniques
- Apply early stopping to prevent overfitting
- Since, student model is not effictively mimicking the teacher model, we can reduce the weight of divergence loss or gradually introduce it after a few epochs
- Can apply learning rate scheduler to help prevent performance degradation

LoRA performed well during the training. The training and validaiton loss gradually decreased with slight fluctuation - but we can see an overall decreasing trend. Which suggests that our model is generalizing well.

# Web application interface documentation
For this assignment, I made the interface using Dash. The entire UI along with necessary model implemenetations are carried out in `app.py` file. It is a simple UI with a prompt field, submit button, some validations and a result section. The demo can be found in the `README.md` file of A7 folder and the actual `demo.gif` file is inside `static` folder.

The model used for this demo is **even-layered distilled** model - as from the above observations it performed the best, even though all three models can classify farily well.

The user interaction flow is as follows:
- User enters a prompt into the field
- Clicks the submit button
- Classification result is displayed as either **Toxic** or **Non-Toxic**

# Demo
![](https://github.com/Bidhan-Bajracharya/NLP/blob/main/A7/static/demo.gif)
