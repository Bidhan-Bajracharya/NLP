## Observations during training and validation:

| Attentions | Training Loss | Traning PPL | Validation Loss | Validation PPL |
|----------|----------|----------|----------|----------|
| General Attention    | 7.111     | 1225.625     | 6.568     | 712.002     |
| Multiplicative Attention    | 6.839     | 933.637     | 6.216     | 500.941     |
| Additive Attention    | 6.860     | 953.436     | 6.254     | 520.322     |

General Attention has the highest Training Loss, Training PPL, Validation Loss and Validation PPL. Multiplicative and Additive Attention has better training loss at `6.839` and `6.860` respectively. 

Multiplicative attention has lowest validation loss `(6.216)` and validation ppl `(500.941)`, while additive came in second with validation loss of `(6.254)` and validation ppl `(520.322)`.

As seen, all 3 attention variations improved their loss and perplexity scores during validation, indicating that they perform well given unseen data.

<hr>

## Observation during testing:

| Attentions | Testing Loss | Testing PPL | Model Size (MB) | Inference Time |Avg. Time per epoch |
|----------|----------|----------|----------|----------|----------|
| General Attention    | 6.568     | 712.002     | 52.69     | 0.030     |181.176     |
| Multiplicative Attention    | 6.216     | 500.941     | 52.69     | 0.016     |185.551    |
| Additive Attention    | 0.390     | 1.477     | 52.69     | 0.025     |202.331     |

All three model has same size of `52.69` MB. In terms of computational efficiency, general attention variation has the lowest average epoch time of `181.176s`, multiplicative coming in second with `185.551s` and finally additive with `202.331s`. 

Similarly, all three variations had low inference time. Among them, Multiplicative attention having the lowest inference time suggests that it can do faster predictions during translation.

<hr>

## Performance Graph

### Additive Attention

![](https://github.com/Bidhan-Bajracharya/NLP/blob/main/A3/static/additive.png)

### General Attention

![](https://github.com/Bidhan-Bajracharya/NLP/blob/main/A3/static/general.png)

### Multiplicative Attention

![](https://github.com/Bidhan-Bajracharya/NLP/blob/main/A3/static/multi.png)

## Demo

![](https://github.com/Bidhan-Bajracharya/NLP/blob/main/A3/static/demo.gif)
