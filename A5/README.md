# DPO
**Direct Preference Optimization (DPO)** is a technique for fine-tuning language models using preference data,  
allowing the model to learn from human feedback without requiring Reinforcement Learning (RL).  

Unlike traditional **Reinforcement Learning with Human Feedback (RLHF)**, which relies on a reward model  
and complex optimization steps (like PPO), **DPO directly optimizes the model's responses** based on  
comparisons between preferred and rejected outputs.

## Why use it?
- **Simpler Training:** Eliminates the need for reinforcement learning (no reward model or PPO required).  
- **More Stable Optimization:** Avoids instability issues common in RLHF training.  
- **Better Alignment:** Models trained with DPO align more naturally with human preferences.  
- **Computationally Efficient:** Requires fewer resources compared to RLHF.  

# Dataset
This project makes use of the **Anthropic HH-RLHF dataset**, which is publicly available on [Hugging Face](https://huggingface.co/datasets/Anthropic/hh-rlhf)

# Training Results

## Training Loss
![]()

## Logits Chosen
![]()

## Logits Rejected
![]()

## Rewards Accuracies
![]()

# To run the app
- Go inside `A5/code folder
- Run `python app.py`
- Go to `http://127.0.0.1:8050/` on the browser of your choice

# Model
My model can be found in HuggingFace, over [here](https://huggingface.co/bidhan-ait/gpt2-a5) 🤗

# Demo
![]()