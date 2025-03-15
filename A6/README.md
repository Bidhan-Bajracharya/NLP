## List of the retriever and generator models used

###  Retriever model
**Embedding Model** <br>
For embeddings I used `sentence-transformers/all-mpnet-base-v2` from HuggingFace.

**Vector Store**<br>
For vector store I used `FAISS` which stands for Facebook AI Similarity Search, from langchain.

### Generator model
For the llm model I went ahead and chose `llama3.2` from Ollama

## Analysis of issues related to the models providing unrelated information.

**Embedding Model Limitations**

The embedding model `(sentence-transformers/all-mpnet-base-v2)` converts text into vector representations for similarity searches. However, issues may arise if:
- The embeddings do not accurately capture the semantic meaning of a query, leading to retrieval of irrelevant documents.
- The model struggles with understanding domain-specific terminology or names.
- Short or ambiguous queries produce vague embeddings, retrieving unrelated results.

**For example**: User may ask, "What awards has bidhan achieved?" but it gets information about project achievements.

Possible fixes for this issue would be:
- Use hybrid retrieval (combining FAISS with BM25 or keyword search).

**Vector Store (FAISS) Issues**

FAISS is efficient for storing and retrieving embeddings, but it has its own challenges:

- If the database is small or lacks diversity, FAISS may return the closest match, even if it is not contextually relevant.
- Improper indexing or incorrect similarity metrics can result in documents that are mathematically similar but contextually irrelevant.
- If the retriever fetches too few documents, important context may be missed.

**For example**: User may ask, "Tell me about his job experiences." but the documents it retrieved was of job's project details.

Possible fixes for this issue would be:
- Break documents into overlapping chunks so relevant details remain intact.
- Instead of large paragraphs, creating smaller, context-rich sections.
- Apply reranking methods.

**LLM (LLaMA 3.2) Hallucinations**

The generator model (LLaMA 3.2) may produce unrelated or incorrect information due to:

- Over-reliance on context: If retrieved documents are not strongly relevant, the model may fill gaps with plausible but incorrect content.
- Lack of source verification: Since LLaMA is a general-purpose model, it might blend retrieved information with its own prior knowledge, causing inconsistencies.
- Bias in generation: The model may generate responses that favor more commonly seen patterns, even if they are not the best fit for the query.

**For example**: User may ask, "Tell me about his hobbies." but the retriever provides information relating to his technical skills,  so the llm might makeup informations.

Possible fixes for this issue would be:
- If the retriever finds no relevant documents, the chatbot should ask for clarification instead of hallucinating.
- Reply saying "I have no information on this." or something similar by implementing confidence threshold.
- Finetune the model further.

## Analysis summary: 

| **Issue**                      | **Description** | **Possible Fixes** |
|--------------------------------|---------------|----------------|
| **Embedding Model Limitations** | The embedding model (`sentence-transformers/all-mpnet-base-v2`) may struggle with accurately capturing semantic meaning, domain-specific terminology, or short queries, leading to incorrect document retrieval. | - Use **hybrid retrieval** (combine FAISS with BM25 or keyword search). |
| **Vector Store (FAISS) Issues** | FAISS may return the closest match even if it's contextually irrelevant, due to small dataset size, incorrect indexing, or missing important context. | - Break documents into **overlapping chunks** for better retrieval. <br> - Use **smaller, context-rich sections** instead of large paragraphs. <br> - Apply **reranking methods** to improve result relevance. |
| **LLM (LLaMA 3.2) Hallucinations** | The LLM might generate incorrect responses due to over-reliance on poor retrieval, lack of source verification, or bias in generation. | - Implement **confidence threshold** to prevent hallucinations. <br> - If no relevant document is found, reply with *"I have no information on this."* <br> - Consider **fine-tuning** the model for better accuracy. |

## JSON Response
```
[
    {
        "question": "How old are you?",
        "answer": "You are 23 years old (as of 2025)."
    },
    {
        "question": "What is your highest level of education?",
        "answer": "Your highest level of education is a Master's degree, specifically in Data Science and Artificial Intelligence (DSAI) from Asian Institute Of Technology."
    },
    {
        "question": "What major or field of study did you pursue during your education?",
        "answer": "Bidhan pursued a BSc (Hons) in Computing from Islington College (Affiliated to London Metropolitan University), and later completed his Master's degree in Data Science and AI from the Asian Institute Of Technology."
    },
    {
        "question": "How many years of work experience do you have?",
        "answer": "You have 1 year of work experience as a Software Engineer in the education industry."
    },
    {
        "question": "What type of work or industry have you been involved in?",
        "answer": "Bidhan has been involved in the following types of work and industries:\n\n1. Education (as a Software Engineer for 1 year)\n2. IT and software development (as a Junior Software Engineer at Vurilo Pvt. Ltd., which is likely an IT company)\n3. Data analysis and science (applying concepts learned through Masters education, though specific details are not provided)\n\nHe has worked on various projects, including web applications, data analysis, and machine learning-related tasks."
    },
    {
        "question": "Can you describe your current role or job responsibilities?",
        "answer": "As per the provided context, Bidhan is currently a student with no formal job responsibilities mentioned. However, his role as a Student Academic Representative in previous years (2021 & 2023) involved representing class interests and facilitating communication between students and faculty."
    },
    {
        "question": "What are your core beliefs regarding the role of technology in shaping society?",
        "answer": "Bidhan's core belief is that technology can be a double-edged sword, driving societal progress when used ethically but leading to negative consequences if misused. He emphasizes the importance of using data-driven decision-making, prioritizing empathy and ethical responsibility, and promoting balance between nature and technology."
    },
    {
        "question": "How do you think cultural values should influence technological advancements?",
        "answer": "According to the provided context, Bidhan's cultural values that may influence technological advancements include:\n\n1. Respect for Tradition & Innovation: Integrating traditional wisdom with modern technology.\n2. Community & Collaboration: Using technology to bring people together rather than isolate them.\n3. Empathy & Ethical Responsibility: Considering the impact of technology on people's lives and prioritizing ethical practices.\n\nThese values suggest that Bidhan believes in creating a balance between preserving cultural heritage and embracing innovation, while also prioritizing community, empathy, and ethical responsibility in technological development."
    },
    {
        "question": "As a master's student, what is the most challenging aspect of your studies so far?",
        "answer": "As a Master's student in Data Science and AI at Asian Institute Of Technology, Bidhan has faced challenges such as integrating data science into his fields of interest, managing time for personal projects, figuring out which field to primarily focus on, and catching up with advanced ML concepts."
    },
    {
        "question": "What specific research interests or academic goals do you hope to achieve during your time as a master's student?",
        "answer": "During his time as a master's student, Bidhan aims to:\n\n1. Learn useful ML concepts and apply them in real-case scenarios\n2. Get into a good company for an internship\n3. Network with good people and grow/learn from them\n4. Currently has no intention of doing research, focusing on job opportunities\n\nHe doesn't specify any particular research interests or academic goals."
    }
]
```

## Demo
![](https://github.com/Bidhan-Bajracharya/NLP/blob/main/A6/static/demo.gif)
