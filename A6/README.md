### List of the retriever and generator models used

####  Retriever model
**Embedding Model** <br>
For embeddings I used `sentence-transformers/all-mpnet-base-v2` from HuggingFace.

**Vector Store**<br>
For vector store I used `FAISS` which stands for Facebook AI Similarity Search, from langchain.

#### Generator model
For the llm model I went ahead and chose `llama3.2` from Ollama

### Analysis of issues related to the models providing unrelated information.

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

### Analysis summary: 

| **Issue**                      | **Description** | **Possible Fixes** |
|--------------------------------|---------------|----------------|
| **Embedding Model Limitations** | The embedding model (`sentence-transformers/all-mpnet-base-v2`) may struggle with accurately capturing semantic meaning, domain-specific terminology, or short queries, leading to incorrect document retrieval. | - Use **hybrid retrieval** (combine FAISS with BM25 or keyword search). |
| **Vector Store (FAISS) Issues** | FAISS may return the closest match even if it's contextually irrelevant, due to small dataset size, incorrect indexing, or missing important context. | - Break documents into **overlapping chunks** for better retrieval. <br> - Use **smaller, context-rich sections** instead of large paragraphs. <br> - Apply **reranking methods** to improve result relevance. |
| **LLM (LLaMA 3.2) Hallucinations** | The LLM might generate incorrect responses due to over-reliance on poor retrieval, lack of source verification, or bias in generation. | - Implement **confidence threshold** to prevent hallucinations. <br> - If no relevant document is found, reply with *"I have no information on this."* <br> - Consider **fine-tuning** the model for better accuracy. |
