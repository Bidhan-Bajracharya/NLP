import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.document_loaders import TextLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

current_dir = os.path.dirname(os.path.abspath(__file__))
prompt_template = """
    You are an assistant that provides useful information about Bidhan. Use the following pieces of retrieved context to answer the question. 
    If the question is about Bidhan's education, summarize his degrees, institutions, and fields of study.
    If you don't know the answer, just say that you don't know. Keep the answer concise.
    {context}
    Question: {question}
    Answer:
    """.strip()

PROMPT = PromptTemplate(template= prompt_template)
llm = OllamaLLM(model="llama3.2")

all_documents = []

aboutme_file = os.path.join(current_dir, "../data/aboutme.txt")
cv_file = os.path.join(current_dir, "../data/cv.txt")

files = [aboutme_file, cv_file]
for file in files:
    loader = TextLoader(file)
    documents = loader.load()
    all_documents.extend(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc = text_splitter.split_documents(all_documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_path = os.path.join(current_dir, "../vector-store")
db_file_name = 'personal_info'

vectordb = FAISS.load_local(
    folder_path=os.path.join(vector_path, db_file_name),
    embeddings=embedding_model,
    allow_dangerous_deserialization=True # because I'm sure the file is safe
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer")
qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = retriever,
    return_source_documents = True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Layout
app.layout = dbc.Container([
    html.H1("A6: Let's Talk with Yourself", className="text-center", style={"margin-bottom": "80px"}),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Ask a Question", className="text-center"),
                    dcc.Textarea(
                        id="user-input",
                        placeholder="Type your question here...",
                        style={"width": "100%", "height": "100px"},
                    ),
                    dbc.Button("Submit", id="submit-btn", color="primary", className="mt-2", n_clicks=0)
                        ])
                    ]),
        ], className="p-3 border-end"),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Answer", className="text-center"),
                    # html.Div(id="bot-output", className="p-3 border bg-light", style={"min-height": "100px"}),
                    dcc.Loading(
                        id="loading-spinner",
                        children=[
                            html.Div(
                                id='bot-output',
                                className="p-3 border bg-light", style={"min-height": "100px"}
                            )
                        ],
                        type="circle" 
                    ),
                    dbc.Collapse(
                        dbc.Card(
                            dcc.Loading(
                                id="load-spinner",
                                children=[
                                    dbc.CardBody([
                                        html.H5("Sources", className="card-title"),
                                        html.Ul(id="source-list")]
                                    )
                                ],
                                type="circle" 
                            ), style={"margin-top": "20px"}
                        ),
                        id="source-collapse",
                        is_open=False,
                    ),
                ]),
            ], className="mt-3"),
        ]),
    ])
], fluid=True, style={"display": "flex", "flex-direction": "column", "height": "100vh", "padding": "30px 60px"})

# Callbacks
@app.callback(
    [Output("bot-output", "children"), Output("source-list", "children"), Output("source-collapse", "is_open")],
    [Input("submit-btn", "n_clicks")],
    [State("user-input", "value")]
)
def update_chat(n_clicks, user_input):
    sources = []
    if n_clicks > 0 and user_input:
        result = qa_chain({"query": user_input})
        source_docs = result.get('source_documents', [])

        for doc in source_docs:
            sources.append(doc.metadata.get('source', 'Unknown Source'))
        
        print(sources)
        unique_sources = list(set(sources))

        response = result["result"]
        return response, [html.Li(src) for src in unique_sources], True
    return "", [], False

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
