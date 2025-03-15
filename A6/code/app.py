import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
# from langchain.chains import LLMChain
# from langchain.llms import OpenAI  # Use any local LLM instead of API if needed

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dummy function simulating LangChain response (replace with actual chain)
def chatbot_response(user_input):
    response = f"Bot: This is a response to '{user_input}'"
    sources = ["Source 1", "Source 2", "Source 3"]  # Replace with actual sources
    return response, sources

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
                    html.Div(id="bot-output", className="p-3 border bg-light", style={"min-height": "100px"}),
                    dbc.Collapse(
                        dbc.Card(dbc.CardBody(html.Ul(id="source-list")), style={"margin-top": "20px"}),
                        id="source-collapse",
                        is_open=False,
                    ),
                ])
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
    if n_clicks > 0 and user_input:
        response, sources = chatbot_response(user_input)
        return response, [html.Li(src) for src in sources], True
    return "", [], False

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
