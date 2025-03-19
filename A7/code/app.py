import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from transformers import AutoModelForSequenceClassification, BertTokenizer
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(current_dir, '../model/trained_even_model')

model_even = AutoModelForSequenceClassification.from_pretrained(dir_path)
tokenizer = BertTokenizer.from_pretrained(dir_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_even.to(device)

model_even.eval()

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model_even(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    
    return "Toxic" if prediction == 1 else "Non-Toxic"

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div(
    className="flex-column d-flex  align-items-center vh-100",
    children=[
        html.H1("A7: Training Distillation vs LoRA", style={"margin-top": "30px", "margin-bottom": "80px"}),
        html.Div(
            className="card p-4 shadow", style={"width": "400px"},
            children=[
                html.H4("Toxicity Classifier", className="text-center mb-3"),
                dcc.Input(id="text-input", type="text", placeholder="Enter text...", className="form-control mb-3"),
                html.Button("Submit", id="submit-btn", className="btn btn-primary w-100"),
            ]
        ),
        html.Div(
            id="output-div",
            className="card mt-3 p-3 shadow text-center",
            style={"width": "400px", "display": "none"},
        ),
    ]
)

# Callback for classification
@app.callback(
    Output("output-div", "children"),
    Output("output-div", "style"),
    Input("submit-btn", "n_clicks"),
    State("text-input", "value"),
    prevent_initial_call=True
)
def classify(n_clicks, text):
    if not text:
        return "Please enter text", {"width": "400px", "display": "block", "color": "red"}
    
    result = classify_text(text)
    color = "red" if result == "Toxic" else "green"
    return result, {"width": "400px", "display": "block", "color": color}

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)