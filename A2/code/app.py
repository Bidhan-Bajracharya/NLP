from dash import Dash, html, dcc, Input, Output, State
import numpy as np
import pickle
import torch, torchtext
from torchtext.data.utils import get_tokenizer
from lstm import LSTMLanguageModel
import os

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize a basic English tokenizer from torchtext
tokenizer = get_tokenizer('basic_english')

# Absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the model
model_path = os.path.join(current_dir, "../model/vocab_lm.pkl")

# Loading the vocabulary from the saved file
with open(model_path, 'rb') as f:
    loaded_vocab = pickle.load(f)

# Loading the trained LSTM language model
model_path_2 = os.path.join(current_dir, "../model/best-val-lstm_lm.pt")

# Same hyperparameters as used during training
vocab_size = len(loaded_vocab)
emb_dim = 1024
hid_dim = 1024
num_layers = 2
dropout_rate = 0.65     

lstm_model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
lstm_model.load_state_dict(torch.load(model_path_2, map_location=device))

def generate_text(prompt, max_seq, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    tokens  = tokenizer(prompt)
    indices = [vocab[token] for token in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for _ in range(max_seq):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            probability = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction  = torch.multinomial(probability, num_samples=1).item()    
            
            while prediction == vocab['<unk>']:  # sample again if <unk>
                prediction = torch.multinomial(probability, num_samples=1).item()

            if prediction == vocab['<eos>']:  # stop if <eos>
                break

            indices.append(prediction)  # output becomes input because autoregressive

    itos    = vocab.get_itos() # List mapping indices to tokens
    tokens  = [itos[i] for i in indices]
    return tokens


app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("A2: Language Model", style={'textAlign': 'center', 'font-family': 'Arial, sans-serif', 'margin-top': '20px'}),
    html.Div([
        html.Div([
            dcc.Input(
                id='search-query',
                type='text',
                placeholder='Enter your text(s)...',
                style={
                    'width': '70%',
                    'margin': '0 auto',
                    'padding': '10px',
                    'display': 'block'
                }
            ),
            html.Button(
                'Search',
                id='search-button',
                n_clicks=0,
                style={
                    'padding': '10px 20px',
                    'background-color': '#007BFF',
                    'color': 'white',
                    'border': 'none',
                    'border-radius': '5px',
                    'margin-top': '20px',
                    'display': 'block',
                    'margin-left': 'auto',
                    'margin-right': 'auto'
                }
            ),
        ], style={
            'textAlign': 'center',
            'padding': '20px',
            'background-color': '#f9f9f9',
            'border': '1px solid #e0e0e0',
            'border-radius': '10px',
            'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            'width': '50%',
            'margin': '0 auto'
        }),
    ], style={'margin-top': '40px'}),
    html.Div(
        id='search-results',
        style={
            'margin-top': '40px',
            'padding': '20px',
            'font-family': 'Arial, sans-serif',
            'display': 'flex',
            'justify-content': 'center',
        }
    ),
])

# Callback to handle search queries
@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks')],
    [State('search-query', 'value')]
)
def search(n_clicks, query):
    if n_clicks > 0:
        if not query:
            return html.Div("Please enter some word(s).", style={'color': 'red'})

        else:
            max_seq_len = 30
            seed = 122
            temperatures = [0.5, 0.7, 0.75, 0.8, 1.0] # same temperatures as used during inference

            results = []

            for temperature in temperatures:
                tokens = generate_text(
                    prompt      = query,
                    max_seq     = max_seq_len,
                    temperature = temperature,
                    model       = lstm_model,
                    tokenizer   = tokenizer,
                    vocab       = loaded_vocab,
                    device      = device,
                    seed        = seed
                )

                results.append(html.Div([
                    html.H5(f"Temperature: {temperature}", style={'margin-bottom': '10px', 'font-family': 'Arial, sans-serif'}),
                    html.P(" ".join(tokens), style={'color': 'black', 'font-family': 'Arial, sans-serif', 'textAlign': 'left'})
                ]))

            return html.Div(results, style={
                'background-color': '#f9f9f9',
                'border': '1px solid #e0e0e0',
                'border-radius': '10px',
                'padding': '20px',
                'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                'textAlign': 'left',
                'max-width': '50%',
            })

    return html.Div("Enter a query to see results.", style={'color': 'gray'})

# Running the app
if __name__ == '__main__':
    app.run_server(debug=True)
