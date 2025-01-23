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

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model
model_path = os.path.join(current_dir, "../model/vocab_lm.pkl")

# Load the vocabulary from the saved file
with open(model_path, 'rb') as f:
    loaded_vocab = pickle.load(f)

# Load the trained LSTM language model
model_path_2 = os.path.join(current_dir, "../model/best-val-lstm_lm.pt")
# model_path_2 = os.path.join(current_dir, "../model/lstm_model.pkl")

vocab_size = len(loaded_vocab)
emb_dim = 1024
hid_dim = 1024
num_layers = 2
dropout_rate = 0.65             
lr = 1e-3   
lstm_model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
lstm_model.load_state_dict(torch.load(model_path_2, map_location=device))

def generate_text(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            # prediction: [batch size, seq len, vocab size]
            # prediction[:, -1]: [batch size, vocab size] # probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']:  # if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:  # if it is eos, we stop
                break

            indices.append(prediction)  # autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
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
            'textAlign': 'center',
            'font-family': 'Arial, sans-serif'
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
            temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]

            results = []

            # path = os.path.join(current_dir, "../model/lstm_model.pkl")
            # with open(path, 'rb') as f:
            #     lstm_model = pickle.load(f)

            # lstm_model.load_state_dict(torch.load(path, map_location=device))
            # lstm_model.to(device)
            # lstm_model.eval()

            for temperature in temperatures:
                tokens = generate_text(
                    prompt=query,
                    max_seq_len=max_seq_len,
                    temperature=temperature,
                    model=lstm_model,
                    tokenizer=tokenizer,
                    vocab=loaded_vocab,
                    device=device,
                    seed=seed
                )

                results.append(html.Div([
                    html.H5(f"Temperature: {temperature}"),
                    html.P(" ".join(tokens), style={'color': 'black', 'font-family': 'Arial, sans-serif'})
                ]))

            return html.Div(results)

    return html.Div("Enter a query to see results.", style={'color': 'gray'})

# Running the app
if __name__ == '__main__':
    app.run_server(debug=True)
