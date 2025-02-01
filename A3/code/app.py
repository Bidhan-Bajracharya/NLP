import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from torchtext.data.utils import get_tokenizer
from nepalitokenizers import WordPiece
import torch, torchtext
from S2S import Seq2SeqTransformer
import os

from Decoder import Decoder, DecoderLayer
from Encoder import Encoder
from Encoder_Layer import EncoderLayer
from S2S import Seq2SeqTransformer 
from Feed_Forward import PositionwiseFeedforwardLayer
from Additive_Attention import AdditiveAttention
from Mutihead_Attention import MultiHeadAttentionLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARG_LANG ='ne'
SRC_LANG='en'

token_transform = {}

current_dir = os.path.dirname(os.path.abspath(__file__))
vocab_path = os.path.join(current_dir, "../model/vocab")

vocab_transform = torch.load(vocab_path)

model_path = '../model/additive_Seq2SeqTransformer.pt'
params, state = torch.load(model_path)
model = Seq2SeqTransformer(**params, device=device).to(device)
model.load_state_dict(state)

# Function to apply sequential operations
def sequential_operation(*transforms):
    """
    Example:
        >>> token_transform["en"]("Hello world!")
        ...     ['hello', 'world', '!']
        >>> vocab_transform["en"](['hello', 'world', '!'])
        ...     [123, 456, 789]
        >>> tensor_transform([123, 456, 789])
        ...     tensor([123, 456, 789])
    """

    def text_operation(input_text):
        for transform in transforms:
            try:
                input_text = transform(input_text)
            except:
                input_text = transform.encode(input_text).tokens # Encoding if error occurs
        return input_text
    return text_operation

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([2]), torch.tensor(token_ids), torch.tensor([2]))) # adding special tokens

text_transform = {}
token_transform["en"] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform["ne"] = WordPiece()

for ln in [SRC_LANG, TARG_LANG]:
    text_transform[ln] = sequential_operation(token_transform[ln], vocab_transform[ln], tensor_transform)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H3("A3: Translation App", className='text-center mt-4'), width=6
        ), justify="center"
    ),
    dbc.Row(
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Enter text to translate:"),
                    dcc.Input(id='user-input', type='text', className='form-control', placeholder='Type something in English...'),
                    html.Br(),
                    dbc.Button("Translate", id='translate-btn', color='primary')
                ])
            ], className='mt-4'), width=6
        ), justify="center"
    ),
    dbc.Row(
        dbc.Col(width=6, id='output-card'), 
        justify="center"
    ),
    
    # Image at the bottom
    html.Div(
        html.Img(src="/assets/shilloute.png", className="silhouette-img"),
        className="silhouette-container"
    ),

    html.Div(
        html.Img(src="/assets/flag.png", className="flag-img"),
        className="flag-container"
    ),

], className='mt-5')

# External CSS for positioning the image
app.css.append_css({"external_url": "/assets/style.css"})

@app.callback(
    Output('output-card', 'children'),
    Input('translate-btn', 'n_clicks'),
    State('user-input', 'value')
)
def translate_text(n_clicks, text):
    nepali_result = None
    
    if n_clicks and n_clicks > 0:
        if not text:
            return dbc.Card(
                [dbc.CardBody([
                        html.P(id='translation-output', children="Please input some text to translate.", className='text-muted')]
                )], className='mt-4')
    
        model.eval() # putting the model in eval mode

        # Tokenize and transform the input sentence to tensors
        input = text_transform[SRC_LANG](text).to(device)
        output = text_transform[TARG_LANG]("").to(device)
        input = input.reshape(1,-1)
        output = output.reshape(1,-1)

        with torch.no_grad():
            output, _ = model(input, output)

        output = output.squeeze(0)
        output = output[1:]
        output_max = output.argmax(1)
        mapping = vocab_transform[TARG_LANG].get_itos()

        nepali_result = [] # translated nepali output

        for token in output_max:
            token_str = mapping[token.item()]
            # Ignoring special tokens
            if token_str not in ['[CLS]', '[SEP]', '[EOS]','<eos>']:
                nepali_result.append(token_str)

        # Joining the tokens to form a sentence
        nepali_result = ' '.join(nepali_result)

        return dbc.Card([dbc.CardBody([
                html.H5("Translated Nepali Output:"),
                html.P(id='translation-output', children=nepali_result, className='text-muted')
            ])], className='mt-4')
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
