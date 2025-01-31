import dash
from dash import dcc, html, Input, Output
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

TRG_LANGUAGE ='ne'
SRC_LANGUAGE='en'

token_transform = {}

current_dir = os.path.dirname(os.path.abspath(__file__))
vocab_path = os.path.join(current_dir, "../model/vocab")

vocab_transform = torch.load(vocab_path)

model_path = '../model/additive_Seq2SeqTransformer.pt'
params, state = torch.load(model_path)
model = Seq2SeqTransformer(**params, device=device).to(device)
model.load_state_dict(state)

# Function to preprocess a source sentence (tokenization, normalization, etc.)
def preprocess_src_sentence(sentence, lang):
    token_transform["en"] = get_tokenizer('spacy', language='en_core_web_sm')
    return {lang: token_transform[lang](sentence.lower())}

# Function to preprocess a target sentence (tokenization, normalization, etc.)
def preprocess_trg_sentence(sentence, lang):
    token_transform["ne"] = WordPiece()
    return {lang: token_transform[lang].encode(sentence.lower()).tokens}

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            try:
                txt_input = transform(txt_input)
            except:
                # If an exception occurs, assume it's an encoding and use encode function
                txt_input = transform.encode(txt_input).tokens
        return txt_input
    return func

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([2]), torch.tensor(token_ids), torch.tensor([2])))


# src and trg language text transforms to convert raw strings into tensors indices
text_transform = {}
token_transform["en"] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform["ne"] = WordPiece()
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform)

text_transform = {}
token_transform["en"] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform["ne"] = WordPiece()
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform)


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
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Translated Nepali Output:"),
                    html.P(id='translation-output', children="", className='text-muted')
                ])
            ], className='mt-4'), width=6
        ), justify="center"
    )
], className='mt-5')

@app.callback(
    Output('translation-output', 'children'),
    Input('translate-btn', 'n_clicks'),
    Input('user-input', 'value')
)
def translate_text(n_clicks, text):
    if not text:
        return ""
    
    translation_result = None
    
    if n_clicks and text:
        model.eval() # putting the model in eval mode

        # Tokenize and transform the input sentence to tensors
        input = text_transform[SRC_LANGUAGE](text).to(device)
        print("==",input)
        output = text_transform[TRG_LANGUAGE]("").to(device)
        input = input.reshape(1,-1)
        output = output.reshape(1,-1)

        with torch.no_grad():
            output, _ = model(input, output)

        output = output.squeeze(0)
        output = output[1:]
        print(output)
        output_max = output.argmax(1)
        print("OutputMax",output_max)
        mapping = vocab_transform[TRG_LANGUAGE].get_itos()

        translation_result = []

        # Process the output tokens
        for token in output_max:
            token_str = mapping[token.item()]
            if token_str not in ['[CLS]', '[SEP]', '[EOS]','<eos>']:
                translation_result.append(token_str)
                print(translation_result)

        translation_result = ' '.join(translation_result)

        return f"Translated: {translation_result}" 
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
