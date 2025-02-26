import dash
from dash import dcc, html, Input, Output, State
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
        style={
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
    },
    children=[
        html.Div(
            style={
                'fontFamily': 'Arial, sans-serif',
                'backgroundColor': '#f4f7f6',
                'padding': '30px',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                'width': '40%',
            },
            children=[
                html.H2(
                    "Start Chatting d(￣◇￣)b",
                    style={
                        'textAlign': 'center',
                        'color': '#4a90e2',
                        'fontSize': '32px',
                        'marginBottom': '20px'
                    }
                ),
                
                # User input section
                html.Div(
                    children=[
                        dcc.Textarea(
                            id='user-input',
                            style={
                                'width': '96%',
                                'height': 120,
                                'padding': '10px',
                                'borderRadius': '8px',
                                'border': '1px solid #ccc',
                                'fontSize': '16px',
                                'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
                                'resize': 'none'
                            }
                        ),
                        html.Button(
                            'Submit',
                            id='submit-button',
                            n_clicks=0,
                            style={
                                'backgroundColor': '#4a90e2',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'fontSize': '16px',
                                'borderRadius': '5px',
                                'cursor': 'pointer',
                                'marginTop': '10px',
                                'width': '100%',
                                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                                'transition': 'background-color 0.3s'
                            }
                        ),
                    ],
                    style={'marginBottom': '20px'}
                ),
                
                # Store the user input in a state
                dcc.Store(id='user-input-store'),
            ]
        ),
        html.Div(
            style={
                'fontFamily': 'Arial, sans-serif',
                'backgroundColor': '#f4f7f6',
                'padding': '30px',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                'width': '40%',
                'marginTop': '30px',
            },
            children=[
                html.Div(
                    children=[
                        html.H3("Response:", style={'marginTop': '20px'}),
                        dcc.Loading(
                            id="loading-spinner",
                            children=[
                                html.Div(
                                    id='model-response',
                                    style={
                                        'padding': '20px',
                                        'border': '1px solid #ddd',
                                        'borderRadius': '8px',
                                        'backgroundColor': '#fff',
                                        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
                                        'whiteSpace': 'pre-wrap',
                                        'fontSize': '16px',
                                        'color': '#333',
                                        'maxHeight': '300px',
                                        'overflowY': 'auto',
                                    }
                                )
                            ],
                            type="circle" 
                        ),
                    ],
                ),
            ]
        ),
    ]
)

@app.callback(
    Output('user-input-store', 'data'),
    Input('submit-button', 'n_clicks'),
    State('user-input', 'value')
)
def update_input_state(n_clicks, user_input):
    if n_clicks > 0 and user_input:
        return {'input': user_input}
    return dash.no_update

@app.callback(
    Output('model-response', 'children'),
    Input('user-input-store', 'data')
)
def update_response(user_input_data):
    if user_input_data:
        return generate_response(user_input_data['input'])
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
