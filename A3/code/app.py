import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

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
    if n_clicks and text:
        return f"Translated: {text}" 
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)
