import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas 
import dash
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer 
import plotly.graph_objs as go
from wordcloud import WordCloud
from time import time
import pickle
import numpy as np
from collections import defaultdict
from app import app

# #####################Table et autres ###################

df = pandas.read_csv('apps/data/Emotion_final_.csv')

# ############################## Fin ###################

# #########################Navbar #########################
nav_2 = html.Nav(className='container', children=[
    html.Ul(className='basse', children=[
        html.Li(className='one', children=[
           dcc.Link(className='lien',children=['Home'], href='/')
        ]),
        html.Li(className='two', children=[
            dcc.Link(className='lien',children=['Resultat des algos'], href='/app2')
        ]),
         html.Li(className='three', children=[
            dcc.Link(className='lien',children=['Brief'], href='/brief')
        ]),
        html.Hr()
    ])
])
# #################################
def print_table1(res1):
    # Compute mean 
    final = {}
    for model in res1:
        arr = np.array(res1[model])
        final[model] = {
            "name" : model, 
            "time" : arr[:, 0].mean().round(2),
            "f1_score": arr[:,1].mean().round(3),
            #"Precision" : arr[:,2].mean().round(3),
            #"Recall" : arr[:,2].mean().round(3)
        }
    df4 = pandas.DataFrame.from_dict(final, orient="index").round(3)
    return df4
def print_table2(res):
    # Compute mean 
    final = {}
    for model in res:
        arr = np.array(res[model])
        final[model] = {
            "name" : model, 
            "time" : arr[:, 0].mean().round(2),
            "f1_score": arr[:,1].mean().round(3),
            #"Precision" : arr[:,2].mean().round(3),
            #"Recall" : arr[:,2].mean().round(3)
        }
    df2 = pandas.DataFrame.from_dict(final, orient="index").round(3)
    return df2
    
def print_table3(res2):
    # Compute mean 
    final = {}
    for model in res2:
        arr = np.array(res2[model])
        final[model] = {
            "name" : model, 
            "time" : arr[:, 0].mean().round(2),
            "f1_score": arr[:,1].mean().round(3),
            #"Precision" : arr[:,2].mean().round(3),
            #"Recall" : arr[:,2].mean().round(3)
        }
    df3 = pandas.DataFrame.from_dict(final, orient="index").round(3)
    return df3
filename1 = 'apps/data/filename.joblib'
filename2 = 'apps/data/donnee2.joblib'
filename3 = 'apps/data/donnee_Gen.joblib'
with open(filename1, 'rb') as f1:
    pickles = print_table1(pickle.load(f1))
with open(filename2, 'rb') as f2:
    pickles2 = print_table2(pickle.load(f2))
with open(filename3, 'rb') as f3:
    pickles3 = print_table3(pickle.load(f3))



# ##########################"fin "########"#################
layout = html.Div([
  nav_2,
  html.H1(id='test',children=['Resultat des Algorithmes']),
  html.Div([
      
      html.Div([
          html.Div([
            dcc.Dropdown(
            id='demo-dropdown',
            options=[
                {'label': 'Données Kaggle', 'value': 'NYC'},
                {'label': 'Données de data.world', 'value': 'MTL'},
                {'label': 'Global', 'value': 'DG'}
        ],
        value='DG', style={'width':'15vw','background-color':'#3F3680', 'color':'#7FDBFF','border-color':'#7FDBFF', 'cursor':' pointer'}
    ),

    html.Div(id='dd-output-container')
])
      ]),
     
       
  ])


])

# dash_table.DataTable(
#         id='joblib2',
#         columns=[{"name": i, "id": i} for i in pickles.columns],
#         data=pickles.to_dict('records'), 
#        editable=True, style_cell={"fontFamily": "Arial", "size": 2, 'textAlign': 'left', 'background-color':'#3F3680', 'color':'#7FDBFF'}),
@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    if value == 'NYC' :
        tableau = dash_table.DataTable(
        id='joblib',
        columns=[{"name": i, "id": i} for i in pickles.columns],
        data=pickles.to_dict('records'), 
       editable=True, style_cell={"fontFamily": "Arial", "size": 2, 'textAlign': 'left', 'background-color':'#3F3680', 'color':'#7FDBFF'}),
    elif value == 'MTL' :
        tableau = dash_table.DataTable(
        id='joblib2',
        columns=[{"name": i, "id": i} for i in pickles2.columns],
        data=pickles2.to_dict('records'), 
       editable=True, style_cell={"fontFamily": "Arial", "size": 2, 'textAlign': 'left', 'background-color':'#3F3680', 'color':'#7FDBFF'}),
    elif value == 'DG' :
        tableau = dash_table.DataTable(
        id='joblib3',
        columns=[{"name": i, "id": i} for i in pickles3.columns],
        data=pickles3.to_dict('records'), 
       editable=True, style_cell={"fontFamily": "Arial", "size": 2, 'textAlign': 'left', 'background-color':'#3F3680', 'color':'#7FDBFF'}),
    return tableau
