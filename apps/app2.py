import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer 
import plotly.graph_objs as go
from wordcloud import WordCloud
from time import time
import pickle
import numpy as np
from collections import defaultdict

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

filename1 = 'apps/data/filename.joblib'
with open(filename1, 'rb') as f1:
    pickles = print_table1(pickle.load(f1))



# ##########################"fin "########"#################
layout = html.Div([
  nav_2,
  html.H1(id='test',children=['Resultat des Algorithmes']),
  html.Div([
     dash_table.DataTable(
        id='joblib',
        columns=[{"name": i, "id": i} for i in pickles.columns],
        data=pickles.to_dict('records'), 
       editable=True),
        # style_cell={"fontFamily": "Arial", "size": 10, 'textAlign': 'left'}
  ])


])

