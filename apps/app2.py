import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer 
import plotly.graph_objs as go
from wordcloud import WordCloud
from time import time

import numpy as np

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
# ##########################"fin "########"#################
layout = html.Div([
  nav_2,
  html.H1(id='test',children=['Resultat des Algorithmes'])   
])


