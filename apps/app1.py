import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer 
import plotly.graph_objs as go
from wordcloud import WordCloud
from time import time

import numpy as np

df = pd.read_csv('apps/data/Emotion_final_.csv')
dft_2 = df.iloc[:20,:]
table_donnees = go.Figure(data=[go.Table(
    header=dict(values=list(dft_2.columns),
                fill_color='#3F3680',
                align='left'),
    cells=dict(values=[dft_2.Emotion, dft_2.Text],
               fill_color='#3F3680',
               align='left'))
])


corpus = df['Text']
target = df['Emotion']
vec = CountVectorizer(stop_words="english")
X = vec.fit_transform(corpus)
words = vec.get_feature_names()

print("vocabulary size; %d" % len(words) )

# Compute rank
wsum = np.array(X.sum(0))[0]
ix = wsum.argsort()[::-1]
wrank = wsum[ix] 
labels = [words[i] for i in ix]

# Sub-sample the data to plot.
# take the 20 first + the rest sample with the given step 
def subsample(x, step=900):
    return np.hstack((x[:30], x[30:30:step]))
freq = subsample(wrank)
fig_histo = go.Figure()
fig_histo = go.Figure(data=[go.Histogram(x=df.Emotion, name='words count'), 
                      go.Histogram(x=df.Emotion, cumulative_enabled=True, name='cumulative words count')],
               layout ={
                   'title':'Emotions Histogram',
                   'xaxis_title_text': 'Emotions',
                   'yaxis_title_text': 'Count',
               })

fig_histo.update_layout(
    plot_bgcolor='#3F3680' ,
    paper_bgcolor='#3F3680',
    font_color='#7FDBFF',
    
    
)


x = subsample(labels)
y = freq
trace = go.Bar(
                x = x,
                y = y,
                name = "Words ordered by rank. The first rank is the most frequent words and the last one is the less present",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line = dict(color ='rgb(0,0,0)',width =1.5)))


layout = go.Layout(barmode = "group",plot_bgcolor='#3F3680' ,
    paper_bgcolor='#3F3680',
    font_color='#7FDBFF', )

fig_2 = go.Figure(data = trace, layout = layout)

# ############################## Fin ###################
# #########################Nav Bar #########################
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
  html.H2(id='test', children=["L'analyse et au traitement des données."]),
  
  html.Section(id="block_1", children=[
      html.Article(id="article_1_block_1",children=[
          html.Div(id='WordCloud', children=[
              dcc.Graph(
                  figure= fig_histo, id = 'table'
              )

          ])
      ]),
      html.Article(id="article_2_block_1",children=[
          dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} 
                 for i in dft_2.columns],
        data=dft_2.to_dict('records'),
        style_cell={'textAlign':'left', 'width':'10vw'},
        style_header=dict(backgroundColor="#000"),
        style_data=dict(backgroundColor="#3F3680"), 
        style_table={
            'height': '80vh', 'width': '56vw', 'overflowY': 'auto', 'color':'#7FDBFF'

        }
    )
      ])
  ]),
  html.Section(id="block_2", children=[
      
    dcc.Graph(
                  figure= fig_2, id="table_2"
              ) ,
  ])
   
])
