import dash_core_components as dcc
import dash_html_components as html

layout = html.Div(className="Erreur_G", children=[
    dcc.Link(style={"color":'#fff'},children=['Home'], href='/'),
    html.Div(className='error', children=[
        '404'
    ]),
    html.Br(),
    html.Br(),
    html.Span(className="info", children=[
        'File not found'
    ]),
    html.Img(src="http://images2.layoutsparks.com/1/160030/too-much-tv-static.gif", className="static")
])

