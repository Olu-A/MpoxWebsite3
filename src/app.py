import dash
import dash_bootstrap_components as dbc
#import dash_html_components as html
from dash import html
import requests
import pandas as pd
#import dash_core_components as dcc
from dash import dcc
import plotly.express as px
import numpy as np
from dash.dependencies import Input,Output
#import dash_table
from dash import dash_table
import datetime
from datetime import datetime, timedelta
from datetime import date
from pycountry_convert import country_name_to_country_alpha3
from dash import State
from textwrap import dedent
import plotly.graph_objects as go
#from apscheduler.schedulers.background import BackgroundScheduler
#from apscheduler.schedulers.blocking import BlockingScheduler
#from flask_apscheduler import APScheduler

#from Capstone import forecast

app = dash.Dash(__name__,external_stylesheets = [ dbc.themes.SOLAR], title='Mpox', update_title=None, meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])
server = app.server

#png
MPOX_IMG = "https://ichef.bbci.co.uk/news/976/cpsprodpb/183FD/production/_124852399_hi067948842-1.jpg"
colors = {'background': '#2D2D2D','text': '#E1E2E5','figure_text': '#ffffff','confirmed_text':'#3CA4FF','deaths_text':'#f44336','recovered_text':'#5A9E6F','highest_case_bg':'#393939',}
forecast = pd.read_csv('forecast.csv')
Cdata = pd.read_csv('Cdata.csv')
Fdata = forecast.set_index('ds')
"""
sched = APScheduler()

def my_job():
    id='my_job'
    print('Finished')
    import Capstone.py

#Probably set to 30 minutes
job = sched.add_job(func = my_job, trigger = 'interval', id='my_job', seconds = 10, replace_existing=True)
#job = sched.add_job(func = my_job, trigger = 'cron', id='my_job', hour=24, replace_existing=True, args=['job executed!!!!'])

sched.start()
"""
"""
while True:
        sleep(1)
"""
tday = date.today()
c0 = tday + timedelta(14)
C0 = c0.strftime("%Y-%m-%d")


Start = forecast['ds'].iloc[1]
End = forecast['ds'].iloc[-2]
if End > C0:
    End = C0

#TEMP ISE5 Stuff
c1 = tday - timedelta(1)
c2 = tday - timedelta(0)
c3 = tday + timedelta(1)
Y1 = c1.strftime("%Y-%m-%d")
T1 = c2.strftime("%Y-%m-%d")
N1 = c3.strftime("%Y-%m-%d")
#################################   Functions for creating Plotly graphs and data card contents ################
Y2= c1.strftime("%#m/%#d/%Y")
T2= c2.strftime("%#m/%#d/%Y")
N2= c3.strftime("%#m/%#d/%Y")

def get_continent(Cdata):
    try:
        a3code =  country_name_to_country_alpha3(Cdata)
    except:
        a3code = 'Unknown'
    return (a3code)

#Lines 83 to 91 need to be put into a callback
Cdata1 = pd.DataFrame()
Cdata1 ['date'] = Cdata ['date']
Cdata1 ['Countries'] = Cdata ['location']
Cdata1 ['Cases'] = Cdata ['new_cases_smoothed']
Cdata1 ['Codes'] = Cdata1 ['Countries'].apply(get_continent)
Cdata1['Country Code'] = Cdata1 ['Codes'].apply(lambda x: x[0] + x[1] + x[2])
Cdata1 = Cdata1.set_index('date')
Cdata1 = Cdata1[Cdata1['Country Code'] !="Unk"]
Cdata1.drop('Codes',axis = 1, inplace = True)
#Cdata1.to_csv('Cdata2.csv')
Sdata = Cdata1[Cdata1.index == T1]
Sdata2 = Cdata1.loc[Cdata1['Countries'] == "United States"]
"""
yesterday = round(Sdata2.loc[Y2].at['Cases'],2)
today = round(Sdata2.loc[T2].at['Cases'],2)
tommorow = round(Sdata2.loc[N2].at['Cases'],2)
"""
yesterday = round(Fdata.loc[Y1, 'yhat'],2)
today = round(Fdata.loc[T1].at['yhat'],2)
tommorow = round(Fdata.loc[N1].at['yhat'],2)

row_heights = [350]
template = {"layout": {"paper_bgcolor": "f3f3f1", "plot_bgcolor": "f3f3f1"}}

def blank_fig(height):
    """
    Build blank figure with the requested height
    """
    return {
        "data": [],
        "layout": {
            "height": height,
            "template": template,
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
        },
    }

def world_map(Sdata):
    fig1 = px.choropleth(Sdata, locations='Country Code', locationmode = 'ISO-3',color = 'Cases',
                        hover_data = ['Countries'],
                        projection="orthographic",
                        #color_continuous_scale=px.colors.sequential.Oranges,
                        color_continuous_scale=px.colors.sequential.BuPu,
                        range_color=(0, 5),
                        labels = {"Cases": "Reported Cases"})
    fig1.update_layout(coloraxis_colorbar_title_text = " # Reported Cases")
    fig1.update_layout(height=350, margin={"r":0,"t":0,"l":0,"b":0})
    fig1.update_geos(projection_scale = 0.90,)
    return(fig1)

date =Sdata2.index
Cases =Sdata2['Cases']

#Sdata2.to_csv('Sdata2.csv')
def trend_line(Sdata2):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x = date, y = Cases))
    #fig2.update_layout(title_text="Time series with range slider and selectors")
    fig2.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1month", step="month", stepmode="backward"),
                    dict(count=6, label="6months", step="month", stepmode="backward"),
                    #dict(count=1, label="1year", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),type="date"
        ),height=350
    )
    return(fig2)

#WORKING
def data_for_cases(header, total_cases):
    card_content = [
        dbc.CardHeader(header),

        dbc.CardBody(
            [dcc.Markdown( dangerously_allow_html = True,
                   children = ["{0} <br><sub>".format(total_cases)])])]

    return card_content

card_body1 = dbc.Col(dbc.Card(data_for_cases("Yesterday",f'{yesterday:,}' ' cases'), color="primary", id="card_data1", style = {'text-align':'center'}, inverse = True),
xs = 12, sm = 12, md = 4, lg = 4, xl = 4, style = {'padding':'12px 12px 12px 12px'})
card_body2 = dbc.Col(dbc.Card(data_for_cases("Today",f'{today:,}' ' cases'), color="secondary", id="card_data2", style = {'text-align':'center'}, inverse = True),
xs = 12, sm = 12, md = 4, lg = 4, xl = 4, style = {'padding':'12px 12px 12px 12px'})
card_body3 = dbc.Col(dbc.Card(data_for_cases("Tommorow",f'{tommorow:,}' ' cases'), color = 'warning', id="card_data3", style = {'text-align':'center'}, inverse = True),
xs = 12, sm = 12, md = 4, lg = 4, xl = 4, style = {'padding':'12px 12px 12px 12px'})

def build_modal_info_overlay(id, side, content):
    div = html.Div([
            html.Div([
                html.Div([html.H4(["Info",html.Img(id=f"close-{id}-modal",src="assets/times-circle-solid.svg",
                n_clicks=0,className="info-icon",style={"margin": 0},),],className="container_title",
                style={"color": "white"},),
                            dcc.Markdown(content),])],className=f"modal-content {side}",),
            html.Div(className="modal"),],id=f"{id}-modal",style={"display": "none"},)
    return div
############################################ body of the dashboard ###########################
body_app = dbc.Container( id = 'body_app', children = [

    #dbc.Row(html.Marquee("Mpox Model Last Updated: 2/14/22"), style = {'color':'green'}),

    html.Br(),

    dbc.Row(
    [
        dbc.Col(
            html.Div(
            [dcc.ConfirmDialog(id='Broken', message='This prototype website only supports The Country input United States'),
             dcc.Dropdown(id = 'country-dropdown',
                options = [{'label':i, 'value': i} for i in np.append(['All'],Cdata1 ['Countries'].unique()) ],
                value = 'United States',
                #disabled=True
                )]),
            style = {'width':'50%', 'color':'black', 'text-align':'center', 'display':'inline-block'},
            width={"size": 1, "offset": 0},
        ),
        dbc.Col(
            html.Div([
                dcc.DatePickerSingle(
                id='calender_dropdown',
                min_date_allowed = Start,
                max_date_allowed = End,
                date = tday,
                show_outside_days=True,
                day_size=32,
                display_format='MM/DD/YYYY',
                clearable=False,
            )]),
            style = {'width':'50%', 'color':'black', 'text-align':'left', 'display':'inline-block'})
    ]),
    dbc.Row(
        id = "card_row", children = [card_body1,card_body2,card_body3]
        ),


    html.Br(),

    dbc.Row([html.Div(html.H4('Global Impact of Mpox',className="banner1"),
                      style = {'color':'white','textAlign':'center','fontWeight':'bold','family':'georgia','width':'100%'})]),

    html.Br(),

    html.Div([
        html.Div(children =[build_modal_info_overlay("graph-info","bottom",dedent("""
        This interactive globe shows the # of reported cases in each country based on a color gradient
        scale. The globe can be moved to see the number of reported cases in each country. Countries
        with a gray color do not have a clear value of reported cases and are not included in the data
        that we pulled information from, hovering over most countries will give more information on the
        number of cases (Data on the globe only works for past dates)""")),
        build_modal_info_overlay("graphic-info","bottom",dedent("""
        This graph displays the amount of Mpox cases within a specified country from the first recorded cases
        up to our models predicted cases 2 weeks in the future. The tabs can be used to show the total number
        of cases in the time span selected including predicted cases. Also, the small graph underneath it can
        be used to manually observe the trends of Mpox over the recorded period.""")),
            html.Div(children=[html.Div(children=[
                html.H4(["Interactive Globe",html.Img(id="show-graph-info-modal",src="assets/question-circle-solid.svg",className="info-icon")],className="container_title2"),
                 dcc.Loading(dcc.Graph(id = 'world-graph', figure = world_map(Sdata),config={"displayModeBar": False}),style = {'height':'400px', 'width':"100%"})], id="graph-info-div",className="six columns pretty_container"),
                 #style = {'height':'400px', 'width':"625px"},
            html.Div(children=[
                html.H4(["Mpox Cases Trend",html.Img(id="show-graphic-info-modal",src="assets/question-circle-solid.svg",className="info-icon")],className="container_title2"),
                dcc.Loading(dcc.Graph(id = 'trend-graph', figure = trend_line(Sdata2),config={"displayModeBar": False}),style = {'height':'400px', 'width':"625px"})], id="graphic-info-div",className="six columns pretty_container"),
                #style = {'height':'400px', 'width':"625px"},
                #figure=blank_fig(row_heights[0])
            ]),
        ])
    ]),

#    html.Div([
#        html.Div(children =[build_modal_info_overlay("blank1-info","bottom",dedent(""Stuff1")),
#                            build_modal_info_overlay("blank2-info","bottom",dedent(""Stuff2")),
#                            build_modal_info_overlay("blank3-info","bottom",dedent(""Stuff3")),
#            html.Div(children=[html.Div(children=[
#                html.H4(["Blank 1",html.Img(id="show-Blank1-info-modal",src="assets/question-circle-solid.svg",className="info-icon")],className="container_title2"),
#                 dcc.Loading(dcc.Graph(id = 'blank1-graph', figure = blank_fig(row_heights[0]), config={"displayModeBar": False}),style = {'height':'400px', 'width':"100%"})], id="blank1-info-div",className="four columns pretty_container"),
#            html.Div(children=[
#                html.H4(["Blank 2",html.Img(id="show-Blank2-info-modal",src="assets/question-circle-solid.svg",className="info-icon")],className="container_title2"),
#                dcc.Loading(dcc.Graph(id = 'blank2-graph', figure =  blank_fig(row_heights[0]), config={"displayModeBar": False}),style = {'height':'400px', 'width':"625px"})], id="blank2-info-div",className="four columns pretty_container"),
#            html.Div(children=[
#                html.H4(["Blank 3",html.Img(id="show-Blank3-info-modal",src="assets/question-circle-solid.svg",className="info-icon")],className="container_title2"),
#                dcc.Loading(dcc.Graph(id = 'blank3-graph', figure =  blank_fig(row_heights[0]), config={"displayModeBar": False}),style = {'height':'400px', 'width':"625px"})], id="blank3-info-div",className="four columns pretty_container"),
#            ]),
#        ])
#    ]),


    html.Br(),

    html.Div([
                html.H4("Acknowledgements", style={"margin-top": "0"}),
                dcc.Markdown(
                    """\
 - Dashboard written in Python using the [Dash](https://dash.plot.ly/) web framework and [Flask] (https://flask.palletsprojects.com/en/2.2.x/) alongside CSS style sheets.
 - Parallel and distributed calculations implemented using the python based compiler [Datalore](https://datalore.jetbrains.com/).
 - Base graphic containers used from [World Cell Towers dashboard] (https://github.com/plotly/dash-world-cell-towers)
 - Mpox dataset provided by [?] (https://datalore.jetbrains.com/)
 - Icons provided by [Font Awesome](https://fontawesome.com/) and used under the
[_Font Awesome Free License_](https://fontawesome.com/license/free).

 _Created by Oluwaseun Adesina, Sarah Donovan, Kris Florentino, Bradley James, and Michael Krycun_

 __More information on the design of our website and model can be found on our [github] (https://github.com/Mpox-Predictor/Mpox-Code) page where you can also give feedback__
"""

#https://github.com/owid/monkeypox
                )],
                style={
                    'width': '100%',
                    'margin-right': '5',
                    'padding': '10px',
                    'horizontal-align': 'middle',
                },
                className="twelve columns pretty_container",
        ),

    ],fluid = True,)
############################## navigation bar ################################
content = """The purpose of this website is to offer the user an international prediction model for Mpox
cases. All data is accessed thorugh a combination of the country and date inputs. The website will output
on default the number of reported cases one day before your selected day, the number of cases on your
selected date, and the number of predicted cases one day after your selected date."""

startup = html.Div([html.Div([
        html.Div([html.H4(["Info",html.Img(id=f"close-basic-modal",src="assets/times-circle-solid.svg",
        n_clicks=0,className="info-icon",style={"margin": 0},),],className="container_title",
        style={"color": "white"},),
        dcc.Markdown(content),])],className=f"modal-content top",),
        html.Div(className="modal")],id=f"basic-modal",style={"display": "block"})
        #html.Div(className="modal"),],id=f"basic-modal",style={"display": "none"},)
        #])


#startup = html.Div(children = build_modal_info_overlay("basic","top",dedent("""Basic info"""))),

navbar = dbc.Navbar( id = 'navbar', children = [


    html.A(
    dbc.Row([
        dbc.Col(html.Img(src = MPOX_IMG, height = "70px")),
        dbc.Col(
            dbc.NavbarBrand("Mpox Live Tracker", style = {'color':'black', 'fontSize':'25px','fontFamily':'Impact'}
    ))],
    align = "center",),href = '/'
    ),

    dbc.Row([
        #dbc.Col(dbc.Button(id = 'button', children = "Github", color = "warning", className = 'ms-2', href = 'https://github.com/Mpox-Predictor/Mpox-Code')),

        html.Div(children =[build_modal_info_overlay("general","top",dedent(
        """The purpose of this website is to offer the user an international prediction model for Mpox
        cases. All data is accessed thorugh a combination of the country and date inputs. The website will output
        on default the number of reported cases one day before your selected day, the number of cases on your
        selected date, and the number of predicted cases one day after your selected date.""")),
        html.Div(html.H4(["Info",html.Img(id="show-general-modal",src="assets/question-circle-solid.svg",className="info-icon")],className="container_title"),
        id="general-div")]),
    ],className="g-0 ms-auto flex-nowrap mt-3 mt-md-0")
])

#interval =  html.Div([dcc.Interval(id='trigger', interval=3, n_intervals=0)])

app.layout = html.Div(id = 'parent', children = [navbar,body_app,startup])
#dcc.Interval(id="trigger", interval=86400),
#app.layout = dcc.Iframe(src='https://www.Mpox123.com', width='100%', height='500', Web)

#################################### Callback for adding interactivity to the dashboard #######################

@app.callback(
               [
                Output('card_data1','children'),
                Output('card_data2','children'),
                Output('card_data3','children'),
               ],
              Input(component_id = 'calender_dropdown', component_property = 'date'),
              Input(component_id = 'country-dropdown', component_property = 'value'),
              prevent_initial_call=True
              )

def update_cards1(value1, value2):
    try:
        date_object = datetime.fromisoformat(value1)
    except UnboundLocalError:
        date_object =  tday

    c1 = date_object - timedelta(1)
    c2 = date_object - timedelta(0)
    c3 = date_object + timedelta(1)
    card_value1 = c1.strftime(("%Y-%m-%d"))
    card_value2 = c2.strftime(("%Y-%m-%d"))
    card_value3 = c3.strftime(("%Y-%m-%d"))

#card_value3 = c3.strftime(("%#m/%#d/%Y"))
    Sdata0 = Cdata1.loc[Cdata1['Countries'] == value2]

    if value2 == "United States":
        card_value1 = c1.strftime(("%Y-%m-%d"))
        card_value2 = c2.strftime(("%Y-%m-%d"))
        card_value3 = c3.strftime(("%Y-%m-%d"))
        dayBefore =  round(Fdata.loc[card_value1].at['yhat'],2)
        thisDay =  round(Fdata.loc[card_value2].at['yhat'],2)
        dayAfter =  round(Fdata.loc[card_value3].at['yhat'],2)


    else:
        try:
            dayBefore =  round(Sdata0.loc[card_value1].at ['Cases'],2)
        except KeyError:
            dayBefore =  round(0,2)
        try:
            thisDay =  round(Sdata0.loc[card_value2].at ['Cases'],2)
        except KeyError:
            thisDay =  round(0,2)
        try:
            dayAfter =  round(Sdata0.loc[card_value3].at ['Cases'],2)
        except KeyError:
            dayAfter =  round(0,2)

    card_body1 = dbc.Card(data_for_cases(card_value1,f'{dayBefore:,}' ' cases'), color="primary", style = {'text-align':'center'}, inverse = True)
    card_body2 = dbc.Card(data_for_cases(card_value2,f'{thisDay:,}' ' cases'), color="secondary",style = {'text-align':'center'}, inverse = True)
    card_body3 = dbc.Card(data_for_cases(card_value3,f'{dayAfter:,}' ' cases'), color = 'warning',style = {'text-align':'center'}, inverse = True)

    return (card_body1, card_body2, card_body3)

@app.callback(
               Output('world-graph','figure'),
              Input(component_id = 'calender_dropdown', component_property = 'date'),
              prevent_initial_call=True
              )

def graph1(value1):
    date_object = datetime.fromisoformat(value1)
    c = date_object + timedelta(0)
    date = c.strftime("%Y-%m-%d")
    Sdata = Cdata1.loc[Cdata1.index == date]

    fig1 = px.choropleth(Sdata, locations='Country Code', locationmode = 'ISO-3',color = 'Cases',
                        hover_data = ['Countries'],
                        projection="orthographic",
                        #color_continuous_scale=px.colors.sequential.Oranges,
                        color_continuous_scale=px.colors.sequential.BuPu,
                        range_color=(0, 5),
                        labels = {"Cases": "Reported Cases"})
    fig1.update_layout(coloraxis_colorbar_title_text = " # Reported Cases")
    fig1.update_layout(height=350, margin={"r":0,"t":0,"l":0,"b":0})
    fig1.update_geos(projection_scale = 0.90,)
    return(fig1)


@app.callback(
               Output('trend-graph','figure'),
              Input(component_id = 'country-dropdown', component_property = 'value'),
              prevent_initial_call=True
              )
def graph2(value2):
    Sdata1 = Cdata1.loc[Cdata1['Countries'] == value2]

    date =Sdata1.index
    Cases =Sdata1['Cases']

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=date, y=Cases))
    #fig2.update_layout(title_text="Time series with range slider and selectors")
    fig2.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1month", step="month", stepmode="backward"),
                    dict(count=6, label="6months", step="month", stepmode="backward"),
                    #dict(count=1, label="1year", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),type="date"
        ),height=350
    )#go.Figure(data=trace)
    return(fig2)

@app.callback(Output('Broken_dropdown', 'displayed'),
              Input('country-dropdown', 'value'),
              prevent_initial_call=True)

def display_confirm(value3):
    if value3 != 'United States':
        return True
    return False

for id in ["general","graph-info","graphic-info","Blank1","Blank2","Blank3"]:

    @app.callback(
        [Output(f"{id}-modal", "style"), Output(f"{id}-div", "style")],
        [Input(f"show-{id}-modal", "n_clicks"), Input(f"close-{id}-modal", "n_clicks")],
    )
    def toggle_modal(n_show, n_close):
        ctx = dash.callback_context
        if ctx.triggered and ctx.triggered[0]["prop_id"].startswith("show-"):
            return {"display": "block"}, {"zIndex": 1003}
        else:
            return {"display": "none"}, {"zIndex": 0}

@app.callback(
    Output(f"basic-modal", "style"),
    Input(f"close-basic-modal", "n_clicks"),
    prevent_initial_call=True
)
def infoDump(clicks):
    if clicks is None:
        return {"display": "block"}
    else:
        return {"display": "none"}
    #raise PreventUpdate

#figure = world_map(Sdata))
#html.Img(src = app.get_asset_url('For1.png'), height = "350px", width = "605px")

if __name__ == "__main__":
    app.run_server()
