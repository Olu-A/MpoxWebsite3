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
#from Capstone import forecast

#app = dash.Dash(external_stylesheets = [ dbc.themes.FLATLY],)
app = dash.Dash(__name__,external_stylesheets = [ dbc.themes.SOLAR], title='Mpox', update_title=None)
server = app.server

#png
MPOX_IMG = "https://ichef.bbci.co.uk/news/976/cpsprodpb/183FD/production/_124852399_hi067948842-1.jpg"
Forcast_IMG = "https://user-images.githubusercontent.com/60261890/220716415-f797d5c1-747f-481c-abff-be3973b043f7.png"

colors = {'background': '#2D2D2D','text': '#E1E2E5','figure_text': '#ffffff','confirmed_text':'#3CA4FF','deaths_text':'#f44336','recovered_text':'#5A9E6F','highest_case_bg':'#393939',}
forecast = pd.read_csv('forecast.csv')
#if forecast[0] = "line number";
#    del forecast[forecast.columns[0]]
Fdata = forecast.set_index('ds')
Rdata0 = pd.read_csv('Rdata.csv')
Rdata = Rdata0.set_index('date')
Gdata = pd.read_csv('Gdata.csv')

Start = forecast['ds'].iloc[1]
End = forecast['ds'].iloc[-2]

df = pd.read_csv('pred_dates.csv')
df1 = df.T
df2 = df.set_index('Year').T
Sdata = df2.loc[:,['1/1/2022','1/1/2023','1/1/2024']]

tday = date.today()
#T1 = tday.strftime("%#m/%#d/%y")
T1 = tday.strftime("%Y-%m-%d")
c1 = tday - timedelta(1)
c3 = tday + timedelta(1)
#Y1 = c1.strftime("%#m/%#d/%y")
Y1 = c1.strftime("%Y-%m-%d")
#N1 = c3.strftime("%#m/%#d/%Y")
N1 = c3.strftime("%Y-%m-%d")

#Fdata1 = pd.DataFrame.from_dict(Fdata)
#Fdata1.to_csv('forecast2.csv')

yesterday = round(Fdata.loc[Y1, 'yhat'],2)
today = round(Fdata.loc[T1].at['yhat'],2)
tommorow = round(Fdata.loc[N1].at['yhat'],2)
#################################   Functions for creating Plotly graphs and data card contents ################

def get_continent(Sdata):
    try:
        a3code =  country_name_to_country_alpha3(Sdata)
    except:
        a3code = 'Unknown'
    return (a3code)

Sdata ['Countries'] = Sdata.index
Sdata ['Codes'] = Sdata ['Countries'].apply(get_continent)
Sdata['Country Code'] = Sdata ['Codes'].apply(lambda x: x[0] + x[1] + x[2])
Sdata.drop('Codes',axis = 1, inplace = True)

Sdata2 = {}
Sdata2 ['Countries'] = Sdata ['Countries']
Sdata2['current_year'] = Sdata['1/1/2023']
Sdata2['Country Code'] = Sdata['Country Code']

def world_map(Sdata2):
    fig = px.choropleth(Sdata2, locations='Country Code', locationmode = 'ISO-3',color = 'current_year',
                        hover_data = ['Countries'],
                        projection="orthographic",
                        #color_continuous_scale=px.colors.sequential.Oranges,
                        color_continuous_scale=px.colors.sequential.BuPu,
                        range_color=(0, 20),
                        labels = {"Cases": "Reported Cases"},)

    fig.update_layout(title_text = "Global cases", coloraxis_colorbar_title_text = " # Reported Cases", margin = dict(l=0,r=0,t=0,b=0),
    geo=dict(bgcolor = "rgb(255,255,255)"),
    autosize = False,
    width =600,
    height=350)
    fig.update_geos(projection_scale = 0.90,)
    return fig

#WORKING
def data_for_cases(header, total_cases):
    card_content = [
        dbc.CardHeader(header),

        dbc.CardBody(
            [dcc.Markdown( dangerously_allow_html = True,
                   children = ["{0} <br><sub>".format(total_cases)])])]

    return card_content

card_body1 = dbc.Col(dbc.Card(data_for_cases("Yesterday",f'{yesterday:,}'), color="primary", id="card_data1", style = {'text-align':'center'}, inverse = True),
xs = 12, sm = 12, md = 4, lg = 4, xl = 4, style = {'padding':'12px 12px 12px 12px'})
card_body2 = dbc.Col(dbc.Card(data_for_cases("Today",f'{today:,}'), color="secondary", id="card_data2", style = {'text-align':'center'}, inverse = True),
xs = 12, sm = 12, md = 4, lg = 4, xl = 4, style = {'padding':'12px 12px 12px 12px'})
card_body3 = dbc.Col(dbc.Card(data_for_cases("Tommorow",f'{tommorow:,}'), color = 'warning', id="card_data3", style = {'text-align':'center'}, inverse = True),
xs = 12, sm = 12, md = 4, lg = 4, xl = 4, style = {'padding':'12px 12px 12px 12px'})

def build_modal_info_overlay(id, side, content):
    """
    Build div representing the info overlay for a plot panel
    """
    div = html.Div([  # modal div
            html.Div([  # content div
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
    #[dbc.Alert([html.I(id='Broken_dropdown', className="bi bi-info-circle-fill me-2"),"An example info alert with an icon",],is_open=False, color="info",className="d-flex align-items-center",)],
    dbc.Row(
    [
        dbc.Col(
            html.Div(
            #[dbc.Alert([html.I(id='Broken_dropdown', className="bi bi-info-circle-fill me-2"),"An example info alert with an icon",],is_open=False, color="info",className="d-flex align-items-center",),
            [dcc.ConfirmDialog(id='Broken_dropdown', message='This version of the website only supports The Country input United States'),
             dcc.Dropdown(id = 'country-dropdown',
                options = [{'label':i, 'value': i} for i in np.append(['All'],Gdata ['Countries'].unique()) ],
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

    dbc.Row([html.Div(html.H4('Global Impact of Mpox',className="container_title"),
                      style = {'color':'white','textAlign':'center','fontWeight':'bold','family':'georgia','width':'100%'})]),

    html.Br(),

    dbc.Row([
        html.Div(children =[build_modal_info_overlay("graph-info","bottom",dedent("""
        This interactive globe shows the # of reported cases in each country based on a color gradient
        scale. Feel free to move the globe to see the number of reported cases in each country. Countries
        with a gray color do not have a clear value of reported cases and are not included in the data
        that we pulled information from""")),
            build_modal_info_overlay("graphic-info","bottom",dedent("""
            The following is a trendline for data based on other countries for the next 14 days""")),
            html.Div(dbc.Col(children=[
                html.H4(["Reported Cases by year",html.Img(id="show-graph-info-modal",src="assets/question-circle-solid.svg",className="info-icon")],className="container_title"),
                dcc.Graph(id = 'world-graph', figure = world_map(Sdata2))],style = {'height':'400px', 'width':"625px", "margin-right": "4",},xs = 12, sm = 12, md = 6, lg = 6, xl = 6,className="six columns pretty_container"), id="graph-info-div",className='container'),
                #Verticalbreak
            html.Div(dbc.Col(children=[
                html.H4(["Prediction model trend",html.Img(id="show-graphic-info-modal",src="assets/question-circle-solid.svg",className="info-icon")],className="container_title"),
                html.Img(src = Forcast_IMG, height = "350px", width = "605px")],style = {'height':'400px', 'width':"625px", "margin-left": "4",},xs = 12, sm = 12, md = 6, lg = 6, xl = 6,className="six columns pretty_container"), id="graphic-info-div",className='container'),
        ],),
    ]),


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
navbar = dbc.Navbar( id = 'navbar', children = [


    html.A(
    dbc.Row([
        dbc.Col(html.Img(src = MPOX_IMG, height = "70px")),
        dbc.Col(
            dbc.NavbarBrand("Mpox Live Tracker", style = {'color':'black', 'fontSize':'25px','fontFamily':'Times New Roman'}
    ))],
    align = "center",),href = '/'
    ),

    dbc.Row([
        #dbc.Col(dbc.Button(id = 'button', children = "Github", color = "warning", className = 'ms-2', href = 'https://github.com/Mpox-Predictor/Mpox-Code')),

        html.Div(children =[build_modal_info_overlay("general","top",dedent("""
        The purpose of this website is to offer the user an international prediction model for Mpox
        cases. Please select a country and date as the two major inputs. The website will then output
        the number of past historical cases one day before your selected day, the # of cases on your
        selected date, and the # of future predicted cases one day after your selected date.""")),
        html.Div(html.H4(["Info",html.Img(id="show-general-modal",src="assets/question-circle-solid.svg",className="info-icon")],className="container_title"),
        id="general-div")]),
    ],className="g-0 ms-auto flex-nowrap mt-3 mt-md-0")
])

# Define the HTML structure of your index.html file
#index_html = html.Div(id = 'parent', children = [navbar,body_app])

# Write the HTML to a file
#with open('index.html', 'w') as f:
#    f.write(index_html.to_html())




app.layout = html.Div(id = 'parent', children = [navbar,body_app])
#app.layout = dcc.Iframe(src='https://www.Mpox123.com', width='100%', height='500', Web)

#################################### Callback for adding interactivity to the dashboard #######################

@app.callback(
               [
                Output('card_data1','children'),
                Output('card_data2','children'),
                Output('card_data3','children')
               ],
              Input(component_id = 'calender_dropdown', component_property = 'date'),
              prevent_initial_call=True
              )

def update_cards(value):
    date_object = date.fromisoformat(value)
    #card_value2 = date_object.strftime("%#m/%#d/%Y")
    card_value2 = date_object.strftime(("%Y-%m-%d"))
    c1 = date_object - timedelta(1)
    c3 = date_object + timedelta(1)
    #card_value1 = c1.strftime("%#m/%#d/%Y")
    card_value1 = c1.strftime(("%Y-%m-%d"))
    #card_value3 = c3.strftime("%#m/%#d/%Y")
    card_value3 = c3.strftime(("%Y-%m-%d"))

    dayBefore =  round(Fdata.loc[card_value1].at['yhat'],2)
    thisDay =  round(Fdata.loc[card_value2].at['yhat'],2)
    dayAfter =  round(Fdata.loc[card_value3].at['yhat'],2)

    card_body1 = dbc.Card(data_for_cases(card_value1,f'{dayBefore:,}'), color="primary", style = {'text-align':'center'}, inverse = True)
    card_body2 = dbc.Card(data_for_cases(card_value2,f'{thisDay:,}'), color="secondary",style = {'text-align':'center'}, inverse = True)
    card_body3 = dbc.Card(data_for_cases(card_value3,f'{dayAfter:,}'), color = 'warning',style = {'text-align':'center'}, inverse = True)

    return (card_body1, card_body2, card_body3)

@app.callback(Output('Broken_dropdown', 'displayed'),
              Input('country-dropdown', 'value'),
              prevent_initial_call=True)

def display_confirm(value3):
    if value3 != 'United States':
        return True
    return False

for id in ["general","graph-info","graphic-info"]:

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


if __name__ == "__main__":
    app.run_server()
