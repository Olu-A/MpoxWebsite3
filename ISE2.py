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
from datetime import date
from pycountry_convert import country_name_to_country_alpha3
#from Capstone import forecast

#app = dash.Dash(external_stylesheets = [ dbc.themes.FLATLY],)
app = dash.Dash(external_stylesheets = [ dbc.themes.SOLAR], title='Mpox', update_title=None)

#png
MPOX_IMG = "https://ichef.bbci.co.uk/news/976/cpsprodpb/183FD/production/_124852399_hi067948842-1.jpg"
Forcast_IMG = "https://user-images.githubusercontent.com/60261890/220716415-f797d5c1-747f-481c-abff-be3973b043f7.png"

#data = {["data2.csv"]}
df = pd.read_csv('pred_dates.csv')
df1 = df.T
df2 = df.set_index('Year').T
Sdata = df2.loc[:,['1/1/2022','1/1/2023','1/1/2024']]
last_year = Sdata.loc['United States'].at['1/1/2022']
this_year = Sdata.loc['United States'].at['1/1/2023']
next_year = Sdata.loc['United States'].at['1/1/2024']

#yesterday = Sdata.loc['yhat'].at['2/21/2022']
#today = Sdata.loc['yhat'].at['2/22/2023']
#tommorow = Sdata.loc['yhat'].at['2/23/2024']

#################################   Functions for creating Plotly graphs and data card contents ################
#NEED TO REINDEX EVERY COUNTRY FOR THIS TO WORK
""
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

#Ddata = df ['Year'].apply(get_date)

Sdata2 = {}
Sdata2 ['Countries'] = Sdata ['Countries']
Sdata2['current_year'] = Sdata['1/1/2023']
Sdata2['Country Code'] = Sdata['Country Code']
#print(Sdata2['Country Code'])

#Gdata = pd.DataFrame.from_dict(Sdata2)
#Gdata.to_csv('Gdata.csv')

def world_map(Sdata2):
    fig = px.choropleth(Sdata2, locations='Country Code', locationmode = 'ISO-3',color = 'current_year',
                        hover_data = ['Country Code'],
                        projection="orthographic",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        range_color=(0, 20))

    fig.update_layout(margin = dict(l=0,r=0,t=0,b=0))
    autosize = False
    fig.update_geos(projection_scale = 0.90,)
    return fig
""
#WORKING
def data_for_cases(header, total_cases):
    card_content = [
        dbc.CardHeader(header),

        dbc.CardBody(
            [
               dcc.Markdown( dangerously_allow_html = True,
                   children = ["{0} <br><sub>".format(total_cases)])


                ]

            )
        ]

    return card_content
############################################ body of the dashboard ###########################
#WORKING
body_app = dbc.Container([

    #dbc.Row(html.Marquee("Mpox Model Last Updated: 2/14/22"), style = {'color':'green'}),

    html.Br(),

    dbc.Row(
    [
        dbc.Col(
            html.Div(id = 'dropdown-div', children =
             [dcc.Dropdown(id = 'country-dropdown',
                options = [{'label':i, 'value':i} for i in np.append(['All'],Sdata2 ['Countries'].unique()) ],
                value = 'Country',
                placeholder = 'Select the country'
                )]),
            style = {'width':'50%', 'color':'black', 'text-align':'center', 'display':'inline-block'},
            width={"size": 1, "offset": 0},
        ),
        dbc.Col(
            dbc.CardGroup([html.P('Select Date',),
                            dcc.DatePickerSingle(
                                id='calender_dropdown',
                                min_date_allowed = date(2022, 5, 19),
                                max_date_allowed = date(2023, 3, 17),
                                date = date(2023, 1, 1),
                                show_outside_days=True,
                                day_size=32,
                                display_format='DD/MM/YYYY',
                                clearable=True
                            ),
                ]
            ),
            style = {'width':'50%', 'color':'black', 'text-align':'center', 'display':'inline-block'},
            width={"size": 4, "offset": 0}
        )
    ]
    ),
    dbc.Row([
        dbc.Col(dbc.Card(data_for_cases("last year",f'{last_year:,}'), color="success",style = {'text-align':'center'}, inverse = True),
        xs = 12, sm = 12, md = 4, lg = 4, xl = 4, style = {'padding':'12px 12px 12px 12px'}),
        dbc.Col(dbc.Card(data_for_cases("this year",f'{this_year:,}'), color="warning", id="card_data",style = {'text-align':'center'}, inverse = True),
        xs = 12, sm = 12, md = 4, lg = 4, xl = 4, style = {'padding':'12px 12px 12px 12px'}),
        dbc.Col(dbc.Card(data_for_cases("next year",f'{next_year:,}'), color = 'danger',style = {'text-align':'center'}, inverse = True),
        xs = 12, sm = 12, md = 4, lg = 4, xl = 4, style = {'padding':'12px 12px 12px 12px'})
        ]),


    html.Br(),

    dbc.Row([html.Div(html.H4('Global Impact of Mpox'),
                      style = {'textAlign':'center','fontWeight':'bold','family':'georgia','width':'100%'})]),

    html.Br(),
    html.Br(),

    dbc.Row([dbc.Col(dcc.Graph(id = 'world-graph', figure = world_map(Sdata2)),style = {'height':'400px'},xs = 12, sm = 12, md = 6, lg = 6, xl = 6),
    dbc.Col(html.Img(src = Forcast_IMG, height = "350px")),

#            dbc.FormGroup([html.P('Select Dates',
#            style={'textAlign': 'center'}),
#                    dcc.DatePickerRange(
#                        id='calender-dropdown',
#                        min_date_allowed = '01/01/2020',
#                        max_date_allowed = '01/01/2025',
#                        initial_visible_month = datetime.datetime(datetime.datetime.today().year, 1, 1).date(),
#                        # end_date=max_date,
#                        show_outside_days=True,
#                        day_size=32,
#                        display_format='DD/MM/YYYY',
#                        clearable=True
#                    ),
#                ]
#            )

        ])

    ],fluid = True)
############################## navigation bar ################################
forecast = pd.read_csv('forecast.csv')
navbar = dbc.Navbar( id = 'navbar', children = [


    html.A(
    dbc.Row([
        dbc.Col(html.Img(src = MPOX_IMG, height = "70px")),
        dbc.Col(
            dbc.NavbarBrand("Mpox Live Tracker", style = {'color':'black', 'fontSize':'25px','fontFamily':'Times New Roman'}
                            )

            )


        ],align = "center",
        # no_gutters = True
        ),
    href = '/'
    ),

                dbc.Row(
            [
        dbc.Col(
        # dbc.Button(id = 'button', children = "Click Me!", color = "primary"),
            dbc.Button(id = 'button', children = "Feedback", color = "warning", className = 'ml-auto', href = '/')

            )
    ],
            # add a top margin to make things look nice when the navbar
            # isn't expanded (mt-3) remove the margin on medium or
            # larger screens (mt-md-0) when the navbar is expanded.
            # keep button and search box on same row (flex-nowrap).
            # align everything on the right with left margin (ms-auto).
     className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
)
    # dbc.Button(id = 'button', children = "Support Us", color = "primary", className = 'ml-auto', href = '/')


    ])


app.layout = html.Div(id = 'parent', children = [navbar,body_app])

#################################### Callback for adding interactivity to the dashboard #######################

@app.callback([Output(component_id = 'card_data2', component_property ='children')],
              [Input(component_id = 'calender_dropdown', component_property = 'value')])


#forecast = pd.read_csv('forecast.csv')
#forecast = df.set_index('ds')
def update_output(value):
    card_value2 = forecast.loc['value'].at['yhat']
    return card_value2
        #Data = forecast.loc['date_value'].at['yhat']

#redundant = datetime.datetime.strptime("date_value", "%m/%d/%Y").strftime("%Y-%m-%d")
#Ddata = update_output()
#print(Ddata)

if __name__ == "__main__":
    app.run_server()
