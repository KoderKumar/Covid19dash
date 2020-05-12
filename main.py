import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from data import df1, df2, filter_df, resources, df_from_path, xth_infection_date, xth_date
import plotly.graph_objects as go
import plotly.express as px
import datetime
import numpy as np
import dash_ui as dui

external_stylesheets = [
    'bodystyle.css',
    'ui.css',
    'check.css',
    ]

CONFIRMED_COLOUR = 'rgba(102, 153, 255, 0.8)'
RECOVERED_COLOUR = 'rgba(52, 189, 45, 0.8)'
DEATHS_COLOUR = 'rgba(255, 102, 102, 0.8)'
FONT = "Courier New, monospace"

TOOLBAR_BUTTONS = [
  "zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
  "hoverClosestCartesian", "hoverCompareCartesian",
  "zoom3d", "pan3d", "resetCameraDefault3d", "resetCameraLastSave3d", "hoverClosest3d",
  "orbitRotation", "tableRotation",
  "zoomInGeo", "zoomOutGeo", "resetGeo", "hoverClosestGeo",
  "toImage",
  "sendDataToCloud",
  "hoverClosestGl2d",
  "hoverClosestPie",
  "toggleHover",
  "resetViews",
  "toggleSpikelines",
  "resetViewMapbox"
]

def add_buttons_to_default(buttons=["toImage", "resetViews", "resetViewMapbox", "resetGeo", "resetScale2d"], hide_buttons=TOOLBAR_BUTTONS):
    for button in buttons:
        hide_buttons.remove(button)
    return hide_buttons

CHOSEN_BUTTONS = add_buttons_to_default()

MINIMALIST_CONFIG ={
    "displaylogo": False,
    "modeBarButtonsToRemove": CHOSEN_BUTTONS,
}

headline_df = df_from_path(resources['worldwide-aggregated'])
current_date = headline_df.iloc[len(headline_df)-1][0]
current_confirmed = headline_df.iloc[len(headline_df)-1][1]
current_recovered = headline_df.iloc[len(headline_df)-1][2]
current_deaths = headline_df.iloc[len(headline_df)-1][3]
prev_date = headline_df.iloc[len(headline_df)-2][0]
prev_confirmed = headline_df.iloc[len(headline_df)-2][1]
prev_recovered = headline_df.iloc[len(headline_df)-2][2]
prev_deaths = headline_df.iloc[len(headline_df)-2][3]
confirmed_growth = (current_confirmed/prev_confirmed) - 1
recovered_growth = (current_recovered/prev_recovered) - 1
deaths_growth = (current_deaths/prev_deaths) - 1

def formatted_mvmt(figure, text):
    if figure >= 0:
        text += "(↑"
    else:
        text += "(↓"
    text += f" {figure:,.1%})"
    return text

confirmed_growth_text = formatted_mvmt(confirmed_growth, "")
recovered_growth_text = formatted_mvmt(recovered_growth, "")
deaths_growth_text = formatted_mvmt(deaths_growth, "")
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
grid = dui.Grid(_id="grid", num_rows=12, num_cols=12, grid_padding=0)

app.layout = html.Div(
    dui.Layout(
        grid=grid,
    ),
    style={
        'height': '100vh',
        'width': '100vw'
    }
)

app.config.suppress_callback_exceptions = True

def generate_map_w_options(df, plot_cases=True, plot_recoveries=True, plot_deaths=True):
    latest_date = df.iloc[len(df)-1][2]
    df = filter_df(df,"Date",latest_date)
    
    fig = go.Figure(go.Scattergeo())
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="rgb(199, 205, 214)",
        coastlinecolor = "rgb(199, 205, 214)",
        countrywidth=0.5,
        coastlinewidth=0.8,
        )
    
    titletext = "<b>Confirmed"
    if plot_cases:
        titletext += " cases"
    if plot_recoveries:
        titletext += " recoveries"
    if plot_deaths:
        titletext += " deaths"
    titletext += "</b>"
    
    fig.update_layout(
        title={
            'text': titletext,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            },
        height=500, 
        margin={"r":5,"t":10,"l":5,"b":0},
        showlegend=False,
        font=dict(
            family=FONT,
            size=12,
        ),
        )

    for location in range((len(df))):
        has_province = True if df.iloc[location][6] else False
        
        location_name = df.iloc[location][1]
        if has_province:
            location_name += f" ({df.iloc[location][6]})"

        if plot_cases:
            try:
                confirmed = 0 if np.isnan(df.iloc[location][0]) else df.iloc[location][0]
                if confirmed <= 0:
                    continue
                fig.add_trace(go.Scattergeo(
                    lon=[float(df.iloc[location][5])],
                    lat=[df.iloc[location][4]],
                    text=f"{location_name}: {confirmed:,.0f} confirmed",
                    name=location_name + " - confirmed",
                    marker=dict(
                        size=int(confirmed**(0.5))/10,
                        color=CONFIRMED_COLOUR,
                        line_color='rgba(0,0,0,0.35)',
                        line_width=0.5,
                    )
                ))
            except:
                pass

        if plot_deaths:
            try:
                deaths = 0 if np.isnan(df.iloc[location][3]) else df.iloc[location][3]
                if deaths <= 0:
                    continue
                fig.add_trace(go.Scattergeo(
                    lon=[float(df.iloc[location][5])],
                    lat=[df.iloc[location][4]],
                    text=f"{location_name}: {deaths:,.0f} dead",
                    name=location_name + " - deaths",
                    marker=dict(
                        size=int(deaths**(0.5))/10,
                        color=DEATHS_COLOUR,
                        line_color='rgba(0,0,0,0.35)',
                        line_width=0.5,
                    )
                ))
            except:
                pass

        if plot_recoveries:
            try:
                recoveries = 0 if np.isnan(df.iloc[location][7]) else df.iloc[location][7]
                if recoveries <= 0:
                    continue
                fig.add_trace(go.Scattergeo(
                    lon=[float(df.iloc[location][5])],
                    lat=[df.iloc[location][4]],
                    text=f"{location_name}: {recoveries:,.0f} recovered",
                    name=location_name + " - recoveries",
                    marker=dict(
                        size=int(recoveries**(0.5))/10,
                        color=RECOVERED_COLOUR,
                        line_color='rgba(0,0,0,0.35)',
                        line_width=0.5,
                    )
                ))
            except:
                pass

    return fig

def generate_deathrates_by_country(resources=resources, max_rows=30, min_cases=1000, min_deaths=200, date=False):
    df = df_from_path(resources['countries-aggregated'])
    
    if not date:
        date = df.iloc[len(df)-1,0] 
    df = filter_df(df, "Date", date)
    
    x_data = []
    y_data = []
    y_label = []

    for location in range((len(df))):            
        location_name = df.iloc[location][1]
        if len(location_name.split(" ")) > 1 and len(location_name) > 13:
            location_abbv = ""
            for word in location_name.split(" "):
                location_abbv += word[0]
            location_name = location_abbv

        cases = df.iloc[location][2]
        if cases == 0:
            cases+=1
        deaths = df.iloc[location][4]

        if cases > min_cases and deaths > min_deaths:
            death_rate = deaths/cases
            x_data.append(death_rate)
            y_data.append(location_name)
            y_label.append(f"{death_rate*100:.1f}% ({deaths:,}/{cases:,})")

    num_rows = min(max_rows, len(x_data))

    colours = []
    for row in range(num_rows):
        colours.append(f'rgb(255,{230-128*(row/num_rows)},{230-128*(row/num_rows)})')

    total_cases = df.sum(axis=0)[2]
    total_deaths = df.sum(axis=0)[4]
    average_death_rate = total_deaths / total_cases
    x_data.append(average_death_rate)
    y_data.append("Average")
    y_label.append(f"{average_death_rate*100:.1f}% ({total_deaths:,}/{total_cases:,})")

    x_data, y_data, y_label = (list(t) for t in zip(*sorted(zip(x_data, y_data, y_label))))

    x_data = x_data[len(x_data)-max_rows:len(x_data)]
    y_data = y_data[len(y_data)-max_rows:len(y_data)]
    y_label = y_label[len(y_label)-max_rows:len(y_label)]

    try:
        colours[y_data.index('Average')] = 'rgba(168, 102, 255, 0.8)'
    except:
        print(f"Average not in top {num_rows}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=y_data[::-1],
        y=x_data[::-1],
        text=y_label[::-1],
        name="Death rate summary",
        orientation='v',
        marker=dict(
            color=colours[::-1],
            line=dict(
                color='rgba(38, 24, 74, 0.8)',
                width=1)
        )
    ))

    fig.update_yaxes(range=[0, 0.2])

    fig.update_layout(
        title={
            'text': f"<b>Death rates ({min_cases}+ cases, {min_deaths}+ deaths)</b>",
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
        },
        barmode='stack',
        margin={"r":15,"t":40,"l":10,"b":10, "pad":5},
        width=1*num_rows,
        yaxis=dict(
            tickformat=".1%",
        ),
        xaxis_tickangle=-90,
        font=dict(
            family=FONT,
            size=12,
        )
        )

    fig.update_yaxes(tickfont=dict(size=12),)
    return fig

def generate_world_ts_options(resources=resources, plot_confirmed=True, plot_recovered=True, plot_deaths=True):
    df = df_from_path(resources['worldwide-aggregated']) 
    date = df['Date'].tolist()
    confirmed = df['Confirmed'].tolist()
    recovered = df['Recovered'].tolist()
    deaths = df['Deaths'].tolist()

    fig = go.Figure()

    if plot_confirmed:
        fig.add_trace(go.Scatter(x=date, y=confirmed, name='Cases',
                                line=dict(color=CONFIRMED_COLOUR, width=2)))
    if plot_recovered:
        fig.add_trace(go.Scatter(x=date, y=recovered, name = 'Recoveries',
                                line=dict(color=RECOVERED_COLOUR, width=2)))
    if plot_deaths:
        fig.add_trace(go.Scatter(x=date, y=deaths, name='Deaths',
                                line=dict(color=DEATHS_COLOUR, width=2)))

    fig.update_layout(
        title={
            'text': f"<b>Global confirmed numbers over time</b>",
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
        },
        margin={"r":10,"t":40,"l":10,"b":5},
        font=dict(
            family=FONT,
            size=12,
        ),
        xaxis_title='Date',
        yaxis_title='Confirmed numbers',
        hovermode='x')

    fig.update_yaxes(nticks=10)
    fig.update_xaxes(nticks=10)

    return fig

ts_df = df_from_path(resources['countries-aggregated'])

@app.callback(
    Output('comp-output', 'figure'),
    [Input(component_id='plot', component_property='value')]
)
def generate_comparable_time_series(
        plot="Confirmed",
        countries=["China", "United Kingdom", "Italy", "Spain", "Iran", "US", "Korea, South"],
        df=ts_df,
        xth=100
        ):
    
    countries=["China", "United Kingdom", "Italy", "Spain", "Iran", "US", "Korea, South"]
    df=df_from_path(resources['countries-aggregated'])
    xth=100 

    fig = go.Figure()

    if plot == "Confirmed":
        plot_word = "case"
        plural_plot_word = "cases"
    elif plot == "Recovered":
        plot_word = "recovery"
        plural_plot_word = "recoveries"
    elif plot == "Deaths":
        plot_word = "death"
        plural_plot_word = "deaths"
    else:
        raise ValueError(f"'plot' variable must be equal to 'Confirmed', 'Recovered' or 'Deaths'. Your input was '{plot}'")

    for country in countries:

        base_date = xth_date(country, xth, data=plural_plot_word, df=df)
        country_df = df.loc[df["Country"] == country]
        try:
            country_df = country_df[country_df['Date'] >= base_date]
        except:
            continue

        x_axis_data = []
        y_axis_data = country_df[plot].tolist()

        for i in range(len(y_axis_data)):
            x_axis_data.append(i)

        fig.add_trace(go.Scatter(x=x_axis_data, y=y_axis_data, name=country, mode='lines'))

    fig.update_layout(
        title={
            'text': f"<b>Confirmed {plural_plot_word} over time (day 0 = {xth:,} {plural_plot_word})</b>",
            'x': 0.5,
            'xanchor': 'center',
        },
        margin={"r":10,"t":30,"l":10,"b":0},
        font=dict(
            family=FONT,
            size=12,
        ),
        xaxis_title=f'Days since {xth}th {plot_word}',
        yaxis_title=f'Confirmed {plural_plot_word}',
        hovermode='x')

    fig.update_yaxes(nticks=10)
    fig.update_xaxes(nticks=10)

    return fig

def generate_datatable(df=df_from_path(resources['countries-aggregated']),date=False):
    if not date:
        date = df.iloc[len(df)-1,0] 
    df = filter_df(df, "Date", date)

    df = df[['Country', 'Confirmed', 'Recovered', 'Deaths']]

    for i in range(len(df)):
        if len(df.iloc[i][0]) > 15:
            df.at[df.iloc[i].name, 'Country'] = f"{df.iloc[i][0][:12]}..."
    
    return df

df_datatable = generate_datatable()

grid.add_element(col=1, row=1, width=4, height=4, element=dcc.Graph(
    id='World map of confirmed cases',
    config=MINIMALIST_CONFIG,
    figure=generate_map_w_options(df2, plot_recoveries=False, plot_deaths=False),
    style={"height": "100%", "width": "100%"}
))

grid.add_element(col=5, row=1, width=4, height=4, element=dcc.Graph(
    id='World map of confirmed recoveries',
    config=MINIMALIST_CONFIG,
    figure=generate_map_w_options(df2, plot_cases=False, plot_deaths=False),
    style={"height": "100%", "width": "100%"}
))

grid.add_element(col=9, row=1, width=4, height=4, element=dcc.Graph(
    id='World map of confirmed deaths',
    config=MINIMALIST_CONFIG,
    figure=generate_map_w_options(df2, plot_cases=False, plot_recoveries=False),
    style={"height": "100%", "width": "100%"}
))


grid.add_element(col=1, row=5, width=3, height=4, element=dash_table.DataTable(
    id="Table",
    columns=[{"name": i, "id": i} for i in df_datatable.columns],
    data=df_datatable.to_dict('records'),
    sort_action="native",
    sort_by=[{"column_id": "Confirmed", "direction": "desc"}],
))



grid.add_element(col=4, row=5, width=3, height=4, element=html.Div(
    [
        html.H4(
            ["Covid-19 dashboard"],
            style={"font-weight": "bold"}
        ),
        html.P(
            ["Worldwide headline figures:"],
            style={"font-weight": "bold"}
        ),
        html.Div([
            html.Div([
                html.H5(
                    [f"Cases: {current_confirmed:,}"],
                    style={"color": CONFIRMED_COLOUR, "font-weight": "bold", "display": "inline"}
                ),
                html.P(
                    [f" {confirmed_growth_text}"],
                    style={"font-size": "1.2rem", "display": "inline"}
                )
            ]),
            html.Div([
                html.H5(
                    [f"Recoveries: {current_recovered:,}"],
                    style={"color": RECOVERED_COLOUR, "font-weight": "bold", "display": "inline"}
                ),
                html.P(
                    [f" {recovered_growth_text}"],
                    style={"font-size": "1.2rem", "display": "inline"}
                )
            ]),
            html.Div([
                html.H5(
                    [f"Deaths: {current_deaths:,}"],
                    style={"color": DEATHS_COLOUR, "font-weight": "bold", "display": "inline"}
                ),
                html.P(
                    [f" {deaths_growth_text}"],
                    style={"font-size": "1.2rem", "display": "inline"}
                )
            ]),
            html.P(
                [f"Data accurate as at {current_date}"],
                style={"margin-top": "0.75em"}
            ),
            html.P(
                ["Created by ArthK"]
            ),

        ]),
    ],
    style={
        "font-family": FONT, 
        "text-align":"center", 
        "background-color": "white", 
        "height": "100%",
        "display": "flow-root"},
))


grid.add_element(col=7, row=5, width=6, height=4, element=html.Div([
    dcc.RadioItems(
        id="plot", 
        options=[
            {'label': "Cases", 'value': "Confirmed"},
            {'label': "Recoveries", 'value': "Recovered"},
            {'label': "Deaths", 'value': "Deaths"},
        ],
        value="Confirmed",
        labelStyle={
            "display": "inline-block",
            },
        style={
            "height": "10%",  
            "font-family": FONT, 
            "text-align":"center", 
            "background-color": "white",
        }),
    dcc.Graph(
        id="comp-output",
        config=MINIMALIST_CONFIG,
        style={"height": "90%", "width": "100%"}
        )
    ],
    style={"height": "100%", "width": "100%"},
))


grid.add_element(col=1, row=9, width=7, height=4, element=dcc.Graph(
    id="Overall time series",
    config=MINIMALIST_CONFIG,
    figure=generate_world_ts_options(),
    style={"height": "100%", "width": "100%"}
))

grid.add_element(col=8, row=9, width=5, height=4, element=dcc.Graph(
    id="Death rates",
    config=MINIMALIST_CONFIG,
    figure=generate_deathrates_by_country(max_rows=21),
    style={"height": "100%", "width": "100%",}
))


if __name__ == '__main__':
    app.run_server(debug=True)