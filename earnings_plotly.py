import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --------------------------------------------------------------------------------
# GLOBAL SETTINGS & CONFIG
# --------------------------------------------------------------------------------

index_options = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC"}

BG_COLOR = '#f8f9fa'
FONT_COLOR = '#2c3e50'
ACCENT_COLOR = '#3498db'
DANGER_COLOR = '#e74c3c'
CONTROL_PANEL_STYLE = {
    'padding': '20px', 
    'borderRadius': '10px', 
    'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 
    'backgroundColor': 'white'
}
SLIDERS_CONTAINER_STYLE = {'overflowY': 'auto', 'maxHeight': '50vh', 'marginBottom': '20px'}

initial_year = 2021
initial_index = "S&P 500"
initial_values = {
    "Pre 2 Weeks": -50, "Pre 1 Week": -50, "Pre 3 Days": -50, "Pre 2 Days": -50, "Pre 1 Day": -50,
    "Earnings Move": -50, "Opening Gap": 10, "Open-to-High": -50, "Open-to-Low": -50,
    "Drift Open-to-Close": -50, "Post 1 Day": -50, "Post 2 Days": -50, "Post 3 Days": -50,
    "Post 1 Week": -50, "Post 2 Weeks": -50
}
selected_date = None

# --------------------------------------------------------------------------------
# FETCH EARNINGS DATA FROM GOOGLE SHEETS
# --------------------------------------------------------------------------------

# Define the scope and credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(
    r"", scope
)
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open("Earnings Analysis").worksheet("Sheet1")

# Fetch all data and convert to DataFrame
earnings_data = pd.DataFrame(sheet.get_all_records())

# List of columns that should be numeric (based on initial_values keys)
numeric_columns = [
    "Pre 2 Weeks", "Pre 1 Week", "Pre 3 Days", "Pre 2 Days", "Pre 1 Day",
    "Earnings Move", "Opening Gap", "Open-to-High", "Open-to-Low",
    "Drift Open-to-Close", "Post 1 Day", "Post 2 Days", "Post 3 Days",
    "Post 1 Week", "Post 2 Weeks"
]

# Convert 'Adjusted Trading Date' to datetime and numeric columns to float
earnings_data["Adjusted Trading Date"] = pd.to_datetime(earnings_data["Adjusted Trading Date"])
for col in numeric_columns:
    if col in earnings_data.columns:
        earnings_data[col] = pd.to_numeric(earnings_data[col], errors='coerce')

print(
    f"Earnings data loaded: {earnings_data.shape[0]} rows\n"
    f"from {earnings_data['Adjusted Trading Date'].min()} to {earnings_data['Adjusted Trading Date'].max()}"
)
print(f"Earnings data sample:\n{earnings_data.head()}")

years = [y for y in sorted(earnings_data["Adjusted Trading Date"].dt.year.unique()) if y >= 2021]

# --------------------------------------------------------------------------------
# HELPER: FETCH & PREP INDEX DATA
# --------------------------------------------------------------------------------

def fetch_index_data(ticker, start_date="2000-01-01", end_date="2025-12-31"):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
    close_cols = [c for c in data.columns if c.startswith("Close")]
    if not close_cols:
        print("No 'Close' column found in downloaded data. Returning empty DataFrame.")
        return pd.DataFrame(columns=["Date", "Close"])
    main_close_col = close_cols[0]
    data = data.reset_index().rename(columns={main_close_col: "Close"})[["Date", "Close"]]
    data = data.sort_values("Date").reset_index(drop=True)
    print(f"Fetched {ticker} data: {data.shape[0]} rows from {data['Date'].min()} to {data['Date'].max()}")
    print(f"First few rows:\n{data.head()}")
    return data

def get_index_close(date, index_data):
    if date in index_data.index:
        return index_data.loc[date, "Close"]
    else:
        idx = index_data.index.searchsorted(date) - 1
        if idx < 0:
            idx = 0
        return index_data.iloc[idx]["Close"]

# --------------------------------------------------------------------------------
# PROCESS DATA: FILTER EARNINGS + MAP INDEX
# --------------------------------------------------------------------------------

def process_data(year, index_type, **filters):
    start_date = pd.to_datetime(f"01/01/{int(year)}")
    end_date = pd.to_datetime(f"12/31/{int(year)}")
    print(f"\nProcessing data for {index_type}, year {year}, start_date: {start_date}, end_date: {end_date}\n")

    index_ticker = index_options[index_type]
    raw_index_data = fetch_index_data(index_ticker, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
    if raw_index_data.empty:
        print("Warning: No index data returned for that range.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw_index_data["MA10"] = raw_index_data["Close"].rolling(10).mean()
    raw_index_data["MA20"] = raw_index_data["Close"].rolling(20).mean()
    raw_index_data["MA50"] = raw_index_data["Close"].rolling(50).mean()

    index_filtered = raw_index_data[(raw_index_data["Date"] >= start_date) & (raw_index_data["Date"] <= end_date)].copy()
    index_filtered = index_filtered.sort_values("Date")
    print(f"index_filtered rows: {index_filtered.shape[0]} from {index_filtered['Date'].min()} to {index_filtered['Date'].max()}")

    e_filtered = earnings_data[
        (earnings_data["Adjusted Trading Date"] >= start_date) &
        (earnings_data["Adjusted Trading Date"] <= end_date)
    ].copy()
    print(f"e_filtered after date filter: {e_filtered.shape[0]} rows from {e_filtered['Adjusted Trading Date'].min()} to {e_filtered['Adjusted Trading Date'].max()}")

    indexed_for_lookup = raw_index_data.set_index("Date").sort_index()
    e_filtered["Index_Close"] = e_filtered["Adjusted Trading Date"].apply(lambda x: get_index_close(x, indexed_for_lookup))

    for metric, min_value in filters.items():
        if metric in e_filtered.columns:
            before = e_filtered.shape[0]
            e_filtered = e_filtered[e_filtered[metric].notna()]
            e_filtered = e_filtered[e_filtered[metric] >= min_value]
            after = e_filtered.shape[0]
            print(f"e_filtered after {metric} >= {min_value}: {after} rows (was {before})")

    blue_data = e_filtered[e_filtered["Drift Open-to-Close"] > 0].groupby("Adjusted Trading Date").agg({"Drift Open-to-Close": "sum", "Index_Close": "first"}).reset_index()
    red_data = e_filtered[e_filtered["Drift Open-to-Close"] <= 0].groupby("Adjusted Trading Date").agg({"Drift Open-to-Close": "sum", "Index_Close": "first"}).reset_index()
    print(f"blue_data rows: {blue_data.shape[0]}, red_data rows: {red_data.shape[0]}")

    diff_data = pd.merge(
        blue_data[["Adjusted Trading Date", "Drift Open-to-Close", "Index_Close"]],
        red_data[["Adjusted Trading Date", "Drift Open-to-Close", "Index_Close"]],
        on="Adjusted Trading Date", how="outer", suffixes=('_blue', '_red')
    )
    diff_data.fillna({"Drift Open-to-Close_blue": 0, "Drift Open-to-Close_red": 0}, inplace=True)
    diff_data["Net"] = diff_data["Drift Open-to-Close_blue"] + diff_data["Drift Open-to-Close_red"]
    diff_data["Index_Close"] = diff_data["Index_Close_blue"].fillna(diff_data["Index_Close_red"])
    print(f"diff_data rows: {diff_data.shape[0]}")

    bubble_scale = 100
    blue_data["Bubble_Size"] = np.clip(blue_data["Drift Open-to-Close"].abs() * bubble_scale, 50, 1000)
    red_data["Bubble_Size"] = np.clip(red_data["Drift Open-to-Close"].abs() * bubble_scale, 50, 1000)
    diff_data["Bubble_Size"] = np.clip(diff_data["Net"].abs() * bubble_scale, 50, 1000)
    diff_data["Bubble_Color"] = diff_data["Net"].apply(lambda x: "#1f77b4" if x > 0 else ("#d62728" if x < 0 else "grey"))

    return index_filtered, blue_data, red_data, diff_data, e_filtered

# --------------------------------------------------------------------------------
# MAKE FIGURE
# --------------------------------------------------------------------------------

def make_figure(index_df, blue_df, red_df, diff_df, index_name="S&P 500"):
    print(f"\nmake_figure called with: index_df rows={index_df.shape[0]}, "
          f"blue_df rows={blue_df.shape[0]}, red_df rows={red_df.shape[0]}, diff_df rows={diff_df.shape[0]}")
    if not index_df.empty:
        print(f"index_df columns: {list(index_df.columns)}, sample:\n{index_df.head()}")
    if not blue_df.empty:
        print(f"blue_df columns: {list(blue_df.columns)}, sample:\n{blue_df.head()}")
    if not red_df.empty:
        print(f"red_df columns: {list(red_df.columns)}, sample:\n{red_df.head()}")
    if not diff_df.empty:
        print(f"diff_df columns: {list(diff_df.columns)}, sample:\n{diff_df.head()}")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=("Positive Drift", "Negative Drift", "Net Drift")
    )

    if index_df.empty:
        fig.update_layout(
            height=600,
            title_text=f"No {index_name} data available for this range."
        )
        print("Returning empty figure due to no index data.")
        return fig

    index_df = index_df.copy()
    index_df["Date"] = pd.to_datetime(index_df["Date"])

    # Merge bubble data with correct column names
    rename_dict = {}
    for df, col_name in [(blue_df, "Drift Open-to-Close"), (red_df, "Drift Open-to-Close"), (diff_df, "Net")]:
        if not df.empty:
            df["Adjusted Trading Date"] = pd.to_datetime(df["Adjusted Trading Date"])
            index_df = index_df.merge(
                df[["Adjusted Trading Date", col_name]],
                left_on="Date", right_on="Adjusted Trading Date", how="left"
            )
            if "Adjusted Trading Date" in index_df.columns:
                index_df = index_df.drop(columns=["Adjusted Trading Date"])
            if col_name == "Drift Open-to-Close" and df is blue_df:
                rename_dict["Drift Open-to-Close"] = "Positive_Drift"
            elif col_name == "Drift Open-to-Close" and df is red_df:
                rename_dict["Drift Open-to-Close"] = "Negative_Drift"
            elif col_name == "Net":
                rename_dict["Net"] = "Net_Drift"

    index_df = index_df.rename(columns={k: v for k, v in rename_dict.items() if k in index_df.columns})

    for col in ["Positive_Drift", "Negative_Drift", "Net_Drift"]:
        if col not in index_df.columns:
            index_df[col] = ""
        index_df[col] = index_df[col].fillna("")

    hover_template = (
        "Date: %{customdata[0]|%Y-%m-%d}<br>"
        f"{index_name} Close: %{{y:.2f}}<br>"
        "MA10: %{customdata[1]:.2f}<br>"
        "MA20: %{customdata[2]:.2f}<br>"
        "MA50: %{customdata[3]:.2f}<br>"
        "%{customdata[4]}%{customdata[5]}%{customdata[6]}"
    )

    customdata = np.stack([
        index_df["Date"],
        index_df["MA10"].fillna("N/A"),
        index_df["MA20"].fillna("N/A"),
        index_df["MA50"].fillna("N/A"),
        index_df["Positive_Drift"].apply(lambda x: f"Positive Drift: {x:.2f}<br>" if x != "" else ""),
        index_df["Negative_Drift"].apply(lambda x: f"Negative Drift: {x:.2f}<br>" if x != "" else ""),
        index_df["Net_Drift"].apply(lambda x: f"Net Drift: {x:.2f}<br>" if x != "" else "")
    ], axis=-1)

    # Row 1: Positive
    fig.add_trace(
        go.Scatter(
            x=index_df["Date"], y=index_df["Close"], mode="lines",
            name=f"{index_name} Close", line=dict(color="black"),
            customdata=customdata,
            hovertemplate=hover_template
        ),
        row=1, col=1
    )
    if "MA10" in index_df.columns:
        fig.add_trace(
            go.Scatter(
                x=index_df["Date"], y=index_df["MA10"], mode="lines",
                name="MA10", line=dict(color="blue", dash="dot"), showlegend=True,
                hoverinfo="skip"
            ),
            row=1, col=1
        )
    if "MA20" in index_df.columns:
        fig.add_trace(
            go.Scatter(
                x=index_df["Date"], y=index_df["MA20"], mode="lines",
                name="MA20", line=dict(color="orange", dash="dot"), showlegend=True,
                hoverinfo="skip"
            ),
            row=1, col=1
        )
    if "MA50" in index_df.columns:
        fig.add_trace(
            go.Scatter(
                x=index_df["Date"], y=index_df["MA50"], mode="lines",
                name="MA50", line=dict(color="green", dash="dot"), showlegend=True,
                hoverinfo="skip"
            ),
            row=1, col=1
        )
    if not blue_df.empty:
        fig.add_trace(
            go.Scatter(
                x=blue_df["Adjusted Trading Date"], y=blue_df["Index_Close"],
                mode="markers",
                marker=dict(size=blue_df["Bubble_Size"]/50, color="#1f77b4", opacity=0.6, line=dict(width=0.5, color="black")),
                name="Positive Drift",
                customdata=blue_df[["Adjusted Trading Date", "Drift Open-to-Close"]],
                hovertemplate="Date: %{customdata[0]|%Y-%m-%d}<br>Positive Drift: %{customdata[1]:.2f}%"
            ),
            row=1, col=1
        )

    # Row 2: Negative
    fig.add_trace(
        go.Scatter(
            x=index_df["Date"], y=index_df["Close"], mode="lines",
            name=f"{index_name} Close", line=dict(color="black"), showlegend=False,
            customdata=customdata,
            hovertemplate=hover_template
        ),
        row=2, col=1
    )
    if "MA10" in index_df.columns:
        fig.add_trace(
            go.Scatter(
                x=index_df["Date"], y=index_df["MA10"], mode="lines",
                name="MA10", line=dict(color="blue", dash="dot"), showlegend=False,
                hoverinfo="skip"
            ),
            row=2, col=1
        )
    if "MA20" in index_df.columns:
        fig.add_trace(
            go.Scatter(
                x=index_df["Date"], y=index_df["MA20"], mode="lines",
                name="MA20", line=dict(color="orange", dash="dot"), showlegend=False,
                hoverinfo="skip"
            ),
            row=2, col=1
        )
    if "MA50" in index_df.columns:
        fig.add_trace(
            go.Scatter(
                x=index_df["Date"], y=index_df["MA50"], mode="lines",
                name="MA50", line=dict(color="green", dash="dot"), showlegend=False,
                hoverinfo="skip"
            ),
            row=2, col=1
        )
    if not red_df.empty:
        fig.add_trace(
            go.Scatter(
                x=red_df["Adjusted Trading Date"], y=red_df["Index_Close"],
                mode="markers",
                marker=dict(size=red_df["Bubble_Size"]/50, color="#d62728", opacity=0.6, line=dict(width=0.5, color="black")),
                name="Negative Drift",
                customdata=red_df[["Adjusted Trading Date", "Drift Open-to-Close"]],
                hovertemplate="Date: %{customdata[0]|%Y-%m-%d}<br>Negative Drift: %{customdata[1]:.2f}%"
            ),
            row=2, col=1
        )

    # Row 3: Net
    fig.add_trace(
        go.Scatter(
            x=index_df["Date"], y=index_df["Close"], mode="lines",
            name=f"{index_name} Close", line=dict(color="black"), showlegend=False,
            customdata=customdata,
            hovertemplate=hover_template
        ),
        row=3, col=1
    )
    if "MA10" in index_df.columns:
        fig.add_trace(
            go.Scatter(
                x=index_df["Date"], y=index_df["MA10"], mode="lines",
                name="MA10", line=dict(color="blue", dash="dot"), showlegend=False,
                hoverinfo="skip"
            ),
            row=3, col=1
        )
    if "MA20" in index_df.columns:
        fig.add_trace(
            go.Scatter(
                x=index_df["Date"], y=index_df["MA20"], mode="lines",
                name="MA20", line=dict(color="orange", dash="dot"), showlegend=False,
                hoverinfo="skip"
            ),
            row=3, col=1
        )
    if "MA50" in index_df.columns:
        fig.add_trace(
            go.Scatter(
                x=index_df["Date"], y=index_df["MA50"], mode="lines",
                name="MA50", line=dict(color="green", dash="dot"), showlegend=False,
                hoverinfo="skip"
            ),
            row=3, col=1
        )
    if not diff_df.empty:
        fig.add_trace(
            go.Scatter(
                x=diff_df["Adjusted Trading Date"], y=diff_df["Index_Close"],
                mode="markers",
                marker=dict(size=diff_df["Bubble_Size"]/50, color=diff_df["Bubble_Color"], opacity=0.6, line=dict(width=0.5, color="black")),
                name="Net Drift",
                customdata=diff_df[["Adjusted Trading Date", "Net"]],
                hovertemplate="Date: %{customdata[0]|%Y-%m-%d}<br>Net Drift: %{customdata[1]:.2f}%"
            ),
            row=3, col=1
        )

    fig.update_layout(
        height=1000,
        width=1600,
        title_text=f"{index_name} Earnings Drift Analysis",
        showlegend=True,
        plot_bgcolor='white',
        hovermode="x unified"
    )

    if not fig.data:
        print("No traces added. Using fallback dummy trace.")
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="Dummy", mode="lines"), row=1, col=1)
        fig.update_layout(title_text="No Data: Fallback Plot")

    return fig

# --------------------------------------------------------------------------------
# DASH APP
# --------------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "Earnings Drift Analyzer"

index_filtered, blue_data, red_data, diff_data, earnings_filtered_global = process_data(initial_year, initial_index, **initial_values)
initial_fig = make_figure(index_filtered, blue_data, red_data, diff_data, initial_index)

app.layout = html.Div([
    html.Div([
        html.H1("Earnings Drift Analysis", style={'color': FONT_COLOR, 'marginBottom': 20, 'fontWeight': 'bold'}),
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Index", className="control-label"),
                    dcc.Dropdown(
                        id='index-slider',
                        options=[{'label': k, 'value': k} for k in index_options.keys()],
                        value=initial_index,
                        clearable=False,
                        style={'width': '100%', 'marginBottom': '15px'}
                    ),
                    html.Label("Year", className="control-label"),
                    dcc.Dropdown(
                        id='year-slider',
                        options=[{'label': str(y), 'value': y} for y in years],
                        value=initial_year,
                        clearable=False,
                        style={'width': '100%', 'marginBottom': '15px'}
                    ),
                    html.Div([
                        html.Label("Pre 2 Weeks (%)", className="control-label"),
                        dcc.Slider(id='pre-2-weeks-slider', min=-50, max=50, value=initial_values["Pre 2 Weeks"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Pre 1 Week (%)", className="control-label"),
                        dcc.Slider(id='pre-1-week-slider', min=-50, max=50, value=initial_values["Pre 1 Week"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Pre 3 Days (%)", className="control-label"),
                        dcc.Slider(id='pre-3-days-slider', min=-50, max=50, value=initial_values["Pre 3 Days"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Pre 2 Days (%)", className="control-label"),
                        dcc.Slider(id='pre-2-days-slider', min=-50, max=50, value=initial_values["Pre 2 Days"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Pre 1 Day (%)", className="control-label"),
                        dcc.Slider(id='pre-1-day-slider', min=-50, max=50, value=initial_values["Pre 1 Day"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Earnings Move (%)", className="control-label"),
                        dcc.Slider(id='earnings-move-slider', min=-50, max=50, value=initial_values["Earnings Move"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Opening Gap (%)", className="control-label"),
                        dcc.Slider(id='opening-gap-slider', min=-50, max=50, value=initial_values["Opening Gap"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Open-to-High (%)", className="control-label"),
                        dcc.Slider(id='open-to-high-slider', min=-50, max=50, value=initial_values["Open-to-High"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Open-to-Low (%)", className="control-label"),
                        dcc.Slider(id='open-to-low-slider', min=-50, max=50, value=initial_values["Open-to-Low"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Drift Open-to-Close (%)", className="control-label"),
                        dcc.Slider(id='drift-slider', min=-50, max=50, value=initial_values["Drift Open-to-Close"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Post 1 Day (%)", className="control-label"),
                        dcc.Slider(id='post-1-day-slider', min=-50, max=50, value=initial_values["Post 1 Day"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Post 2 Days (%)", className="control-label"),
                        dcc.Slider(id='post-2-days-slider', min=-50, max=50, value=initial_values["Post 2 Days"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Post 3 Days (%)", className="control-label"),
                        dcc.Slider(id='post-3-days-slider', min=-50, max=50, value=initial_values["Post 3 Days"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Post 1 Week (%)", className="control-label"),
                        dcc.Slider(id='post-1-week-slider', min=-50, max=50, value=initial_values["Post 1 Week"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                        html.Label("Post 2 Weeks (%)", className="control-label"),
                        dcc.Slider(id='post-2-weeks-slider', min=-50, max=50, value=initial_values["Post 2 Weeks"], 
                                   marks={i: str(i) for i in range(-50, 51, 10)}, step=1),
                    ], style=SLIDERS_CONTAINER_STYLE),
                    html.Div([
                        html.Div([
                            html.Button("Apply Filters", id="update-button", className="button-primary", style={'marginRight': 10}),
                            html.Button("Reset", id="reset-button", className="button-danger"),
                        ], style={'marginTop': 20}),
                        html.Div(id="selected-date", 
                                 style={'marginTop': 20, 'padding': 10, 
                                        'border': f'1px solid {ACCENT_COLOR}', 
                                        'borderRadius': 5, 'fontWeight': '500'}),
                        html.Button("Show Selected Date Data", id="show-data-button", className="button", 
                                    style={'marginTop': 20, 'width': '100%'}),
                        html.Button("Download Filtered Data (CSV)", id="download-button", className="button", 
                                    style={'marginTop': 10, 'width': '100%'})
                    ]),
                    dcc.Download(id="download-dataframe-csv"),
                ], style=CONTROL_PANEL_STYLE)
            ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                dcc.Graph(
                    id="bubble-chart",
                    figure=initial_fig,
                    config={'displayModeBar': True},
                    style={'height': '70vh', 'borderRadius': 10, 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}
                )
            ], style={'width': '80%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
        ], style={'display': 'flex'}),
        html.Div(
            id="data-table-container",
            children=[
                dash_table.DataTable(
                    id="data-table",
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'center', 'padding': '8px'},
                    style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'borderBottom': '2px solid #dee2e6'}
                )
            ],
            style={'width': '100%', 'marginTop': '20px', 'padding': '20px', 
                   'backgroundColor': 'white', 'borderRadius': '10px', 
                   'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'resize': 'vertical', 
                   'overflow': 'auto', 'minHeight': '100px', 'maxHeight': '50vh'}
        )
    ], style={'padding': 30, 'backgroundColor': BG_COLOR})
], style={'fontFamily': 'Arial, sans-serif', 'minHeight': '100vh'})

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            .control-label {{
                font-weight: 600; 
                color: {FONT_COLOR}; 
                margin-top: 15px; 
                display: block;
            }}
            .slider .rc-slider-track {{
                background-color: {ACCENT_COLOR};
            }}
            .rc-slider-handle {{
                border-color: {ACCENT_COLOR} !important;
            }}
            .button-primary {{
                background-color: {ACCENT_COLOR}; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                border-radius: 5px; 
                cursor: pointer; 
                transition: all 0.3s ease;
            }}
            .button-primary:hover {{
                background-color: #2980b9;
            }}
            .button-danger {{
                background-color: {DANGER_COLOR}; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                border-radius: 5px; 
                cursor: pointer; 
                transition: all 0.3s ease;
            }}
            .button-danger:hover {{
                background-color: #c0392b;
            }}
            .button {{
                background-color: #6c757d; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                border-radius: 5px; 
                cursor: pointer; 
                transition: all 0.3s ease;
            }}
            .button:hover {{
                background-color: #5a6268;
            }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

@app.callback(
    Output("bubble-chart", "figure"),
    [Input("update-button", "n_clicks")],
    [
        State("index-slider", "value"), 
        State("year-slider", "value"),
        State("pre-2-weeks-slider", "value"), 
        State("pre-1-week-slider", "value"),
        State("pre-3-days-slider", "value"), 
        State("pre-2-days-slider", "value"),
        State("pre-1-day-slider", "value"), 
        State("earnings-move-slider", "value"),
        State("opening-gap-slider", "value"), 
        State("open-to-high-slider", "value"),
        State("open-to-low-slider", "value"), 
        State("drift-slider", "value"),
        State("post-1-day-slider", "value"), 
        State("post-2-days-slider", "value"),
        State("post-3-days-slider", "value"),
        State("post-1-week-slider", "value"),
        State("post-2-weeks-slider", "value")
    ]
)
def update_figure(
    n_clicks, index_type, year,
    pre_2_weeks, pre_1_week, pre_3_days, pre_2_days, pre_1_day,
    earnings_move, opening_gap, open_to_high, open_to_low, drift_open_to_close,
    post_1_day, post_2_days, post_3_days, post_1_week, post_2_weeks
):
    global earnings_filtered_global
    filters = {
        "Pre 2 Weeks": pre_2_weeks, "Pre 1 Week": pre_1_week, "Pre 3 Days": pre_3_days, 
        "Pre 2 Days": pre_2_days, "Pre 1 Day": pre_1_day, "Earnings Move": earnings_move, 
        "Opening Gap": opening_gap, "Open-to-High": open_to_high, "Open-to-Low": open_to_low, 
        "Drift Open-to-Close": drift_open_to_close, "Post 1 Day": post_1_day, 
        "Post 2 Days": post_2_days, "Post 3 Days": post_3_days, "Post 1 Week": post_1_week, 
        "Post 2 Weeks": post_2_weeks
    }
    index_f, blue_d, red_d, diff_d, earnings_filtered_global = process_data(year, index_type, **filters)
    fig = make_figure(index_f, blue_d, red_d, diff_d, index_type)
    return fig

@app.callback(
    Output("selected-date", "children"),
    [Input("bubble-chart", "clickData"), Input("reset-button", "n_clicks")]
)
def update_selected_date(clickData, reset_clicks):
    global selected_date
    if reset_clicks and reset_clicks > 0:
        selected_date = None
        return "No date selected"
    if clickData:
        point = clickData["points"][0]
        selected_date = pd.to_datetime(point["customdata"][0])
        return f"Selected Date: {selected_date.strftime('%Y-%m-%d')}"
    return "No date selected"

@app.callback(
    Output("data-table", "data"),
    Output("data-table", "columns"),
    Output("data-table", "style_data_conditional"),
    [Input("show-data-button", "n_clicks")],
    prevent_initial_call=True
)
def show_data(n_clicks):
    global selected_date, earnings_filtered_global
    if selected_date is None:
        return [], [], []
    filtered = earnings_filtered_global[earnings_filtered_global["Adjusted Trading Date"] == selected_date].copy()
    for col in ["Adjusted Trading Date", "Index_Close"]:
        if col in filtered.columns:
            filtered.drop(columns=[col], inplace=True)
    data = filtered.round(2).to_dict('records')
    columns = [{"name": i, "id": i} for i in filtered.columns]
    style_data_conditional = []
    for col in filtered.columns:
        try:
            filtered[col].astype(float)
            style_data_conditional.extend([
                {'if': {'filter_query': f'{{{col}}} < 0', 'column_id': col}, 'color': DANGER_COLOR, 'fontWeight': 'bold'},
                {'if': {'filter_query': f'{{{col}}} >= 0', 'column_id': col}, 'color': 'black', 'fontWeight': 'bold'}
            ])
        except (ValueError, TypeError):
            pass
    return data, columns, style_data_conditional

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    [
        State("year-slider", "value"),
        State("index-slider", "value"),
        State("pre-2-weeks-slider", "value"), 
        State("pre-1-week-slider", "value"),
        State("pre-3-days-slider", "value"), 
        State("pre-2-days-slider", "value"),
        State("pre-1-day-slider", "value"), 
        State("earnings-move-slider", "value"),
        State("opening-gap-slider", "value"), 
        State("open-to-high-slider", "value"),
        State("open-to-low-slider", "value"), 
        State("drift-slider", "value"),
        State("post-1-day-slider", "value"), 
        State("post-2-days-slider", "value"),
        State("post-3-days-slider", "value"),
        State("post-1-week-slider", "value"),
        State("post-2-weeks-slider", "value")
    ],
    prevent_initial_call=True
)
def download_filtered_data(
    n_clicks, year, index_type,
    pre_2_weeks, pre_1_week, pre_3_days, pre_2_days, pre_1_day,
    earnings_move, opening_gap, open_to_high, open_to_low, drift_open_to_close,
    post_1_day, post_2_days, post_3_days, post_1_week, post_2_weeks
):
    filters = {
        "Pre 2 Weeks": pre_2_weeks, "Pre 1 Week": pre_1_week, "Pre 3 Days": pre_3_days, 
        "Pre 2 Days": pre_2_days, "Pre 1 Day": pre_1_day, "Earnings Move": earnings_move, 
        "Opening Gap": opening_gap, "Open-to-High": open_to_high, "Open-to-Low": open_to_low, 
        "Drift Open-to-Close": drift_open_to_close, "Post 1 Day": post_1_day, 
        "Post 2 Days": post_2_days, "Post 3 Days": post_3_days, "Post 1 Week": post_1_week, 
        "Post 2 Weeks": post_2_weeks
    }
    _, _, _, _, earnings_filtered = process_data(year, index_type, **filters)
    filename = f"earnings_filtered_{index_type.replace(' ', '_')}_{year}.csv"
    return dcc.send_data_frame(earnings_filtered.to_csv, filename=filename, index=False)

# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True)  # Set debug=True for better error reporting