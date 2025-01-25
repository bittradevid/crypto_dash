import streamlit as st 
import pandas as pd
import yfinance as yf
import cufflinks as cf
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAI
from google.cloud import aiplatform
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from htmlCss import css
import numpy as np
from datetime import datetime
from tzlocal import get_localzone
import feedparser
import json
import os
import ssl
import requests


###SETUP DE LA APP -- HORARIO --  RSS Y MAS 

# Crear un contexto SSL que no verifique certificados
ssl._create_default_https_context = ssl._create_unverified_context

# URL del feed RSS
RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss"

# Obtener la zona horaria local del sistema
local_timezone = get_localzone()

#### IMPORTAR TOKENS JSON

# Ruta del archivo JSON
json_file_path = "tokens.json"


# Leer tokens desde archivo JSON
def load_tokens_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data.get("tokens", [])  # Devuelve la lista de tokens o una lista vacía si no existe

# Cargar tokens
tokens = load_tokens_from_json(json_file_path)

#######TERMINA IMPORTAR TOKENS


# Configura tu proyecto y ubicación
project_id = "morningbriefing-447904"  # Cambia esto por el ID de tu proyecto
location = "us-central1"     # Asegúrate de usar la región correcta

# Inicializa Vertex AI
vertexai.init(project=project_id, location=location)

# Cargar el modelo y la sesión de chat
if 'chat' not in st.session_state:
    model = GenerativeModel("gemini-1.0-pro")
    st.session_state.chat = model.start_chat()


###INICIA APP - - 

def main():

    if 'sidebar_state' not in st.session_state:
        st.session_state.sidebar_state = 'expanded'

    
st.set_page_config(page_title="CryptoDesk", page_icon="📈",menu_items={'About':'NQPMEDIA iA_Apps 2024-2025'})




st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

st.write(css, unsafe_allow_html=True)
st.markdown("""
    <style>
        [data-testid=stSidebar] {
        margin-right: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    

# App 
st.markdown('''
# CRYPTO DESK 1.3 📈''')
st.subheader('Boost your Trading - Optimize your Time ⌚')
st.write('---')
st.write('Today Quotes 🧠')
st.markdown('#### 🎨 A designer knows he has achieved perfection when there is nothing left to take away')
st.markdown('<div style="text-align: right;">Michael J. Saylor 📝</div>', unsafe_allow_html=True)
st.write('---')

# Pandas Options
pd.options.display.float_format = '${:,.2f}'.format

# Load from .env
load_dotenv()
base_url = os.getenv("BASE_URL")
# Configurar VertexAI
vertex_ai = VertexAI()


# Binance ticker's list DataFrame
dd = pd.read_json('https://api.binance.com/api/v3/ticker/price')

# Function for Binance URL builder
def make_klines_url(symbol, **kwargs):
    url = base_url + f"?symbol={symbol}" 

    for key, value in kwargs.items():
        url += f"&{key}={value}"
    
    return url

# Custom function for rounding values
def round_value(input_value):
    if isinstance(input_value, (pd.Series, pd.DataFrame)):
        if input_value.values > 1:
            return float(round(input_value, 2))
        else:
            return float(round(input_value, 8))
    else:
        # Manejo para valores flotantes directamente
        if input_value > 1:
            return round(input_value, 2)
        else:
            return round(input_value, 8) 

# STREAMLIT Sidebar Price
st.sidebar.image('assets/portada1.png', )  

st.sidebar.header('🌤️ MORNING BRIEFING 📈')

price_ticker = st.sidebar.selectbox('Select Token', (tokens))
st.sidebar.write('or type to search 🔎')
interval_selectbox = st.sidebar.selectbox('Interval', ("1w","1d", "4h", "1h", "30m", "15m", "5m", "1m"))

st.sidebar.subheader('📟 Explore more Tools ⬇️')


# Obtener datos de precios actuales desde la API de Binance
df_current_prices = pd.read_json('https://api.binance.com/api/v3/ticker/price')
df_top_gainers = pd.read_json('https://api.binance.com/api/v3/ticker/24hr')

# Filtrar el precio actual del token seleccionado
col_df = df_current_prices[df_current_prices.symbol == price_ticker] 
col_price = round_value(float(col_df.price))


#### MAIN - DESK 

# STREAMLIT Price metric
st.metric(label=price_ticker,value=col_price) 
 
# Binance klines DataFrame Preparation
pd.options.display.float_format = '${:,.2f}'.format
klines_url = make_klines_url(price_ticker, interval=interval_selectbox)
klines_ticker_price = pd.read_json(klines_url)
klines_ticker_price.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
                               'Quote Asset Volume', 'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']
klines_ticker_price.drop(['Close Time', 'Quote Asset Volume', 'Number of Trades', 'TB Base Volume', 'TB Quote Volume','Ignore'], axis=1, inplace=True)
klines_ticker_price['Date'] = (
    pd.to_datetime(klines_ticker_price['Date'] / 1000, unit='s')
    .dt.tz_localize('UTC')
    .dt.tz_convert(local_timezone)
)


# Función para verificar la tendencia de los precios
def check_price_trend(prices):
    last_8_prices = prices.tail(8)
    increasing_count = (last_8_prices.diff() > 0).sum()
    decreasing_count = (last_8_prices.diff() < 0).sum()
    
    if increasing_count / 8 > 0.52:
        return "🟢 ASCENDING 👆"
    elif decreasing_count / 8 > 0.52:
        return "🔴 DESCENDING 🔻"
    else:
        return "🟠 CORRECTION 🔄"


# STREAMLIT HISTORICAL PRICE 
st.subheader(f'🔵 {price_ticker} ⚙️ Historical Price - TimeFrame ({interval_selectbox})🔥')
klines_ticker_price['Date'] = klines_ticker_price['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
table_html = (
    klines_ticker_price
    .sort_values(by='Date')
    .tail()
    .reset_index(drop=True)
    .to_html(index=False)  # No mostrar índice
)
st.markdown(table_html, unsafe_allow_html=True)

text = '''---'''
st.markdown(text)

price_trend = check_price_trend(klines_ticker_price['Close'])
st.subheader(f'Price Trend 📈 for 🔵 {price_ticker} on {interval_selectbox} 🕒')
st.header(f" {price_trend}")


# STREAMLIT functions klines Dataframe Plotting
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=klines_ticker_price['Date'], y=klines_ticker_price['Close'], name='Close', fillcolor='blue', hovertext='On Range'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_raw_data_log():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=klines_ticker_price['Date'], y=klines_ticker_price['Close'], name="Close"))
	fig.update_yaxes(type="log")
	fig.layout.update(xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

def plot_bb_data(): ### CHART LIVE #####
    qf=cf.QuantFig(klines_ticker_price,legend='top',name='GS')
    ##qf._add_study()  ##HACER EL ESTUDIO DE ENTRY LONG - ENTRY SHORT - TPs 
    qf.add_ema(periods=[21,55], color=['green', 'red'])
    qf.add_volume(colorchange=True) 
    fig = qf.iplot(asFigure=True, title=f'{price_ticker} on {interval_selectbox} Chart')
    st.plotly_chart(fig, theme='streamlit')

def plot_bb_rsi(): ### CHART RSI#####
    qf=cf.QuantFig(klines_ticker_price,legend='top',name='RSI')
    qf.add_rsi(showbands=True)
    qf.add_bollinger_bands() 
    fig = qf.iplot(asFigure=True, title=f'Bollinger Bands & RSI for {price_ticker}', )
    st.plotly_chart(fig)    


######VERTEX AI 

# Función para obtener la respuesta del chat
def get_chat_response(prompt: str):
    response = st.session_state.chat.send_message(prompt)
    return response.text


###PROMPT WITH IN HOUSE DATA 

# Función para obtener los datos relevantes para el prompt
def extract_data_for_prompt(klines_data, price_ticker, interval_selectbox):
    # Extrae solo los últimos datos relevantes de 'klines_ticker_price' como ejemplo
    latest_data = klines_data.tail(23)  # Tomando los últimos 5 valores como ejemplo

    # Extraer los datos de 'Close', 'EMA' (o cualquier otro indicador que quieras), y el 'RSI'
    open_prices = latest_data['Open'].tolist()
    close_prices = latest_data['Close'].tolist()
    volume_data = latest_data['Volume'].tolist()

    # Construye el prompt con los datos extraídos
    prompt = f"Generate a prediction for the next 8 candles for the token {price_ticker} on a {interval_selectbox} timeframe. The latest opening prices are: {open_prices}, the latest closing prices are: {close_prices}, and the volume was: {volume_data}. Summarize the expected trend and key characteristics (e.g., bullish, bearish, or sideways movement) for the upcoming 8 candles. After the prediction, propose a strategy to maximize profits based on this forecast. Avoid listing the candles individually and provide the response in a concise and actionable format. Exclude titles, additional information, and resources. Include a disclaimer at the end."
    
    return prompt

######NEWS FEED 

def fetch_news(url):
    """Fetch news items from the RSS feed."""
    feed = feedparser.parse(url)
    return feed.entries


def display_news(news_entries):
    """Display the latest 3 news items with styled containers."""
    for i in range(min(3, len(news_entries))):  # Mostrar máximo 5 noticias
        entry = news_entries[i]
        title = entry.title
        link = entry.link

        # Renderizar cada noticia con estilo
        st.markdown(f"""
        <div style="
            background-color: #eaeaea;
            border: 5px solid #0e1117; 
            border-radius: 10px; 
            padding: 3px; 
            margin-bottom: 10px;">
            <a href="{link}" target="_blank" style="
                text-decoration: none; 
                font-weight: bold; 
                color: #0e1117;
                font-size: 16px;">
                📰 {title} ...see more 👀
            </a>
        </div>
        """, unsafe_allow_html=True)

#####TOP GAINERS

def get_top_gainers():
    # Hacer una solicitud GET a la API de Binance para obtener los tickers de las últimas 24 horas
    url = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(url)
    
    # Comprobar que la solicitud fue exitosa
    if response.status_code != 200:
        st.error(f"Error al obtener datos de la API de Binance: {response.status_code}")
        return []

    # Obtener los datos de la respuesta en formato JSON
    tickers = response.json()

    # Filtrar los tickers con un cambio positivo en el precio
    gainers = [ticker for ticker in tickers if float(ticker['priceChangePercent']) > 0]

    # Ordenar los tickers por el mayor porcentaje de ganancia
    gainers_sorted = sorted(gainers, key=lambda x: float(x['priceChangePercent']), reverse=True)

    return gainers_sorted[:10]  # Top 10 gainers


###CALCULADORA - - 

def calculate_percentage_gain_loss(entry, exit, position):
        if position == 'LONG':
            # Formula for LONG: ((exit - entry) / entry) * 100
            return ((exit - entry) / entry) * 100
        elif position == 'SHORT':
            # Formula for SHORT: ((entry - exit) / entry) * 100
            return ((entry - exit) / entry) * 100
        else:
            return 0.0

# STREAMLIT Multi Plot Display without Dropdown 

text = '''---'''
st.markdown(text)

st.subheader(f'🤖 Create Trading Strategy for 🔵 {price_ticker} on a {interval_selectbox} Chart')

# Botón para generar la estrategia de trading
if st.button(f'Launch AI Trading Strategy for {price_ticker}'):
    # Construir el prompt usando las selecciones del usuario
    prompt = extract_data_for_prompt(klines_ticker_price, price_ticker, interval_selectbox)
    
    # Obtener la respuesta del modelo
    response = get_chat_response(prompt)
    
    # Mostrar la respuesta
    st.write(response)

text = '''---'''
st.markdown(text)    

st.subheader(f'🔵 {price_ticker} 💵🔥 Live Chart 📊 {interval_selectbox} 🔥💶 ')
plot_bb_data()  # Asegúrate de que esta función esté definida

text = '''---'''
st.markdown(text)

st.subheader(f'🔵 {price_ticker} 💵🔥 BB & RSI 📊 {interval_selectbox} 🔥💶 ')
plot_bb_rsi()  # Asegúrate de que esta función esté definida  

text = '''---'''
st.markdown(text)

st.subheader(f'🔵 {price_ticker} 🕰️ Historical Volume 📊')
express = px.area(klines_ticker_price, x='Date', y='Volume')
st.write(express)

text = '''---'''
st.markdown(text)

st.subheader(f'🔵 {price_ticker} 🔥📈 On Range Tool 🛠️🔥')
plot_raw_data()  # Asegúrate de que esta función esté definida


text = '''---'''
st.markdown(text)

st.subheader(f'🌎 Global Market 🔥💣 Liquidations 💣🔥 12h 🚨🚨')
st.image('assets/liquidations.png')         

text = '''---''' 
st.markdown(text)  

st.subheader(f'🟢 Crypto 🔥😰 Fear and Greed 😤 Index 🔥 ')
st.image("https://alternative.me/crypto/fear-and-greed-index.png", caption="Latest Crypto Fear & Greed Index")

text = '''---'''
st.markdown(text)



with st.sidebar:
    st.header("🛩️ Latest Crypto News 🗞️ ")
    display_news(fetch_news(RSS_URL))

    st.write('___')

    ### TOP GAINERS

    def show_top_gainers():
        top_gainers = get_top_gainers()

        st.sidebar.markdown('# Top 10 Gainers 🏆 (24h)')

        if top_gainers:
            for gainer in top_gainers:
                symbol = gainer['symbol']
                percent_change = gainer['priceChangePercent']
                st.sidebar.markdown(f"""
                    <div style="
                        background-color: #eaeaea;
                        border: 3px solid #0e1117; 
                        border-radius: 10px; 
                        padding: 3px; 
                        text-decoration: none; 
                        font-weight: bold;
                        color: #0e1117;
                        font-size: 16px;
                        margin-bottom: 10px;">🟢 {symbol}: {percent_change}% 🏅
                    </div>
                        """, unsafe_allow_html=True)
        else:
            st.sidebar.text("No se pudo obtener los datos de los top gainers.")

# Llamamos a la función para mostrar los top gainers
    show_top_gainers()

    st.write('___')

    ## CALCULATOR
    
    st.subheader('📈 POSITION CALCULATOR 📊')
    ### TOKEN
    st.subheader(f'{price_ticker} = {col_price}')
    
    # Position size in USD
    position_size = st.number_input(label="POSITION SIZE (USD) 💵", min_value=0.0, value=1000.0, step=100.0)

    # LONG OR SHORT
    position_type = st.selectbox('🟢 LONG / SHORT 🔴', ('LONG', 'SHORT'))
    
    # Input 1: Entry Price
    entry_price = st.number_input(label="ENTRY PRICE 🟢", value=col_price, step=0.1)
    
    # Input 2: Exit Price
    exit_price = st.number_input(label="EXIT PRICE 🛑", value=col_price, step=0.1)
    
    # Function to calculate gain/loss percentage and USD
    def calculate_percentage_gain_loss(entry, exit, position, size):
        if entry > 0:  # Avoid division by zero
            if position == 'LONG':
                percent_change = ((exit - entry) / entry) * 100
                dollar_change = (exit - entry) * (size / entry)
            elif position == 'SHORT':
                percent_change = ((entry - exit) / entry) * 100
                dollar_change = (entry - exit) * (size / entry)
            return percent_change, dollar_change
        else:
            return 0.0, 0.0

    # Button to calculate
    if st.button("CALCULATE 💹"):
        percent_result, dollar_result = calculate_percentage_gain_loss(entry_price, exit_price, position_type, position_size)
        if percent_result != 0:  # Ensure meaningful output
            st.write(f"### 🚀 {position_type.upper()} RESULT:")
            st.write(f"📊 Percentage Change: {'+' if percent_result > 0 else ''}{percent_result:.2f}%")
            st.write(f"💵 Dollar Change: {'+' if dollar_result > 0 else ''}${dollar_result:.2f}")
        else:
            st.error("Ensure entry price is greater than 0 and valid values are entered.")




st.sidebar.write('<div style="text-align: center;">By🎙️_0xdEVbEN_🎸 </div>', unsafe_allow_html=True)

