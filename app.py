import streamlit as st 
import pandas as pd
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
from io import BytesIO
from PIL import Image
import time
from st_paywall import add_auth
import pyperclip


###SETUP DE LA APP -- HORARIO --  RSS Y MAS de eso

# Crear un contexto SSL que no verifique certificados
ssl._create_default_https_context = ssl._create_unverified_context

# URL del feed RSS
RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss"

# Obtener la zona horaria local del sistema
local_timezone = get_localzone()


# Cargar los quotes desde el archivo JSON
with open('quotes.json') as f:
    quotes = json.load(f)

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


####BTC Dominance 

# CoinMarketCap API Key (sustituye con tu clave)
CMC_API_KEY = "355e30fb-2b81-4abf-8bc6-18311fc2d2ef"

# URL de la API
CMC_URL = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"

# Encabezados para la solicitud
headers = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": CMC_API_KEY,
}



###INICIA APP - - 

def main():

    if 'sidebar_state' not in st.session_state:
        st.session_state.sidebar_state = 'expanded'

    
st.set_page_config(page_title="CryptoDesk", page_icon="📈",menu_items={'About':'NQPMEDIA iA_Apps 2024-2025'})

# Agregar los meta tags personalizados
def add_meta_tags():
    meta_tags = """
    <meta property="og:title" content="☕ Morning Briefing - Boost your Trading - Optimize your Time ⌚">
    <meta property="og:description" content=" Get real-time trading strategies, chart analysis, whale alerts, liquidation maps, and top gainers & losers 📈📉 –  powered by Gemini 2.0 , Stay ahead of the market! 📈🔍.">
    <meta property="og:image" content="https://nqpmedia.com/assets/preview_image.png">
    <meta property="og:url" content="https://morningbriefing.nqpmedia.com">
    <meta name="twitter:card" content="summary_large_image">
    """
    st.markdown(meta_tags, unsafe_allow_html=True)

add_meta_tags()  # Llamamos a la función para insertar las meta tags


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
# CRYPTO DESK 2.0 📈''')
st.subheader('Boost your Trading - Optimize your Time ⌚') 
st.write('---')

# Obtener el índice del quote del día (usando el día del año para que cambie diariamente)
day_of_year = datetime.now().timetuple().tm_yday
quote_of_the_day = quotes[day_of_year % len(quotes)]

# Mostrar el quote del día
st.write('Today Quotes 🧠')
st.markdown(f"#### 💭 {quote_of_the_day['quote']}💡")
st.markdown(f"<div style='text-align: right;'>— {quote_of_the_day['author']} 📝</div>", unsafe_allow_html=True)
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

####add_auth(required=True)

price_ticker = st.sidebar.selectbox('Select Token', (tokens))
st.sidebar.write('or type to search 🔎')
interval_selectbox = st.sidebar.selectbox('Interval', ("1w","1d", "4h", "1h", "30m", "15m", "5m", "1m"))



st.sidebar.subheader('🟩 Explore more Tools ⬇️')


# Obtener datos de precios actuales desde la API de Binance
df_current_prices = pd.read_json('https://api.binance.com/api/v3/ticker/price')
df_top_gainers = pd.read_json('https://api.binance.com/api/v3/ticker/24hr')

# Filtrar el precio actual del token seleccionado
col_df = df_current_prices[df_current_prices.symbol == price_ticker] 
if not col_df.empty:
    col_price = round_value(float(col_df.price.iloc[0]))  
else:
    col_price = 0  # O algún valor por defecto


####ticker_mod = price_ticker[:-1] if price_ticker.endswith('USDT') else price_ticker
#### MAIN - DESK 


# STREAMLIT Price metric
formatted_price = f"{col_price:,.8f}" if col_price < 1 else f"{col_price:,.2f}" 
st.header(f" {price_ticker}📌  ")
st.markdown(f'# ${formatted_price}')   


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
    last_8_prices = prices.tail(10)
    increasing_count = (last_8_prices.diff() > 0).sum()
    decreasing_count = (last_8_prices.diff() < 0).sum()
    
    if increasing_count / 10 > 0.52:
        return "🟢 ASCENDING 👆"
    elif decreasing_count / 10 > 0.52:
        return "🔴 DESCENDING 🔻"
    else:
        return "🟠 CORRECTION 🔄"

def get_long_entry(prices):
    last_8_prices = prices.tail(10)  # Tomamos los últimos 8 cierres
    lowest_price = last_8_prices.min()  # Encontramos el precio mínimo
    
    return lowest_price

def get_short_entry(prices):
    last_8_prices = prices.tail(10)  # Tomamos los últimos 8 cierres
    highest_price = last_8_prices.max()  # Encontramos el precio máximo
    
    return highest_price


def format_price(price):
    """Formatea el precio según su valor:
       - Si es menor a 1 -> 8 decimales
       - Si es mayor o igual a 1 -> sin decimales, con separador de miles"""
    return f"{price:,.8f}" if price < 1 else f"{price:,.3f}"



text = '''---'''
st.markdown(text)

price_trend = check_price_trend(klines_ticker_price['Close'])
st.subheader(f'Price Action Trend 📈 for 🔵 {price_ticker} on {interval_selectbox} 🕒')
st.header(f" {price_trend}")

text = '''---'''
st.markdown(text)

long_entry_price = get_long_entry(klines_ticker_price['Close'])
short_entry_price = get_short_entry(klines_ticker_price['Close'])

st.subheader(f'🚥 Next Entry Price for 🔵 {price_ticker} on {interval_selectbox} 🕒')
st.header(f"🟢 Long ⬆️ ${format_price(long_entry_price)}")
st.header(f"🔴 Short ⬇️ ${format_price(short_entry_price)}")

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
    fig = qf.iplot(asFigure=True, title=f'Bollinger Bands & RSI for {price_ticker}')
    fig.update_layout(showlegend=False)	
    st.plotly_chart(fig)    


# Obtener datos de la API  BTC Dominance
def get_market_data():
    response = requests.get(CMC_URL, headers=headers)
    if response.status_code == 200:
        data = response.json()["data"]
        return {
            "btc_dominance": round(data["btc_dominance"], 2),
            "eth_dominance": round(data["eth_dominance"], 2),
            "others_dominance": round(100 - data["btc_dominance"] - data["eth_dominance"], 2),
            "history": [
                {"label": "Yesterday", "btc": 59.7, "eth": 10.1, "others": 30.2},
                {"label": "Last Week", "btc": 60.6, "eth": 10.1, "others": 29.3},
                {"label": "Last Month", "btc": 56.9, "eth": 10.9, "others": 32.2},
            ],
        }
    else:
        st.error("Error al obtener datos del mercado")
        return None


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
    prompt = f"Generate a prediction for the next 8 candles for the token {price_ticker} on a {interval_selectbox} timeframe. The latest opening prices are: {open_prices}, the latest closing prices are: {close_prices}, and the volume was: {volume_data}. Summarize the expected trend and key characteristics (e.g., bullish, bearish, or sideways movement) for the upcoming 8 candles. After the prediction, propose a {temp_prompt} strategy to maximize profits based on this forecast proposing entry and exit prices . Avoid listing the candles individually and provide the response in a concise and actionable format. Exclude titles, additional information, and resources. Include a disclaimer at the end."
    
    return prompt

######NEWS FEED 

def fetch_news(url):
    """Fetch news items from the RSS feed."""
    feed = feedparser.parse(url)
    return feed.entries


def display_news(news_entries):
    """Display the latest 3 news items with styled containers."""
    for i in range(min(5, len(news_entries))):  # Mostrar máximo 5 noticias
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

#####TOP LOSERS
def get_top_losers():
    # Hacer una solicitud GET a la API de Binance para obtener los tickers de las últimas 24 horas
    url = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(url)
    
    # Comprobar que la solicitud fue exitosa
    if response.status_code != 200:
        st.error(f"Error al obtener datos de la API de Binance: {response.status_code}")
        return []

    # Obtener los datos de la respuesta en formato JSON
    tickers = response.json()

    # Filtrar los tickers con un cambio negativo en el precio
    losers = [ticker for ticker in tickers if float(ticker['priceChangePercent']) < 0]

    # Ordenar los tickers por el mayor porcentaje de pérdida (más negativo primero)
    losers_sorted = sorted(losers, key=lambda x: float(x['priceChangePercent']))

    return losers_sorted[:5]  # Top 5 losers

###UPDATE IMAGES - - 
def load_whale_alert():
    response = requests.get(f"https://nqpmedia.com/assets/whale_alert.png?t={int(time.time())}")
    return Image.open(BytesIO(response.content))

def load_liquidations():
    response = requests.get(f"https://nqpmedia.com/assets/liquidations.png?t={int(time.time())}")
    return Image.open(BytesIO(response.content))

def load_gold_btc():
    response = requests.get(f"https://nqpmedia.com/assets/gold_btc.png?t={int(time.time())}")
    return Image.open(BytesIO(response.content))

def load_btc_dom():
    response = requests.get(f"https://nqpmedia.com/assets/btc_dom.png?t={int(time.time())}")
    return Image.open(BytesIO(response.content))

# STREAMLIT Multi Plot Display without Dropdown 

text = '''---'''
st.markdown(text)
# Crear un selector sin dropdown
st.subheader(f'🤖 Create Trading Strategy for 🔵 {price_ticker} on a {interval_selectbox} Chart')
temp_prompt = st.radio("Select a Strategy:", ["Secure", "Moderate", "Aggressive"], horizontal=True)

# Botón para generar la estrategia de trading
if st.button(f'Launch AI Trading Strategy for {price_ticker}'):
    # Construir el prompt usando las selecciones del usuario
    prompt = extract_data_for_prompt(klines_ticker_price, price_ticker, interval_selectbox)
    
    # Obtener la respuesta del modelo
    response = get_chat_response(prompt)
    
    # Mostrar la respuesta
    st.write(response)

 # Botón para copiar al portapapeles
    if st.button("📋 Copy to Clipboard"):
        pyperclip.copy(response)
        st.success("Copied to clipboard!")

text = '''---'''
st.markdown(text)  

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
st.image(load_liquidations(), caption="12 Hr Global Liquidation 🔥")         

text = '''---''' 
st.markdown(text)  

st.subheader(f'🟢 Crypto 🔥😰 Fear and Greed 😤 Index 🔥 ')
st.image("https://alternative.me/crypto/fear-and-greed-index.png", caption="Latest Crypto Fear & Greed Index")

text = '''---'''
st.markdown(text)

# Obtener datos
market_data = get_market_data()

# Mostrar métricas
if market_data:
    st.title("📊 Bitcoin Dominance Overview")

    # Sección de métricas principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Bitcoin Dominance", f"{market_data['btc_dominance']}%",delta_color="inverse")
    col2.metric("Ethereum Dominance", f"{market_data['eth_dominance']}%", delta_color="off")
    col3.metric("Others", f"{market_data['others_dominance']}%",delta_color="off")
st.image(load_btc_dom(), caption="🔵 Bitcoin Dominance 💪 // coinmarketcap.com") 

text = '''---''' 
st.markdown(text) 

st.title(f'🟢  BTC vs GOLD 〽️ Index ')
st.image(load_gold_btc(), caption="🟢Bitcoin Price // 🟡Gold Price per Oz〽️ ")         

   
text = '''---''' 
st.markdown(text) 

st.markdown('<div style="text-align: center; ">NQP_Media_iA_Apps </div>', unsafe_allow_html=True)   
text = '''---'''
st.markdown(text)        
st.markdown('<div style="text-align: right;">Report Issues or Suggestions 📝 info@nqpmedia.com</div>', unsafe_allow_html=True)     


with st.sidebar:
    st.header("🟡 Latest Crypto News 🗞️📰 ")
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

    ## WHALE ALERTS
    
    st.markdown('# 🐋  WHALE ALERTS 🚨')
    st.image(load_whale_alert(), caption="Latest Whale Alerts 🚨")
    

# TOP LOSERS
def show_top_losers():
    top_losers = get_top_losers()  # Suponiendo que tienes una función que obtiene los perdedores

    st.sidebar.markdown('# Top 5 Losers 📉 (24h)')

    if top_losers:
        for loser in top_losers:
            symbol = loser['symbol']
            percent_change = loser['priceChangePercent']
            st.sidebar.markdown(f"""
                <div style="
                    background-color: #eaeaea;
                    border: 3px solid #b71c1c; 
                    border-radius: 10px; 
                    padding: 3px; 
                    text-decoration: none; 
                    font-weight: bold;
                    color: #b71c1c;
                    font-size: 16px;
                    margin-bottom: 10px;">🔴 {symbol}: {percent_change}% 📉
                </div>
                    """, unsafe_allow_html=True)
    else:
        st.sidebar.text("No se pudo obtener los datos de los top losers.")

# Llamamos a la función para mostrar los top losers
show_top_losers()

st.sidebar.write('<div style="text-align: center;">By🎙️_0xdEVbEN_🎸 </div>', unsafe_allow_html=True)


