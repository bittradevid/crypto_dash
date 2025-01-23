import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Función para calcular los indicadores
def calculate_indicators(data, length_short=50, length_long=50):
    data['StopLossLevelShort'] = data['High'].rolling(window=length_short).max()
    data['StopLossLevelLong'] = data['Low'].rolling(window=length_long).min()
    return data

# Función para graficar los datos con los indicadores
def plot_with_indicators(data):
    fig = go.Figure()

    # Línea del precio de cierre
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close', line=dict(color='blue')))

    # Indicador de Stop Loss para cortos
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['StopLossLevelShort'],
        name='Shorts Stop Loss',
        line=dict(color='red', width=2)
    ))

    # Indicador de Stop Loss para largos
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['StopLossLevelLong'],
        name='Longs Stop Loss',
        line=dict(color='green', width=2)
    ))

    # Configuración del layout
    fig.layout.update(
        title="Price with Stop Loss Indicators",
        xaxis_rangeslider_visible=True,
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig)

# Ejemplo de datos (reemplázalo con tus datos reales)
# Crear un DataFrame con datos ficticios
data = pd.DataFrame({
    'Date': pd.date_range(start='2024-01-01', periods=100),
    'High': pd.Series([x * 1.05 for x in range(1, 101)]),  # Datos simulados
    'Low': pd.Series([x * 0.95 for x in range(1, 101)]),
    'Close': pd.Series(range(1, 101))
})

# Calcular los indicadores
data = calculate_indicators(data)

# Mostrar los datos con el indicador graficado
plot_with_indicators(data)
