import numpy as np
from helpers import *
import streamlit as st
import plotly.graph_objects as go

# --- Streamlit App Input & Layout --- 
# Title
st.title("Volatility Term Structure")
st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)

# Sidebar 
st.sidebar.header("Option Inputs")

# Ticker
if "term_ticker" not in st.session_state:
    st.session_state.term_ticker = "VOO"
ticker = st.sidebar.text_input("Ticker:", value=st.session_state.term_ticker).upper()
if ticker:
    st.session_state.term_ticker = ticker

# Forward Period (days)
if "term_frwd_period" not in st.session_state:
    st.session_state.term_frwd_period = 15
frwd_period = st.sidebar.number_input("Forward Period (days):", 
                                      min_value=1,
                                      value=st.session_state.term_frwd_period)
st.session_state.term_frwd_period = frwd_period

# Sidebar 
st.sidebar.header("Dashboard Settings")

# Forward Period Validation
if frwd_period < 1:
    st.sidebar.error("Forward period must be larger than one...")

# Start Day
if "term_start_day" not in st.session_state:
    st.session_state.term_start_day = 50
start_day = st.sidebar.number_input("Start Day:", 
                                   min_value=0, 
                                   value=st.session_state.term_start_day)
st.session_state.term_start_day = start_day

# End Day
if "term_end_day" not in st.session_state:
    st.session_state.term_end_day = 300
end_day = st.sidebar.number_input("End Day:", 
                                 min_value=3, 
                                 value=st.session_state.term_end_day)
st.session_state.term_end_day = end_day

# Day Range Validation
if end_day < start_day:
    st.sidebar.error("End day must be larger than start day...")

# ATM Percent Band 
if "term_pct_band" not in st.session_state:
    st.session_state.term_pct_band = 0.05
term_percent_band_display = st.sidebar.slider("ATM Band:",
                                         min_value=5,
                                         max_value=15,
                                         step=1,
                                         value=int(st.session_state.term_pct_band * 100),
                                         format="%d%%")
pct_band = term_percent_band_display / 100.0
st.session_state.term_pct_band = pct_band


# --- Streamlit App Ouput --- 
if all(v is not None for v in [ticker, pct_band, frwd_period]):
    col1, col2 = st.columns([3,3])

    vol = Volatility(ticker=ticker, pct_band=pct_band, frwd_period=int(frwd_period))

    spot_call_iv_ls, spot_put_iv_ls = vol.spot_iv()
    expiration_days_ls = vol.cutoff_expiration_days_dates()[1]

    frwd_call_iv_dict, frwd_put_iv_dict = vol.forward_iv()
    frwd_call_iv_ls = list(frwd_call_iv_dict.values())
    frwd_put_iv_ls = list(frwd_put_iv_dict.values())
    frwd_day_ls = list(frwd_call_iv_dict.keys())

    filtered_spot_days, spot_indices = day_filter_with_indices(expiration_days_ls, start_day, end_day)
    filtered_frwd_days, frwd_indices = day_filter_with_indices(frwd_day_ls, start_day, end_day)

    spot_call_iv_ls = [spot_call_iv_ls[i] for i in spot_indices]
    spot_put_iv_ls = [spot_put_iv_ls[i] for i in spot_indices]

    frwd_call_iv_ls = [frwd_call_iv_ls[i] for i in frwd_indices]
    frwd_put_iv_ls = [frwd_put_iv_ls[i] for i in frwd_indices]

    spot_day_ls = filtered_spot_days
    frwd_call_day_ls = filtered_frwd_days
    frwd_put_day_ls = filtered_frwd_days

    # Call Term Structure
    with col1:
        all_call_iv = np.concatenate((spot_call_iv_ls, frwd_call_iv_ls))
        max_call_iv = np.max(all_call_iv)
        min_call_iv = np.min(all_call_iv)
        step = (max_call_iv - min_call_iv) / len(all_call_iv) if max_call_iv != min_call_iv else 0.01
        master_iv = [max_call_iv - step * i for i in range(0, len(all_call_iv), 4)]
        y_tick_labels = [f"{round(iv*100,1)}%" for iv in master_iv]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_day_ls, y=spot_call_iv_ls, mode='lines+markers', 
                                name='Spot IV', line=dict(color='#A3A8B4'), marker=dict(symbol='circle')))
        fig.add_trace(go.Scatter(x=frwd_call_day_ls, y=frwd_call_iv_ls, mode='lines+markers', 
                                name='Forward IV', line=dict(color='#FF4B4B'), marker=dict(symbol='square')))
        
        fig.update_layout(
            title=dict(text=f"Call IV Term Structure for ATM Options on {ticker}", 
                      font=dict(size=18, color='white')),
            xaxis=dict(title='Days til Expiry', color='white', gridcolor='rgba(255,255,255,0.3)'),
            yaxis=dict(title='Implied Volatility', color='white', gridcolor='rgba(255,255,255,0.3)',
                      tickvals=master_iv, ticktext=y_tick_labels),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), legend=dict(font=dict(color='white')),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    # Put Term Structure
    with col2:      
        all_put_iv = np.concatenate((spot_put_iv_ls, frwd_put_iv_ls))
        max_put_iv = np.max(all_put_iv)
        min_put_iv = np.min(all_put_iv)
        step = (max_put_iv - min_put_iv) / len(all_put_iv) if max_put_iv != min_put_iv else 0.01
        master_iv = [max_put_iv - step * i for i in range(0, len(all_put_iv), 4)]
        y_tick_labels = [f"{round(iv*100,1)}%" for iv in master_iv]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_day_ls, y=spot_put_iv_ls, mode='lines+markers', 
                                name='Spot IV', line=dict(color='#A3A8B4'), marker=dict(symbol='circle')))
        fig.add_trace(go.Scatter(x=frwd_put_day_ls, y=frwd_put_iv_ls, mode='lines+markers', 
                                name='Forward IV', line=dict(color='#FF4B4B'), marker=dict(symbol='square')))
        
        fig.update_layout(
            title=dict(text=f"Put IV Term Structure for ATM Options on {ticker}", 
                      font=dict(size=18, color='white')),
            xaxis=dict(title='Days til Expiry', color='white', gridcolor='rgba(255,255,255,0.3)'),
            yaxis=dict(title='Implied Volatility', color='white', gridcolor='rgba(255,255,255,0.3)',
                      tickvals=master_iv, ticktext=y_tick_labels),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), legend=dict(font=dict(color='white')),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)