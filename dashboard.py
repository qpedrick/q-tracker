import streamlit as st
import pandas as pd
import sqlite3
import time
import plotly.express as px
import os
from PIL import Image

# --- CONFIG ---
DB_FILE = "queue_metrics.db"
IMG_FILE = "latest_debug.jpg"
REFRESH_RATE = 1  # Faster refresh for visual feedback

st.set_page_config(
    page_title="Queue Tracker Live",
    page_icon="🚶",
    layout="wide"
)

# --- TITLE ---
st.title("🚶 Live Queue Analytics")
st.markdown(f"**Device:** Jetson Orin Nano | **Status:** Running")

# --- LAYOUT ---
# Create two main columns: Metrics/Chart on Left, Live View on Right
left_col, right_col = st.columns([2, 1])

# --- PLACEHOLDERS ---
with left_col:
    metric_row = st.container()
    chart_row = st.empty()
    raw_data_row = st.expander("View Raw Data Log", expanded=False)

with right_col:
    st.subheader("👁️ Computer Vision Feed")
    image_placeholder = st.empty()

def load_data():
    """Fetch the last 100 records from the DB"""
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT 100", conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values(by='timestamp')
        return df
    except Exception as e:
        # st.error(f"Waiting for DB... ({e})")
        return pd.DataFrame()

# --- MAIN LOOP ---
while True:
    # 1. Update Metrics & Charts
    df = load_data()
    
    if not df.empty:
        latest_count = df.iloc[-1]['count']
        latest_time = df.iloc[-1]['timestamp']
        avg_count = df['count'].mean()

        with metric_row:
            metric_row.empty() 
            c1, c2, c3 = metric_row.columns(3)
            
            c1.metric(
                label="Queue Length", 
                value=f"{latest_count}",
                delta=f"{latest_count - avg_count:.1f} avg"
            )
            
            # Heuristic: 30s per person
            est_wait = latest_count * 30 
            c2.metric(
                label="Est. Wait", 
                value=f"{est_wait // 60}m {est_wait % 60}s"
            )
            
            c3.metric(
                label="Last Update",
                value=latest_time.strftime("%H:%M:%S")
            )

        fig = px.line(df, x='timestamp', y='count', title="Queue History (Last 100)", markers=True)
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        chart_row.plotly_chart(fig, use_container_width=True)

        with raw_data_row:
            raw_data_row.empty()
            st.dataframe(df.sort_values(by='timestamp', ascending=False).head(5), use_container_width=True)

    # 2. Update Live Image
    if os.path.exists(IMG_FILE):
        try:
            # We open as Image to avoid file lock issues
            image = Image.open(IMG_FILE)
            image_placeholder.image(image, caption="Live Inference View (Updated on Log)", use_container_width=True)
        except Exception:
            pass # Image might be writing, skip this frame

    time.sleep(REFRESH_RATE)