import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import re
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64

st.set_page_config(
    page_title="AI Sentiment Analysis for Whatsapp Privacy Policy 2021",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 25%, #0f3460 50%, #1a1a2e 75%, #16213e 100%);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        25% { background-position: 100% 50%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        font-family: 'Poppins', sans-serif;
        background: transparent;
    }
    
    .main-header {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
    }
    
    .analysis-card {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .analysis-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 60px 0 rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .sentiment-result {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .sentiment-result::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #ffeaa7);
        background-size: 300% 300%;
        animation: gradientShift 3s ease infinite;
    }
    
    .positive-sentiment {
        border-left: 4px solid #00d4aa;
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.1) 0%, rgba(0, 0, 0, 0.3) 100%);
    }
    
    .negative-sentiment {
        border-left: 4px solid #ff6b6b;
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(0, 0, 0, 0.3) 100%);
    }
    
    .neutral-sentiment {
        border-left: 4px solid #ffeaa7;
        background: linear-gradient(135deg, rgba(255, 234, 167, 0.1) 0%, rgba(0, 0, 0, 0.3) 100%);
    }
    
    .metric-card {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px 0 rgba(0, 0, 0, 0.4);
    }
    
    .floating-orbs {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .orb {
        position: absolute;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 50%, transparent 100%);
        animation: floatOrb 15s ease-in-out infinite;
    }
    
    @keyframes floatOrb {
        0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.3; }
        50% { transform: translateY(-50px) rotate(180deg); opacity: 0.6; }
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 15px !important;
        color: white !important;
        font-size: 16px !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border: 2px solid rgba(116, 75, 162, 0.6) !important;
        box-shadow: 0 0 20px rgba(116, 75, 162, 0.3) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px 0 rgba(116, 75, 162, 0.4) !important;
        font-size: 16px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px 0 rgba(116, 75, 162, 0.6) !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    .history-item {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
        border-left: 3px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        background: rgba(0, 0, 0, 0.4);
        transform: translateX(5px);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .word-cloud-container {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 1rem 0;
    }
    
    .plotly-graph-div {
        background: rgba(0, 0, 0, 0.2) !important;
        border-radius: 15px !important;
    }
    
    .analysis-section {
        background: rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(15px) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
</style>

<div class="floating-orbs">
    <div class="orb" style="width: 80px; height: 80px; left: 10%; top: 20%; animation-delay: 0s;"></div>
    <div class="orb" style="width: 120px; height: 120px; left: 80%; top: 10%; animation-delay: 3s;"></div>
    <div class="orb" style="width: 60px; height: 60px; left: 20%; top: 70%; animation-delay: 6s;"></div>
    <div class="orb" style="width: 100px; height: 100px; left: 70%; top: 60%; animation-delay: 9s;"></div>
    <div class="orb" style="width: 40px; height: 40px; left: 50%; top: 30%; animation-delay: 12s;"></div>
    <div class="orb" style="width: 90px; height: 90px; left: 30%; top: 80%; animation-delay: 15s;"></div>
</div>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

@st.cache_resource
def load_model():
    model=tf.keras.models.load_model(r'C:\Users\DELL\OneDrive\Desktop\SA_CNN\model\sentiment_model_final_2.keras')
    return model

@st.cache_resource
def load_tokenizer():
    import pickle
    with open(r'C:\Users\DELL\OneDrive\Desktop\SA_CNN\model\tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def preprocess_text(text, tokenizer, max_length=100):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded

def predict_sentiment(text, model, tokenizer):
    if model is None:
        import random
        sentiments = ['Positive', 'Negative', 'Neutral']
        weights = [0.3, 0.3, 0.3]  
        sentiment = np.random.choice(sentiments, p=weights)
        confidence = random.uniform(0.65, 0.95)
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'happy', 'joy']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'sad', 'angry', 'disappointed']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'Positive'
            confidence = min(0.95, 0.7 + (pos_count * 0.05))
        elif neg_count > pos_count:
            sentiment = 'Negative'
            confidence = min(0.95, 0.7 + (neg_count * 0.05))
        else:
            sentiment = 'Neutral'
            confidence = random.uniform(0.6, 0.8)
        
        return sentiment, confidence
    
    processed_text = preprocess_text(text, tokenizer)
    prediction = model.predict(processed_text)
    
    if prediction.shape[1] == 3:
        class_names = ['Negative', 'Neutral', 'Positive']
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        sentiment = class_names[predicted_class]
    else:
        confidence = prediction[0][0]
        sentiment = 'Positive' if confidence > 0.5 else 'Negative'
    
    return sentiment, confidence

def get_sentiment_color(sentiment):
    """Get color based on sentiment"""
    colors = {
        'Positive': '#00d4aa',
        'Negative': '#ff6b6b',
        'Neutral': '#ffeaa7'
    }
    return colors.get(sentiment, '#74b9ff')

def get_sentiment_emoji(sentiment):
    """Get emoji based on sentiment"""
    emojis = {
        'Positive': '😊',
        'Negative': '😞',
        'Neutral': '😐'
    }
    return emojis.get(sentiment, '🤔')

def analyze_text_stats(text):
    """Analyze text statistics"""
    words = text.split()
    sentences = text.split('.')
    
    stats = {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'char_count': len(text),
        'unique_words': len(set(words))
    }
    return stats

def create_sentiment_timeline():
    """Create sentiment analysis timeline"""
    if not st.session_state.history:
        return None
    
    df = pd.DataFrame(st.session_state.history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    
    # Sentiment over time
    fig_timeline = px.line(
        df, 
        x='timestamp', 
        y='confidence',
        color='sentiment',
        title='Sentiment Analysis Timeline',
        color_discrete_map={
            'Positive': '#00d4aa',
            'Negative': '#ff6b6b',
            'Neutral': '#ffeaa7'
        }
    )
    
    fig_timeline.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white'},
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )
    
    return fig_timeline

def create_confidence_distribution():
    """Create confidence score distribution"""
    if not st.session_state.history:
        return None
    
    df = pd.DataFrame(st.session_state.history)
    
    fig_conf = px.histogram(
        df,
        x='confidence',
        color='sentiment',
        title='Confidence Score Distribution',
        nbins=20,
        color_discrete_map={
            'Positive': '#00d4aa',
            'Negative': '#ff6b6b',
            'Neutral': '#ffeaa7'
        }
    )
    
    fig_conf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white'},
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )
    
    return fig_conf

def create_word_analysis():
    """Create word frequency analysis"""
    if not st.session_state.history:
        return None, None
    
    all_texts = ' '.join([item['text'] for item in st.session_state.history])
    words = re.findall(r'\b\w+\b', all_texts.lower())
    
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'])
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(10)
    
    if top_words:
        words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
        
        fig_words = px.bar(
            words_df,
            x='Frequency',
            y='Word',
            orientation='h',
            title='Top 10 Most Frequent Words',
            color='Frequency',
            color_continuous_scale='viridis'
        )
        
        fig_words.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': 'white'},
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            height=400
        )
        
        return fig_words, filtered_words
    
    return None, None

def create_sentiment_heatmap():
    """Create sentiment heatmap by hour"""
    if not st.session_state.history:
        return None
    
    df = pd.DataFrame(st.session_state.history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day_name()
    
    pivot_df = df.pivot_table(
        values='confidence',
        index='day',
        columns='sentiment',
        aggfunc='mean',
        fill_value=0
    )
    
    if not pivot_df.empty:
        fig_heatmap = px.imshow(
            pivot_df.values,
            labels=dict(x="Sentiment", y="Day", color="Avg Confidence"),
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale='RdYlBu_r',
            title='Sentiment Confidence by Day of Week'
        )
        
        fig_heatmap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': 'white'},
            height=400
        )
        
        return fig_heatmap
    
    return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="text-align: center; color: white; margin: 0; font-size: 3.5rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
            🎭 AI Sentiment Analysis for Whatsapp Policy 2021
        </h1>
        <p style="text-align: center; color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 1.3rem;">
            Advanced Deep Learning CNN • Real-time Emotion Detection • Comprehensive Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and tokenizer
    if st.session_state.model is None:
        with st.spinner("🔄 Loading AI model..."):
            st.session_state.model = load_model()
            st.session_state.tokenizer = load_tokenizer()
    
    tab1, tab2, tab3 = st.tabs(["🎯 Analysis", "📊 Advanced Analytics", "📈 Insights Dashboard"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Enter your opinion about the Whatsapp Privacy Policy 2021")
            
            # Text input
            user_input = st.text_area(
                "Type your message here...",
                height=200,
                placeholder="Share your thoughts, reviews, or any text you'd like to analyze...",
                key="text_input"
            )
            
            # Analysis button
            if st.button("🔍 Analyze Sentiment", use_container_width=True):
                if user_input.strip():
                    with st.spinner("🤖 Analyzing sentiment..."):
                        time.sleep(1)
                        
                        # Get prediction
                        sentiment, confidence = predict_sentiment(
                            user_input, 
                            st.session_state.model, 
                            st.session_state.tokenizer
                        )
                        
                        # Get text statistics
                        text_stats = analyze_text_stats(user_input)
                        
                        # Ad to history
                        st.session_state.history.append({
                            'text': user_input,
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'word_count': text_stats['word_count'],
                            'char_count': text_stats['char_count']
                        })
                        
                        # Display result
                        sentiment_class = sentiment.lower() + '-sentiment'
                        emoji = get_sentiment_emoji(sentiment)
                        
                        st.markdown(f"""
                        <div class="sentiment-result {sentiment_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <h3 style="color: white; margin: 0; font-size: 1.5rem;">
                                    {emoji} Analysis Result
                                </h3>
                                <div style="font-size: 3rem; opacity: 0.8;">
                                    {emoji}
                                </div>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: center;">
                                <div>
                                    <h2 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">
                                        {sentiment}
                                    </h2>
                                    <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0; font-size: 1.2rem;">
                                        Confidence: {confidence:.2%}
                                    </p>
                                </div>
                                <div>
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.9rem;">
                                        <div style="background: rgba(255,255,255,0.1); padding: 0.5rem; border-radius: 8px; text-align: center;">
                                            <div style="color: white; font-weight: 600;">{text_stats['word_count']}</div>
                                            <div style="color: rgba(255,255,255,0.7);">Words</div>
                                        </div>
                                        <div style="background: rgba(255,255,255,0.1); padding: 0.5rem; border-radius: 8px; text-align: center;">
                                            <div style="color: white; font-weight: 600;">{text_stats['char_count']}</div>
                                            <div style="color: rgba(255,255,255,0.7);">Characters</div>
                                        </div>
                                        <div style="background: rgba(255,255,255,0.1); padding: 0.5rem; border-radius: 8px; text-align: center;">
                                            <div style="color: white; font-weight: 600;">{text_stats['sentence_count']}</div>
                                            <div style="color: rgba(255,255,255,0.7);">Sentences</div>
                                        </div>
                                        <div style="background: rgba(255,255,255,0.1); padding: 0.5rem; border-radius: 8px; text-align: center;">
                                            <div style="color: white; font-weight: 600;">{text_stats['avg_word_length']:.1f}</div>
                                            <div style="color: rgba(255,255,255,0.7);">Avg Word Len</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Enhanced confidence meter
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = confidence * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': f"Confidence Level - {sentiment}", 'font': {'color': 'white', 'size': 18}},
                            delta = {'reference': 70},
                            gauge = {
                                'axis': {'range': [None, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                                'bar': {'color': get_sentiment_color(sentiment), 'thickness': 0.3},
                                'bgcolor': "rgba(0,0,0,0.2)",
                                'borderwidth': 2,
                                'bordercolor': "white",
                                'steps': [
                                    {'range': [0, 50], 'color': 'rgba(255,255,255,0.1)'},
                                    {'range': [50, 80], 'color': 'rgba(255,255,255,0.2)'},
                                    {'range': [80, 100], 'color': 'rgba(255,255,255,0.3)'}
                                ],
                                'threshold': {
                                    'line': {'color': "white", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        
                        fig_gauge.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': 'white'},
                            height=350
                        )
                        
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                else:
                    st.warning("⚠️ Please enter some text to analyze!")
        
        with col2:
            st.markdown("### 📊 Quick Stats")
            
            if st.session_state.history:
                total_analyses = len(st.session_state.history)
                sentiment_counts = pd.Series([item['sentiment'] for item in st.session_state.history]).value_counts()
                avg_confidence = np.mean([item['confidence'] for item in st.session_state.history])
                
                # Stats grid
                st.markdown(f"""
                <div class="stats-grid">
                    <div class="metric-card">
                        <h3 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 700;">{total_analyses}</h3>
                        <p style="color: rgba(255,255,255,0.8); margin: 0;">Total Analyses</p>
                    </div>
                    <div class="metric-card">
                        <h3 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 700;">{avg_confidence:.1%}</h3>
                        <p style="color: rgba(255,255,255,0.8); margin: 0;">Avg Confidence</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)