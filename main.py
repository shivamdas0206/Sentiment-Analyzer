import string
import streamlit as st
from collections import Counter 
import nltk

# Explicitly download punkt_tab
try:
    nltk.download('punkt_tab')
except:
    nltk.download('punkt')

# Download other required NLTK resources
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Now import the NLTK modules after downloading resources
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pathlib
import numpy as np

from PIL import Image 
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import json

# Load Lottie JSON files
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_happy = load_lottiefile("assets/happy1.json")
lottie_sad = load_lottiefile("assets/sad1.json")
lottie_neutral = load_lottiefile("assets/neutral1.json")

# Transparent canvas for Lottie
st.markdown("""
<style>
div[class^='stLottie'] canvas {
    background-color: rgba(0,0,0,0) !important;
}
</style>
""", unsafe_allow_html=True)

# Adjust Lottie animation position
st.markdown("""
<style>
div[class^='stLottie'] {
    position: relative;
    top: -50px;  /* Adjust this value as needed */
}
</style>
""", unsafe_allow_html=True)

# Load CSS
def load_css(file_path):
    with open(file_path) as f:
        st.html(f"<style>{f.read()}</style>")

# Load image
img = Image.open("Frame 34.png")
col1, col2, col3 = st.columns([1, 5, 1])
with col2:
    st.image(img, width=500, channels="RGB")

# Load custom CSS file
css_path = pathlib.Path("assets/style.css")
load_css(css_path)

# Font styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Jersey+10&display=swap');

html, body, [class*="css"]  {
    font-family: 'Jersey 10', cursive !important;
    font-size: 20px !important;
}

h1, h2, h3, h4, h5, h6,
div, span, p, label, button, input, textarea {
    font-family: 'Jersey 10', cursive !important;
}

input::placeholder {
    color: white;
    font-size: 30px;
}

.stTextInput input {
    font-family: 'Jersey 10', cursive !important;
    font-size: 30px !important;
    color: white;
    border: 4px solid white;
    border-radius: 10px;
}

.stTextInput input:focus {
    outline: none;
    box-shadow: 0 0 5px 2px #A0E7FF;
    border-color: white;
}

.center-input > div {
    display: flex;
    justify-content: center;
}

div[data-baseweb="input"] > div {
    background-color: #3DB5FF;
    border-radius: 40px;
    text-align: center;
}

input {
    text-align: center;
    font-size: 50px;
}
</style>
""", unsafe_allow_html=True)

# Text input
st.markdown('<div class="center-input">', unsafe_allow_html=True)
text = st.text_input(" ", placeholder="Enter text to analyze sentiment:\n", key="textinput")
st.markdown('</div>', unsafe_allow_html=True)

# Default to empty string if no text is entered
if not text:
    text = ""

# Preprocessing
lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# Use word_tokenize with default language (English)
tokenized_words = word_tokenize(cleaned_text)
final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

# Emotion extraction
emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        if word in final_words:
            emotion_list.append(emotion)

w = Counter(emotion_list)

# Sentiment analysis
sentiment_score = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)
neg = sentiment_score['neg']
pos = sentiment_score['pos']
compound = sentiment_score['compound']

if neg > pos:
    sentiment_label = "Negative"
    sentiment_color = "red"
    sentiment_lottie = lottie_sad
elif pos > neg:
    sentiment_label = "Positive"
    sentiment_color = "green"
    sentiment_lottie = lottie_happy
else:
    sentiment_label = "Neutral"
    sentiment_color = "orange"
    sentiment_lottie = lottie_neutral

# Show sentiment + pie chart side-by-side equally
if sum(w.values()) > 0:
    # Overall Sentiment (centered)
    st.markdown(f"""
    <div style='text-align: center; font-family: "Jersey 10", cursive; font-size: 32px; color: white; height: 100px;' id="sentiment_label">
        Overall Sentiment: <span style='color:{sentiment_color};'>{sentiment_label}</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 3])  # Column layout for sentiment score, emoji, and pie chart
    
    # Left: Sentiment Scores (Negative, Positive, Compound)
    with col1:
        st.markdown(f"""
        <div style='text-align: center; font-family: "Jersey 10", cursive; font-size: 24px; color: white;' id="sentiment_scores">
            Negative Score: <span style='color: red;'>{neg:.2f}</span><br>
            Positive Score: <span style='color: green;'>{pos:.2f}</span><br>
            Compound Score: <span style='color: orange;'>{compound:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Center: Lottie Animation (emoji)
    with col2:
        st_lottie(sentiment_lottie, height=200, key="overall_sentiment", speed=1)

    # Right: Pie Chart with Emotion Breakdown
    with col3:
        st.markdown("<div style='text-align: center; color: white; font-size: 40px; margin-bottom: -30px;'>Emotion Pie ! ! !</div>", unsafe_allow_html=True)

        emotion_df = pd.DataFrame({'Emotion': list(w.keys()), 'Count': list(w.values())})
        labels = emotion_df['Emotion'].tolist()
        target_values = emotion_df['Count'].tolist()
        steps = 50

        combined = sorted(zip(labels, target_values), key=lambda x: x[0])
        labels = [item[0] for item in combined]
        target_values = [item[1] for item in combined]

        base_colors = px.colors.qualitative.Plotly[:len(labels)]
        color_map = {label: color for label, color in zip(labels, base_colors)}

        chart_placeholder = st.empty()
        for step in range(1, steps + 1):
            progress = step / steps
            animated_counts = [v * progress for v in target_values]

            if step < steps:
                temp_labels = labels + ['']
                remaining = sum(target_values) - sum(animated_counts)
                temp_counts = animated_counts + [max(0.01, remaining)]
            else:
                temp_labels = labels
                temp_counts = target_values

            temp_df = pd.DataFrame({
                'Emotion': temp_labels,
                'Count': temp_counts
            })

            colors = [color_map.get(label, 'rgba(0,0,0,0)') for label in temp_labels]

            fig = go.Figure(data=[go.Pie(
                labels=temp_df['Emotion'],
                values=temp_df['Count'],
                hole=0.3,
                sort=False,
                marker=dict(colors=colors),
                textinfo='label+percent' if step == steps else 'none',
                textposition='inside',
                insidetextorientation='radial',
                textfont=dict(color='white', size=14, family='Jersey 10'),
                pull=[0.01] * len(temp_df),
                hovertemplate='%{label}<extra></extra>'
            )])

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Jersey 10'),
                title=dict(text=None),
                width=600,
                height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=False,
                autosize=False
            )

            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.025)

    # JavaScript to scroll to the chart after input
    st.components.v1.html("""
    <script>
        window.onload = function() {
            const sentimentLabel = document.getElementById('sentiment_label');
            if (sentimentLabel) {
                sentimentLabel.scrollIntoView({behavior: 'smooth'});
            }
        };
    </script>
    """, height=0)
else:
    if text:  # Only show sentiment if text was entered
        st.markdown(f"""
        <div style='text-align: center; font-family: "Jersey 10", cursive"; font-size: 32px; color: white;' id="sentiment_label">
            Overall Sentiment: <span style='color:{sentiment_color};'>{sentiment_label}</span>
        </div>
        """, unsafe_allow_html=True)
        st_lottie(sentiment_lottie, height=300, key="overall_sentiment_only", speed=1)
        st.write("Not enough emotional content to generate a pie chart.")
