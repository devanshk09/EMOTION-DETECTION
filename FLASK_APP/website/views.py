from flask import Blueprint, render_template, request, flash,redirect
from flask_login import login_required, current_user
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import re
from collections import Counter
import matplotlib.pyplot as plt
from math import pi
import os
import uuid
from wordcloud import wordcloud
import pandas as pd
import plotly.express as px
import io
import base64

views = Blueprint('views', __name__)

# Load model and tokenizer (done once when app starts)
try:
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    emotion_pipeline = pipeline("text-classification", 
                              model=model, 
                              tokenizer=tokenizer, 
                              return_all_scores=True)
    print("✅ Emotion model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    emotion_pipeline = None

# Disaster synonyms dictionary
disaster_synonyms = {
    "Wildfire": ["fire", "forest fire", "bushfire", "blaze", "inferno","wildfire"],
    "Flood": ["flood", "flooding", "inundation", "deluge", "high water","floods"],
    "Storm": ["storm", "cyclone", "hurricane", "typhoon", "tornado", "thunderstorm"],
    "Drought": ["drought", "dry spell", "water shortage", "arid", "desertification"],
    "Earthquake": ["earthquake", "tremor", "quake", "seismic activity", "aftershock"],
    "Landslide": ["landslide", "mudslide", "avalanche", "rockfall", "slope failure"],
    "Heatwave": ["heatwave", "hot spell", "extreme heat", "global warming", "warming"],
    "Coldwave": ["coldwave", "extreme cold", "freeze", "frost", "polar vortex", "chill"],
    "Famine": ["famine", "starvation", "food shortage", "malnutrition", "hunger crisis"]
}

# Emotion mapping
emotion_mapping = {
    "anger": "anger",
    "disgust": "confusion",
    "fear": "fear",
    "joy": "joy",
    "sadness": "sadness",
    "surprise": "anticipation"
}

# Emoji mapping
top_emotion_emoji = {
    "anger": "😡",
    "confusion": "🤔",
    "fear": "😨",
    "joy": "😄",
    "sadness": "😢",
    "anticipation": "😮",
    "error": "❌",
    "hope": "✨"
}

def extract_disaster(text):
    """Extract the most likely disaster type from text"""
    if not text or not isinstance(text, str):
        return None
    words = re.findall(r'\b\w+\b', text.lower())
    disaster_mentions = []
    for disaster, synonyms in disaster_synonyms.items():
        for synonym in synonyms:
            if synonym in words:
                disaster_mentions.append(disaster)
    return Counter(disaster_mentions).most_common(1)[0][0] if disaster_mentions else None

def predict_emotion(text):
    """Predict emotion with custom rules"""
    if not emotion_pipeline:
        return "error"
    
    if not text or not isinstance(text, str) or text.strip() == "":
        return "hope"
    
    try:
        emotions = emotion_pipeline(text)[0]
        mapped_emotions = {}
        
        for e in emotions:
            label = e['label']
            score = e['score']
            mapped_label = emotion_mapping.get(label, "neutral")
            
            # Custom rules
            if mapped_label == "joy" and score > 0.3:
                mapped_emotions["hope"] = score
            elif mapped_label in ["neutral", "sadness"] and score < 0.5:
                mapped_emotions["hope"] = score
            elif mapped_label in ["fear", "anger"] and score < 0.6:
                mapped_emotions["confusion"] = score
            elif label == "disgust":
                mapped_emotions["confusion"] = score
            else:
                mapped_emotions[mapped_label] = score
        
        return max(mapped_emotions.items(), key=lambda x: x[1])[0]
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return "error"

def generate_pie_chart(emotion_scores):
    """Generate pie chart of emotion distribution"""
    plt.figure(figsize=(6, 6))
    labels = emotion_scores.keys()
    values = emotion_scores.values()
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    plt.title("Emotion Distribution")
    pie_filename = f"pie_{uuid.uuid4().hex}.png"
    filepath = os.path.join("website", "static", pie_filename)
    plt.savefig(filepath)
    plt.close()
    return pie_filename

def generate_radar_chart(emotion_scores):
    """Generate radar chart of emotion scores"""
    labels = list(emotion_scores.keys())
    values = list(emotion_scores.values())
    values += values[:1]
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], labels)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.4)
    plt.title("Emotion Radar Chart")

    radar_filename = f"radar_{uuid.uuid4().hex}.png"
    filepath = os.path.join("website", "static", radar_filename)
    plt.savefig(filepath)
    plt.close()
    return radar_filename


def generate_bar_chart(emotion_scores):
    """Generate bar chart of emotion scores"""
    plt.figure(figsize=(8, 5))
    labels = list(emotion_scores.keys())
    values = list(emotion_scores.values())
    bars = plt.bar(labels, values, color='skyblue')
    plt.title("Emotion Scores")
    plt.ylabel("Score")
    plt.xlabel("Emotion")
    plt.ylim(0, 1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.25, yval + 0.01, f"{yval:.2f}")

    bar_filename = f"bar_{uuid.uuid4().hex}.png"
    filepath = os.path.join("website", "static", bar_filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return bar_filename

@views.route('/')
@login_required
def blank():
    return render_template("blank.html", user=current_user)

@views.route('/emotion-analyzer', methods=['GET', 'POST'])
@login_required
def emotion_analyzer():
    if request.method == 'POST':
        user_input = request.form.get("user_input", "").strip()
        
        if not user_input:
            flash("Please enter some text", category='error')
            return render_template("analyzer.html", user=current_user)
        
        predicted_emotion = predict_emotion(user_input)
        disaster_type = extract_disaster(user_input)
        emoji = top_emotion_emoji.get(predicted_emotion, "")
        
        # Get all emotion scores for charts
        emotion_scores = {}
        if emotion_pipeline:
            try:
                emotions = emotion_pipeline(user_input)[0]
                for e in emotions:
                    label = e['label']
                    score = e['score']
                    mapped_label = emotion_mapping.get(label, "neutral")
                    emotion_scores[mapped_label] = emotion_scores.get(mapped_label, 0) + score
                
                # Generate charts
                pie_chart = generate_pie_chart(emotion_scores)
                radar_chart = generate_radar_chart(emotion_scores)
                bar_chart = generate_bar_chart(emotion_scores)
            except Exception as e:
                print(f"Chart generation error: {e}")
                pie_chart = None
                radar_chart = None
                bar_chart = None
        else:
            pie_chart = None
            radar_chart = None
            bar_chart = None
        
        return render_template("analyzer.html",
                            user=current_user,
                            user_input=user_input,
                            predicted_emotion=predicted_emotion,
                            disaster_type=disaster_type,
                            emoji=emoji,
                            emotion_scores=emotion_scores,
                            pie_chart=pie_chart,
                            radar_chart=radar_chart,
                            bar_chart=bar_chart)
    
    return render_template("analyzer.html", user=current_user)

@views.route('/predicted-results', methods=['GET', 'POST'])
@login_required
def predicted_results():
    plot_div = None
    wordcloud_img = None
    data = None
    
    if request.method == 'POST':
        if 'reddit_file' not in request.files:
            flash('No file part', category='error')
            return redirect(request.url)
        
        file = request.files['reddit_file']
        
        if file.filename == '':
            flash('No selected file', category='error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            try:
                # Read CSV file
                df = pd.read_csv(file)
                
                # Ensure required columns exist
                required_columns = ['Comments', 'Natural Disaster', 'Predicted Emotion', 'Country', 'Continent']
                for col in required_columns:
                    if col not in df.columns:
                        flash(f"CSV file is missing required column: {col}", category='error')
                        return redirect(request.url)
                
                # Clean data
                df = df.dropna(subset=['Comments', 'Predicted Emotion'])
                df['Natural Disaster'] = df['Natural Disaster'].fillna('Unknown')
                df['Country'] = df['Country'].fillna('Unknown')
                df['Continent'] = df['Continent'].fillna('Unknown')
                
                # Get plot type from form
                plot_type = request.form.get('plot_type')
                
                if plot_type == 'sunburst_emotion_country':
                    fig = px.sunburst(
                        df,
                        path=['Continent', 'Country', 'Predicted Emotion'],
                        title='Emotion Distribution by Country'
                    )
                    plot_div = fig.to_html(full_html=False)
                
                elif plot_type == 'sunburst_disaster_country':
                    fig = px.sunburst(
                        df,
                        path=['Continent', 'Country', 'Natural Disaster'],
                        title='Disaster Types by Country'
                    )
                    plot_div = fig.to_html(full_html=False)
                
                elif plot_type == 'sunburst_continent':
                    fig = px.sunburst(
                        df,
                        path=['Continent', 'Country','Natural Disaster', 'Predicted Emotion'],
                        title='Emotion due to Natural Disaster for each country'
                    )
                    plot_div = fig.to_html(full_html=False)
                
                elif plot_type == 'choropleth':

                    # Get most frequent emotion per country
                    top_emotion = (
                        df.groupby(['Country', 'Predicted Emotion'])
                        .size()
                        .reset_index(name='Emotion Count')
                        .sort_values(['Country', 'Emotion Count'], ascending=[True, False])
                        .drop_duplicates('Country')
                        .rename(columns={'Predicted Emotion': 'Top Emotion'})
                    )

                    # Get most frequent disaster per country
                    top_disaster = (
                        df.groupby(['Country', 'Natural Disaster'])
                        .size()
                        .reset_index(name='Disaster Count')
                        .sort_values(['Country', 'Disaster Count'], ascending=[True, False])
                        .drop_duplicates('Country')
                        .rename(columns={'Natural Disaster': 'Top Disaster'})
                    )

                    # Merge both
                    merged = pd.merge(top_emotion[['Country', 'Top Emotion']], top_disaster[['Country', 'Top Disaster']], on='Country', how='inner')

                    # Create hover text
                    merged['hover_text'] = merged['Country'] + '<br>Top Emotion: ' + merged['Top Emotion'] + '<br>Top Disaster: ' + merged['Top Disaster']

                    # Create map using dummy color (since we're only showing info via hover)
                    fig = px.choropleth(
                        merged,
                        locations='Country',
                        locationmode='country names',
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        color=merged['Top Emotion'],  # Just to color countries differently
                        hover_name='Country',
                        hover_data={'Top Emotion': True, 'Top Disaster': True, 'Country': False},
                        title='Most Frequent Emotion and Natural Disaster by Country'
                    )

                    plot_div = fig.to_html(full_html=False)
                    
                data = df.head(10).to_dict('records')
                
            except Exception as e:
                flash(f"Error processing file: {str(e)}", category='error')
                print(f"Error processing file: {e}")
    
    return render_template(
        "predicted_results.html",
        user=current_user,
        plot_div=plot_div,
        wordcloud_img=wordcloud_img,
        data=data)