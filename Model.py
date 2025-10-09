from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
from transformers import pipeline, AutoTokenizer        
from tqdm import tqdm
import re
from collections import Counter

model_name = "j-hartmann/emotion-english-distilroberta-base"

# Download the model and tokenizer before running the pipeline
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Now use it in pipeline
emotion_model = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Define the text
text = "The wildfire is terrifying! I feel so helpless."

# Get emotion scores
emotions = emotion_model(text)

required_emotions = {"anger", "confusion", "fear", "joy", "sadness", "neutral", "anticipation","hope"}
filtered_emotions = [{e["label"]: e["score"]} for e in emotions[0] if e["label"] in required_emotions]

# Print result
print(filtered_emotions)


# Load dataset
df = pd.read_csv(r"C:\Users\Devansh\OneDrive\Desktop\PBL\Processed_NaturalDisasters.csv", on_bad_lines="skip")


# Column containing comments
comment_column = "Comments"  # Update if your column name is different

# Dictionary of disasters with their synonyms
disaster_synonyms = {
    "Wildfire": ["fire", "forest fire", "bushfire", "blaze", "inferno"],
    "Flood": ["flood", "flooding", "inundation", "deluge", "high water"],
    "Storm": ["storm", "cyclone", "hurricane", "typhoon", "tornado", "thunderstorm"],
    "Drought": ["drought", "dry spell", "water shortage", "arid", "desertification"],
    "Earthquake": ["earthquake", "tremor", "quake", "seismic activity", "aftershock"],
    "Landslide": ["landslide", "mudslide", "avalanche", "rockfall", "slope failure"],
    "Heatwave": ["heat", "heatwave", "hot spell", "extreme heat","warm" ,"global warming", "heating"],
    "Coldwave": ["coldwave", "extreme cold", "freeze", "frost", "polar vortex", "chill"],
    "Famine": ["famine", "starvation", "food shortage", "malnutrition", "hunger crisis"]
}

# Function to extract the most frequent disaster
def extract_most_frequent_disaster(text):
    if pd.isna(text) or not isinstance(text, str):
        return None  # Skip empty or invalid comments

    words = re.findall(r'\b\w+\b', text.lower())  # Extract words from text
    disaster_mentions = []

    for disaster, synonyms in disaster_synonyms.items():
        for synonym in synonyms:
            if synonym in words:
                disaster_mentions.append(disaster)

    if not disaster_mentions:
        return None  # No disasters found

    # Count occurrences and return the most frequent disaster
    most_common_disaster = Counter(disaster_mentions).most_common(1)[0][0]
    return most_common_disaster  # Return the best-matching disaster

# Apply function to classify disasters
df["Natural Disaster"] = df[comment_column].apply(extract_most_frequent_disaster)

# Load emotion model & tokenizer
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_model = pipeline("text-classification", model=emotion_model_name, top_k=1)
tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)

emotion_mapping = {
    "anger": "anger",
    "disgust": "confusion", 
    "fear": "fear",
    "joy": "joy",
    "sadness": "sadness",
    "surprise": "anticipation" 
}

# Load emotion model & tokenizer
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_model = pipeline("text-classification", model=emotion_model_name, top_k=1)
tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)

# Function to predict emotion
def predict_custom_emotion(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return "hope"  # Assign "hope" instead of "neutral" for empty cases

    encoded_text = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")

    try:
        emotions = emotion_model(tokenizer.decode(encoded_text["input_ids"][0], skip_special_tokens=True))
        if isinstance(emotions, list) and len(emotions) > 0 and isinstance(emotions[0], list):
            predicted_label = emotions[0][0]["label"]
            score = emotions[0][0]["score"]

            mapped_emotion = emotion_mapping.get(predicted_label, "neutral")

            if mapped_emotion == "joy" and score > 0.3:  # Lowered threshold
                return "hope"

            if mapped_emotion in ["neutral", "sadness"] and score < 0.5:
                return "hope"

            if mapped_emotion in ["fear", "anger"] and score < 0.6:
                return "confusion"

            if predicted_label == "disgust":
                return "confusion"
        return mapped_emotion

    except Exception as e:
        print(f"Error predicting emotion: {str(e)}")

    return "error"



# Apply function
tqdm.pandas()
df["Predicted Emotion"] = df[comment_column].progress_apply(predict_custom_emotion)

# Save the output
df.to_csv("final_emotion_analysis.csv", index=False) 

