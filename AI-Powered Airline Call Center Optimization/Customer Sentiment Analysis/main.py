import os
import google.generativeai as genai
import re
import nltk
import joblib
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv("api_keys.env")
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Verify API key
if API_KEY:
    print("✅ Gemini API Key Loaded Successfully!")
else:
    print("❌ API Key Not Found! Check api_keys.env file.")

# Download necessary NLTK data
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text: str) -> str:
    """Cleans and tokenizes input text."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    words = text.split()  # Tokenization
    return ' '.join(words)

# Function to analyze sentiment using Gemini API (1.5-flash)
def gemini_sentiment_analysis(text: str) -> str:
    """Uses Gemini 1.5-flash API for sentiment analysis."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Updated model name
        response = model.generate_content(
            f"Classify the sentiment of the following text as either 'positive', 'negative', or 'neutral'. "
            f"Do NOT provide any explanation, just output one word.\n\nText: '{text}'"
        )
        
        sentiment = response.text.strip().lower()
        
        if sentiment in ["positive", "negative", "neutral"]:
            return sentiment
        return "unknown"
    
    except Exception as e:
        print("❌ Error with Gemini API:", e)
        return "unknown"

# Train sentiment analysis model using Logistic Regression
def train_sentiment_model(training_data: List[Tuple[str, str]]):
    """Trains a Logistic Regression model for sentiment analysis."""
    
    # Convert to DataFrame
    df = pd.DataFrame(training_data, columns=['text', 'label'])
    
    # Handle missing values
    df.dropna(inplace=True)

    # Preprocess text
    df['text'] = df['text'].apply(preprocess_text)

    # Convert labels to binary values
    df['label'] = df['label'].map({'positive': 1, 'negative': 0})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # TF-IDF Vectorization with n-grams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 Model Accuracy: {accuracy:.2f}")
    print("📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save the trained model and vectorizer
    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    return model

# Function to predict sentiment using Logistic Regression
def predict_sentiment(model, new_text: str) -> str:
    """Predicts the sentiment of a given text using traditional ML model."""
    
    # Load vectorizer
    vectorizer = joblib.load("vectorizer.pkl")
    
    # Preprocess input text
    new_text = preprocess_text(new_text)
    
    # Transform using TF-IDF
    new_text_tfidf = vectorizer.transform([new_text])
    
    # Predict sentiment
    prediction = model.predict(new_text_tfidf)[0]
    
    return "positive" if prediction == 1 else "negative"

# Test cases
if __name__ == "__main__":
    sample_data = [
        ("I love this airline", "positive"),
        ("Worst experience ever", "negative"),
        ("The flight was on time, and the staff was friendly.", "positive"),
        ("I had to wait 3 hours due to a delay. Terrible!", "negative"),
        ("The food was amazing!", "positive"),
        ("The seats were uncomfortable.", "negative"),
        ("The crew was very helpful.", "positive"),
        ("They lost my baggage!", "negative"),
        ("I had a pleasant journey.", "positive"),
        ("The ticket prices are too high.", "negative"),
        ("Seats were comfortable and legroom was good.", "positive"),
    ("The flight was delayed by 5 hours. Very frustrating!", "negative"),
    ("My luggage was lost and never recovered.", "negative"),
    ("The food was excellent and the service was friendly.", "positive"),
    ("I had to wait for a long time at check-in.", "negative"),
    ("The cabin was clean and well-maintained.", "positive"),
    ("The entertainment system was outdated and barely worked.", "negative"),
    ("The takeoff and landing were smooth.", "positive"),
    ("Rude staff made the flight experience unpleasant.", "negative"),
    ("The boarding process was very efficient.", "positive"),
    ("My seat was broken and couldn't recline.", "negative"),
    ("The flight attendants were very professional.", "positive"),
    ("The flight was canceled last minute, terrible service.", "negative"),
    ("The inflight meals were surprisingly tasty.", "positive"),
    ("There was no WiFi available on the flight.", "negative"),
    ("The business class experience was luxurious.", "positive"),
    ("The economy seats were too cramped for comfort.", "negative"),
    ("The pilot provided clear and reassuring announcements.", "positive"),
    ("The airline lost my baggage and offered no help.", "negative"),
    ("The staff was accommodating and helpful.", "positive"),
    ("The airline overbooked my flight and I had to wait.", "negative"),
    ("Check-in was quick and hassle-free.", "positive"),
    ("The lavatories were dirty and smelled bad.", "negative"),
    ("There were no delays, everything went smoothly.", "positive"),
    ("The security check took too long, very annoying.", "negative"),
    ("They served free drinks, which was a nice touch.", "positive"),
    ("The air conditioning was not working properly.", "negative"),
    ("The legroom in premium economy was great.", "positive"),
    ("The crew was rude and unhelpful.", "negative"),
    ("I got an upgrade to business class, best experience ever!", "positive"),
    ("There was turbulence throughout the flight, very scary.", "negative"),
    ("The flight was cheap but service was still good.", "positive"),
    ("They lost my special meal request, very disappointing.", "negative"),
    ("The plane was brand new and very clean.", "positive"),
    ("The overhead compartments were too small for carry-ons.", "negative"),
    ("They gave free snacks, which was a nice surprise.", "positive"),
    ("I had to sit next to a crying baby for the whole flight.", "negative"),
    ("The airport lounge was comfortable and well-equipped.", "positive"),
    ("The baggage claim process was slow and chaotic.", "negative"),
    ("The takeoff was smooth and we landed ahead of schedule.", "positive"),
    ("There was no entertainment system on the flight.", "negative"),
    ("The crew made my child feel special on their birthday.", "positive"),
    ("The airline refused to refund my canceled flight.", "negative"),
    ("My flight experience exceeded expectations.", "positive"),
    ("The boarding gate was changed last minute, very confusing.", "negative"),
    ("The first-class seats were extremely comfortable.", "positive"),
    ("I got food poisoning from the airline meal.", "negative"),
    ("They handled my wheelchair request efficiently.", "positive"),
    ("The check-in staff was rude and unprofessional.", "negative"),
    ("The plane had USB ports at every seat.", "positive"),
    ("There was an issue with my ticket, and no one helped.", "negative"),
    ("The plane had extra legroom for tall passengers.", "positive"),
    ("They lost my pet during transit, absolutely unacceptable.", "negative"),
    ("The in-flight service was impeccable.", "positive"),
    ("The seatbelt sign was on almost the entire flight.", "negative"),
    ("The kids' meal was nutritious and well-prepared.", "positive"),
    ("The flight had a technical issue, which caused delays.", "negative"),
    ("The staff provided blankets and pillows for extra comfort.", "positive"),
    ("The baggage fees were extremely high.", "negative"),
    ("The WiFi worked perfectly throughout the flight.", "positive"),
    ("The food quality was poor and inedible.", "negative"),
    ("The boarding process was organized and smooth.", "positive"),
    ("My in-flight entertainment screen was broken.", "negative"),
    ("The crew helped me find space for my carry-on.", "positive"),
    ("The flight was very noisy due to a large group of passengers.", "negative"),
    ("The pilot's landing was incredibly smooth.", "positive"),
    ("They ran out of meals before serving me.", "negative"),
    ("The customer service team resolved my issue quickly.", "positive"),
    ("The lavatory was out of service for most of the flight.", "negative"),
    ("I was upgraded to an exit row seat for free.", "positive"),
    ("The flight had an emergency landing, very stressful.", "negative"),
    ("They gave free headphones for the entertainment system.", "positive"),
    ("The flight was severely overbooked, terrible experience.", "negative"),
    ("The staff went above and beyond to assist me.", "positive"),
    ("The air conditioning made the cabin too cold.", "negative"),
    ("The priority boarding was handled efficiently.", "positive"),
    ("The food portions were too small.", "negative"),
    ("My baggage was waiting for me when I landed.", "positive"),
    ("There was no legroom in my economy seat.", "negative"),
    ("The airline provided a hotel for my overnight layover.", "positive"),
    ("The check-in kiosk was not working.", "negative"),
    ("The pilot kept us updated throughout the flight.", "positive"),
    ("The security line was unbearably long.", "negative"),
    ("The airplane seats had built-in footrests.", "positive"),
    ("There was a long wait to retrieve checked baggage.", "negative"),
    ("The boarding process was very fast.", "positive"),
    ("The crew ignored my request for assistance.", "negative"),
    ("The plane had charging ports at every seat.", "positive"),
    ("They misplaced my special luggage.", "negative"),
    ("The airline honored my seat selection.", "positive"),
    ("The flight was too expensive for the service provided.", "negative"),
    ("The in-flight announcements were clear and helpful.", "positive"),
    ("The airline changed my seat without informing me.", "negative"),
    ("The boarding area had plenty of seating.", "positive"),
    ("They ran out of vegetarian meal options.", "negative"),
    ("The crew provided excellent service throughout.", "positive"),
    ("The flight was delayed due to bad weather.", "negative"),
    ]

    # Train model
    model = train_sentiment_model(sample_data)

    # Test predictions with ML model
    test_texts = [
        "The seats were comfortable and service was great!",
        "They lost my baggage and were very unhelpful!",
        "Nothing special, just an average flight."
    ]

    print("\n🧠 **ML Model Predictions**:")
    for text in test_texts:
        print(f"📢 Text: {text} --> Sentiment: {predict_sentiment(model, text)}")

    # Test predictions with Gemini API (1.5-flash)
    print("\n🤖 **Gemini API Predictions**:")
    for text in test_texts:
        print(f"📢 Text: {text} --> Sentiment: {gemini_sentiment_analysis(text)}")
