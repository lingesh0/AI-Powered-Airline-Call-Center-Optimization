Airline Call Center Sentiment Analysis


This project analyzes customer sentiment from airline call center feedback using a machine learning model and Gemini 1.5 Flash API.

📁 Project Structure

problem2/
│── venv/                  # Virtual environment
│── api_keys.env           # API keys file (DO NOT SHARE)
│── main.py                # Main script for sentiment analysis
│── README.md              # Project documentation
│── sentiment_model.pkl    # Pre-trained sentiment classification model
│── vectorizer.pkl         # TF-IDF vectorizer for text processing

📌 Features
Machine Learning Model: Predicts sentiment using a trained classifier.

Gemini API (1.5 Flash): Generates sentiment analysis results.

Pre-trained Model: Uses sentiment_model.pkl and vectorizer.pkl for predictions.

Confusion Matrix & Accuracy Metrics: Evaluates ML model performance.

🚀 Installation & Setup
1️⃣ Create & Activate Virtual Environment

python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows
2️⃣ Install Dependencies

pip install -r requirements.txt
3️⃣ Set Up API Key
Create a .env file or update api_keys.env with your Gemini API key:


GEMINI_API_KEY=your_gemini_api_key
4️⃣ Run the Script

python main.py
📊 Example Output

✅ Gemini API Key Loaded Successfully!
📊 Model Accuracy: 0.68
📌 Confusion Matrix:
 [[11  0]
 [ 7  4]]

🧠 ML Model Predictions:
📢 Text: The seats were comfortable and service was great! --> Sentiment: positive
📢 Text: They lost my baggage and were very unhelpful! --> Sentiment: negative

🤖 Gemini API Predictions:
📢 Text: The seats were comfortable and service was great! --> Sentiment: positive
📢 Text: They lost my baggage and were very unhelpful! --> Sentiment: negative
🛠 Dependencies
Python 3.8

google.generativeai (Gemini API)

sklearn (Machine Learning)

nltk (Natural Language Processing)

📌 Future Enhancements
Improve ML model accuracy

Fine-tune Gemini API responses

Add UI for real-time sentiment analysis


 Developed for AI-Powered Airline Call Center Optimization......