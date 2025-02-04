import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')  # Sometimes, this additional WordNet corpus is needed

# Load pre-trained model and vectorizer
model = joblib.load('lr_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function (same as training)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
    tweet = ' '.join([word for word in tweet.split() if word not in stop_words])
    tweet = ' '.join([lemmatizer.lemmatize(word) for word in tweet.split()])
    return tweet

# Classify function
def classify_tweet(tweet):
    preprocessed_tweet = preprocess_tweet(tweet)  # Step 1: Preprocess
    processed_tweet_tfidf = vectorizer.transform([preprocessed_tweet])  # Step 2: TF-IDF transform

    print(f"Processed tweet shape: {processed_tweet_tfidf.shape}")  # Debug: Should be (1, 5000)
    
    prediction = model.predict(processed_tweet_tfidf)  # Step 3: Predict
    
    return "Disaster" if prediction[0] == 1 else "Non-Disaster"


