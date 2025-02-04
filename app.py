from flask import Flask, request, jsonify, send_from_directory
import os
from lr_model import classify_tweet  # Import your model classification function

# Set the absolute path to your static folder
STATIC_FOLDER = r'C:\S.R.M\college projects\tweet classifier\static'

app = Flask(__name__, static_folder=STATIC_FOLDER)

# Route to serve index.html from the static folder
@app.route('/')
def serve_index():
    return send_from_directory(STATIC_FOLDER, 'index.html')


# Route for classifying tweets (using your model)
@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        tweet = data.get('tweet')
        if not tweet:
            return jsonify({'error': 'No tweet provided'}), 400

        classification = classify_tweet(tweet)
        return jsonify({
            'success': True,
            'classification': classification,
            'tweet': tweet
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
