import random
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
nltk.download('punkt')
nltk.download('stopwords')
corpus = [
    "Hello, how can I help you today?",
    "What's your name?",
    "Tell me a joke.",
    "How's the weather today?",
    "what do u want",
    "Can you recommend a movie?",
    # ... Add more training data
]

stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    words = word_tokenize(text)
    words = [w for w in words if not w in stop_words]
    return ' '.join(words)


processed_corpus = [preprocess(sentence) for sentence in corpus]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_corpus)

# Additional features
random_responses = [
    "I'm not sure I understand. Could you rephrase that?",
    "Interesting. Tell me more!",
    "I'm here to assist you.",
    "Let's talk about something else.",
    # ... Add more random responses
]

knowledge_base = {
    "weather": "The weather is expected to be sunny with a high of 25°C.",
    "programming": "Python is a popular programming language known for its simplicity.",
    # ... Add more topics and responses ...
}

greetings = ["Hello!", "Hi there!", "Hey! How can I help you?"]


# Main chatbot loop
def chatbot_response(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])

    similarities = cosine_similarity(user_vector, X)
    max_similarity_idx = similarities.argmax()

    response = corpus[max_similarity_idx]

    if random.random() < 0.2:  # 20% chance of using a random response
        response = random.choice(random_responses)

    for topic, response_text in knowledge_base.items():
        if topic in user_input:
            response = response_text

    if 'hello' in user_input:
        response = random.choice(greetings)

    return response


# Start chatting
print("Chatbot: Hello, I'm your RAW-chatbot. How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Chatbot:", response)
