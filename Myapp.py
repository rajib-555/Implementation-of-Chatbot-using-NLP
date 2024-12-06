import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#.............................................................................
import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Download the 'punkt_tab' data package
nltk.download('punkt_tab') # This line was added to download the necessary 'punkt_tab' data

# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Applies stemming, lemmatization, and stop word removal to the text.
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())  # Convert to lowercase for consistency
    # Remove stop words
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming and lemmatization
    processed_tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in filtered_tokens]
    # Join back into a string
    return " ".join(processed_tokens)

# Configure SSL and NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Define the chatbot response function
def chatbot(input_text):
    # Preprocess the input text
    input_text = preprocess_text(input_text)
    # Vectorize input
    input_text = vectorizer.transform([input_text])
    # Predict tag
    tag = clf.predict(input_text)[0]
 
    # Find response
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "Sorry, I didn't understand that."  # Fallback response


# Counter for unique text input fields
counter = 0

def main():
    global counter
    st.title("Chatbot for Dr.B.C.Roy Engineering College - Ask Questions,Get Answers!")

    # Function to convert an image to base64
    def image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

# Path to your local image
    image_path = "bg11.jpg"
    logo_path ="logo.jpeg"

# Convert the image to base64
    background_image_base64 = image_to_base64(image_path)
    logo_image_base64 = image_to_base64(logo_path)
# Inject custom CSS to set the background image
    st.markdown(
    f"""
    <style>
    .stApp {{
        position: relative;
        background-image: url(data:image/png;base64,{background_image_base64});
        background-size: cover;
        background-position: center;
        height: 100vh;
        color: white;  /* Make text white for better contrast */
        display: flex;
        justify-content: center;
        align-items: flex-start;  /* Align the content towards the top */
        flex-direction: column;
        padding: 10px;  /* Reduce padding around the content */
    }}
    
    .stTitle {{
        margin-bottom: 10px;  /* Reduce the margin under the title */
    }}
    
    .center {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 0;  /* Remove top margin for logo */
        margin-bottom: 10px;  /* Reduce space below logo */
    }}
    
    /* Lightened dark overlay with lower opacity to reduce brightness */
    .stApp::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.2);  /* Light overlay to reduce brightness */
        z-index: -1;  /* Ensure the overlay stays behind text */
    }}
    </style>
    <div class="center">
        <img src="data:image/jpeg;base64,{logo_image_base64}" alt="Logo" width="200"/>
    </div>

    """,
    unsafe_allow_html=True
)

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Curious about the college? Chat with our Bot for quick and helpful information! ")

        # Ensure the log file exists
        log_file = 'chat_log.csv'
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        # User input and chatbot response
        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Log the conversation
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            # Exit if user says goodbye
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting! Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        log_file = 'chat_log.csv'
        
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.subheader("Project Overview")
        st.write("""
        This chatbot is designed to provide instant assistance and information to users about Dr. B.C. Roy Engineering College. 
        It uses advanced Natural Language Processing (NLP) techniques and Logistic Regression to accurately recognize user intents and deliver relevant responses. 
        The interactive interface is built using Streamlit, offering a seamless and engaging platform for users to easily interact with the bot and get answers to their queries.
        """)

        st.subheader("Features")
        st.write("""
        - Intent-based response generation
        - Logging of conversations for future reference
        - User-friendly interface with Streamlit
        - Extensible design for adding more intents and patterns
        """)

        st.subheader("Future Enhancements")
        st.write("""
        - Support for multi-turn conversations
        - Integration with advanced NLP models
        - Multi-language support
        - Personalized responses based on user preferences
        """)

if __name__ == '__main__':
    main()
