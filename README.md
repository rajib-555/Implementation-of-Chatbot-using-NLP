# Implementation-of-Chatbot-using-NLP
## Chatbot for Dr. B.C. Roy Engineering College

This is a Natural Language Processing (NLP)-based chatbot developed for Dr. B.C. Roy Engineering College. The chatbot is built to provide useful information to students, faculty, and visitors of the college through a conversational interface. It uses various NLP tools and machine learning algorithms to understand user inputs and provide relevant responses.

---

## Features
- Simple and interactive interface built with Streamlit for easy communication.
- Provides relevent responses based on matched patterns and user intent.
- Maintains a conversation history that can be viewed by the user.

---

## Technologies Used
- **Python**
- **NLTK**
- **Scikit-learn**
- **Streamlit**
- **JSON** for intents data

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
import nltk
nltk.download('punkt')
```

---

## Usage
To run the chatbot application, execute the following command:
```bash
streamlit run app.py
```

Once the application is running, you can interact with the chatbot through the web interface. Type your message in the input box and press Enter to see the chatbot's response.

---

## Intents Data
The chatbot's behavior is defined by the `intents.json` file, which contains various tags, patterns, and responses. You can modify this file to add new intents or change existing ones.

---

## Conversation History
The chatbot saves the conversation history in a CSV file (`chat_log.csv`). You can view past interactions by selecting the "Conversation History" option in the sidebar.

---

## Acknowledgments
- **NLTK** for natural language processing.
- **Scikit-learn** for machine learning algorithms.
- **Streamlit** for building the web interface.

---
