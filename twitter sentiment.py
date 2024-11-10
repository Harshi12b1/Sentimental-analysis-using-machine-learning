# Imports and setup
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import zipfile

# Google Colab Kaggle API setup (only if running in Google Colab)
# Uncomment the following lines if you're running this in Colab and need to set up Kaggle API
# !pip install kaggle
# from google.colab import files
# files.upload()  # This will prompt you to upload kaggle.json
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# Kaggle dataset download (use only if Kaggle API setup is needed)
# !kaggle datasets download -d kazanova/sentiment140
# with zipfile.ZipFile('sentiment140.zip', 'r') as zip_ref:
#     zip_ref.extractall()

# Download stop words if not already downloaded
nltk.download('stopwords')

# Load dataset
columns = ['target', 'id', 'date', 'flag', 'user', 'text']
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', names=columns)

# Preprocessing
data['target'] = data['target'].replace(4, 1)  # Convert '4' label to '1' for positive sentiment

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Define a function to clean and process each tweet
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]  # Remove stop words and stem
    return ' '.join(words)

# Apply preprocessing to the 'text' column
data['processed_text'] = data['text'].apply(preprocess_text)

# Feature Extraction with TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)  # Limit to 5000 features for efficiency
X = tfidf.fit_transform(data['processed_text']).toarray()  # Transform text to numerical values
y = data['target']  # Target labels

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model Training with Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
