import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./twitter_training.csv', header=None, encoding='utf-8')
# data cleaning
df.dropna(inplace=True)  #removing rows with missing values


#changing column name to undertand them eaily
df.columns = ['Tweet_ID', 'Entity', 'Sentiment', 'Tweet']

# performing EDA
sns.countplot(x='Sentiment', data=df, palette='viridis')
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

sns.countplot(x='Entity', hue='Sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution by Entity')
plt.xlabel('Entity')
plt.xticks(rotation=90)
plt.show()

#preprocesing the text (done so that machine could easily understand it)
stop_words_set = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()  # convert all the words to lower case
    text = re.sub(r'\@\w+|\#','', text)  # removes mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # this will remove punctuation and numbers
    tokens = word_tokenize(text)  # breaks the sentence that is in string in to list of words
    filtered_words = [token for token in tokens if token not in stop_words_set and len(token) > 1]  # Removes stopwords(removes frequently used  words like is , am , are etc)
    return ' '.join(filtered_words) #combines back the list of words to string
df['altered_tw'] = df['Tweet'].apply(preprocess_text) # here preprocess function is applied to each tweet and new column is created with name altered_tw

# Vectorization using TF-IDF(Term Frequency-Inverse Document Frequency)
df.drop(df[df['altered_tw'] == ''].index, inplace=True)  # Remove rows with empty 'altered_tw'
df.drop(df[df['altered_tw'].str.isspace()].index, inplace=True)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['altered_tw'])
le = LabelEncoder()
y = le.fit_transform(df['Sentiment'])  # Encoding sentiment labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')
print(classification_report(y_test, y_pred, target_names=le.classes_))
df.head()
df.info()
df.isnull().sum()
df.describe()
