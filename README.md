
Project Goal
The primary objective of this project was to analyze and visualize sentiment patterns in a large social media dataset. By training a robust machine learning model, we aimed to automatically classify public opinion (Positive, Negative, Neutral, etc.) towards various brands and entities, providing a scalable tool for market research and reputation management.

🚀 The Data Science Pipeline
This project utilized a standard Machine Learning workflow to ensure reliable and verifiable results:

Data Cleaning: Removed missing values (dropna) and normalized column names.

Exploratory Data Analysis (EDA): Visualized sentiment distributions to establish a baseline.

Text Preprocessing: Removed noise (URLs, stopwords, punctuation).

Feature Engineering: Converted the clean text into a numerical format using TF-IDF.

Modeling & Evaluation: Trained a Multinomial Naive Bayes model and evaluated its performance.

📊 Exploratory Data Analysis (EDA) Insights
The initial visualization confirmed strong sentiment patterns that guided the modeling process.

Key Finding: Polarization by Entity
By breaking down sentiment by brand, we immediately identify polarization. Brands show clear differences in public perception, with some receiving concentrated Negative feedback while others maintain high positive sentiment.

Key Finding: Sentiment Distribution
The data, while generally balanced across the whole set, has a significant number of Neutral tweets. This highlights the need for a model that can accurately distinguish between genuine sentiment (Positive/Negative) and simple factual statements (Neutral).

🛠️ Methodology: The Technical Choices
1. Text Preprocessing: Removing Noise
Raw tweet text was thoroughly cleaned to remove noise and standardize the vocabulary:

Normalization: Lowercased all text.

Noise Filtering: Removed mentions (@user), hashtags, and punctuation.

Stopword Removal: Eliminated common words ("the," "is," "a") so the model could focus its learning on sentiment-carrying words like "amazing" or "broken."

2. Feature Engineering: TF-IDF Vectorization
We chose TF-IDF (Term Frequency-Inverse Document Frequency) to convert the clean text into a numerical format.

Why TF-IDF? This method weights words, assigning a high numerical score to words that are frequent in one specific tweet but rare across the entire dataset. This allows the model to prioritize keywords that truly signal a change in sentiment.

3. Model Selection: Multinomial Naive Bayes (MNB)
The Multinomial Naive Bayes algorithm was used as the classifier.

Why MNB? MNB is a highly efficient and reliable probabilistic model well-suited for high-dimensional, sparse data like TF-IDF features. It was chosen to establish a strong, fast baseline for text classification.

✅ Results and Conclusion
Model Performance
The Multinomial Naive Bayes model was trained and evaluated on the final cleaned dataset.

Metric	Score	Interpretation
Overall Accuracy	64.10%	The model correctly predicted the sentiment for 64 out of every 100 tweets, providing a reliable baseline for the four-class problem.

Export to Sheets
Classification Report

The detailed report below indicates that the model struggles most with Recall for the Irrelevant class (0.36), often misclassifying truly irrelevant tweets as Neutral or Negative. However, it is strongest at identifying the Negative class (Recall: 0.81).

              precision    recall  f1-score   support

  Irrelevant       0.73      0.36      0.48        2551
    Negative       0.64      0.81      0.71        4319
     Neutral       0.66      0.52      0.58        3542
    Positive       0.61      0.75      0.67        4040

    accuracy                           0.64       14452
   macro avg       0.66      0.61      0.61       14452
weighted avg       0.65      0.64      0.63       14452

Conclusion
The project successfully achieved its goal of establishing a working sentiment analysis pipeline. The model provides a reliable baseline for classifying public opinion, with high performance in identifying Negative and Positive sentiments. While the model would benefit from further tuning (e.g., using Logistic Regression or N-grams) to improve the Recall for 'Irrelevant' and 'Neutral' classes, it is immediately useful for tracking overall positive/negative trends toward the entities analyzed.
