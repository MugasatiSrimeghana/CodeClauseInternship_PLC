# CodeClauseInternship_PLC
This Project is done by Mugasati Srimeghana for @CodeClause Internship.

## What is plagiarism detector and project overview

In this project, we will be building a plagiarism detector that examines a text file and performs binary classification; labeling that file as either plagiarized or not, depending on how similar that text file is to a provided source text. Detecting plagiarism is an active area of research; the task is non-trivial and the differences between paraphrased answers and original work are often not so obvious.

One of the ways we might go about detecting plagiarism, is by computing similarity features that measure how similar a given text file is as compared to an original source text. We can develop as many features as we want and are required to define a couple as outlined in this paper. In this paper, researchers created features called containment and longest common subsequence.

We will be defining a few different similarity features to compare the two texts. Once we have extracted relevant features, we will explore different classification models and decide on a model that gives us the best performance on a test dataset.

## The Dataset

I have prepare a very simple dataset for this tutorial. It contains sample of plagiarized article contents and non-plagiarized — this is derived by the Label 0 for false and 1 for true.

## 1. Import libraries

The provided code and libraries facilitate the development and evaluation of a plagiarism detection model. It begins by downloading essential Natural Language Toolkit (NLTK) resources, enabling access to valuable linguistic datasets and models. Next, the code imports several crucial libraries.

From scikit-learn, the code imports the cosine_similarity function, which calculates the similarity between vectors and is useful for comparing documents. Additionally, Pandas (import pandas as pd) is imported for efficient data manipulation and analysis.

The code employs the string module for handling string operations, especially text preprocessing, and imports the stopwords corpus from NLTK. This corpus contains common words frequently removed during text preprocessing.

To save and load Python objects, including machine learning models, joblib is imported. For classification tasks, like plagiarism detection, the code imports the LogisticRegression class from scikit-learn.

For dataset splitting during model evaluation, train_test_split is imported. Finally, the code brings in evaluation metrics, such as accuracy_score and classification_report, for assessing model performance.

Lastly, the TfidfVectorizer class from scikit-learn is imported, offering TF-IDF-based text feature extraction—an essential technique in text classification and information retrieval tasks.

## 2. Load CSV

## 3. Cleanup Dataset

The code defines a text preprocessing function that:
- Removes punctuation.
- Converts text to lowercase.
- Eliminates common English stopwords (e.g., “the,” “and”).
It then applies this preprocessing to two text columns in a dataset. The goal is to clean and standardize the text for further analysis or modeling, such as plagiarism detection.

## 4. Vectorization

Next we use the TF-IDF vectorization technique to convert text data from two columns into a numerical format. It combines the text from “source_text” and “plagiarized_text” columns, calculates the TF-IDF values for each word, and stores these values in a feature matrix called “X.” This transformation is essential for preparing the text data for machine learning models.

TF-IDF stands for Term Frequency-Inverse Document Frequency, and it is a numerical statistic used in natural language processing and information retrieval to evaluate the importance of a word within a document relative to a collection of documents (corpus). TF-IDF is a commonly used technique for text data preprocessing and feature extraction.  

## 5. Logistic Regression & Training

Creates a logistic regression model, and model.fit(X, y) trains the model using the features X and the target variable y. This is where we do the training the most.

Data Splitting: It divides the dataset into two subsets: a training set (used for model training) and a testing set (used for model evaluation). The split is 80% for training and 20% for testing.

Model Prediction: It applies a machine learning model to the testing set to make predictions based on the model’s learned patterns.

Accuracy Calculation: The code calculates the accuracy of the model’s predictions by comparing them to the true labels in the testing set. Accuracy quantifies how often the model’s predictions match the actual outcomes.

Classification Report: A detailed classification report is generated, which includes precision (accuracy of positive predictions), recall (proportion of actual positives correctly predicted), F1-score (harmonic mean of precision and recall), and support (the number of occurrences of each class).

## 6. Saving the Model

Save the model so we can use it later on as we like.

Overall, this workflow allows us to gauge the model’s effectiveness in classifying data, providing valuable information for decision-making and further model refinement. The accuracy score provides a high-level summary of performance, while the classification report offers a nuanced assessment of the model’s strengths and weaknesses, making it a valuable tool for evaluating classification models.



