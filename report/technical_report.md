# Technical Report: Simple Search Engine Using 20 Newsgroups

## 1. Problem Definition

The goal of this project is to build a simple search engine that retrieves the
most relevant documents for a user query. The system uses classical Natural
Language Processing and Information Retrieval methods. In addition to document
retrieval, the project trains and evaluates a text classifier in order to
satisfy the machine learning requirements of the course.

The system accepts a raw user query, preprocesses it, converts it into a TF-IDF
vector, compares it with document vectors using cosine similarity, and returns
the top ranked documents. A supervised classifier is also trained to predict the
topic category of new text.

## 2. Dataset Description

The dataset used is the 20 Newsgroups dataset available through scikit-learn's
`fetch_20newsgroups` function. It contains newsgroup posts from 20 topic
categories, including computer graphics, politics, religion, science, sports,
and technology-related discussions.

The dataset is appropriate for this project because each item is a real text
document with a category label. This supports both information retrieval and
supervised text classification.

Dataset reference:
https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset

## 3. Preprocessing Steps

The preprocessing pipeline applies the following steps:

1. Convert all text to lowercase.
2. Remove punctuation and numeric tokens by keeping alphabetic tokens only.
3. Tokenize text using a regular expression.
4. Remove English stop words using scikit-learn's built-in stop-word list.
5. Join the remaining tokens into a cleaned string for TF-IDF vectorization.

The same preprocessing function is used for training documents, search queries,
and prediction input. This consistency is important because the model and search
engine should receive text in the same format during training and inference.

## 4. Feature Extraction

The project uses `TfidfVectorizer` for feature extraction. TF-IDF represents
documents according to how important each term is in a document relative to the
full document collection. Common words receive lower weight, while more
informative terms receive higher weight.

The vectorizer uses unigram and bigram features, a maximum feature limit, and
sublinear term frequency scaling. The fitted vectorizer is saved using `joblib`
as `artifacts/vectorizer.joblib`.

## 5. Model Used

Two classical machine learning classifiers are trained and compared:

- Multinomial Naive Bayes
- Logistic Regression

Both models use the same TF-IDF feature representation. After evaluation, the
model with the highest macro F1-score is selected and saved as
`artifacts/classifier.joblib`.

The project remains classical ML only and does not use deep learning.

## 6. Evaluation Results

The classifier is evaluated using the test split of the 20 Newsgroups dataset.
The following metrics are computed:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix

The numeric metrics are saved in `artifacts/metrics.json`. The confusion matrix
is saved as `artifacts/confusion_matrix.png`.

The training run selected Logistic Regression because it achieved the best macro
F1-score among the compared models:

```json
{
  "selected_model": "logistic_regression",
  "model_comparison": {
    "multinomial_naive_bayes": {
      "accuracy": 0.6832,
      "precision_macro": 0.7149,
      "recall_macro": 0.6642,
      "f1_macro": 0.6568
    },
    "logistic_regression": {
      "accuracy": 0.6905,
      "precision_macro": 0.6918,
      "recall_macro": 0.6768,
      "f1_macro": 0.6767
    }
  }
}
```

## 7. API Deployment with FastAPI

The project includes a FastAPI backend that loads the saved artifacts and exposes
the trained search and classification system through HTTP endpoints.

Implemented endpoints:

- `GET /health`: returns the API status.
- `POST /search`: accepts a query and `top_k`, then returns ranked documents.
- `POST /predict`: accepts raw text and returns the predicted class.
- `GET /labels`: returns the 20 class names.
- `GET /metrics`: returns the saved evaluation metrics.

The API uses Pydantic schemas for request and response validation. The code is
organized into route modules, schema modules, and service modules.

## 8. GUI Implementation with Streamlit

The Streamlit frontend provides a simple graphical interface for the project.
It includes:

- A search section for entering queries and choosing the number of results.
- A classification section for predicting the category of raw text.
- A metrics section showing saved evaluation metrics and the confusion matrix.
- A labels section displaying all dataset categories.

The Streamlit app first attempts to call the FastAPI backend. If the backend is
not running, it loads the saved artifacts directly as a fallback.

## 9. Conclusion

This project demonstrates a complete classical NLP and Information Retrieval
pipeline. It loads a real text dataset, preprocesses the documents, extracts
TF-IDF features, builds a cosine-similarity search engine, trains and compares
classifiers, evaluates model performance, and exposes the system through both an
API and a graphical interface.

Future improvements could include more advanced ranking methods, additional
hyperparameter tuning, stemming or lemmatization, user feedback for relevance
improvement, and deployment to a hosted web service.
