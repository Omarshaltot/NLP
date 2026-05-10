# Study Guide for TA Discussion

## 1. What Is This Project?

This project is a simple NLP and Information Retrieval system using the
20 Newsgroups dataset from scikit-learn.

It has two main parts:

1. A search engine that retrieves the most relevant documents for a user query.
2. A text classifier that predicts which newsgroup category a piece of text
   belongs to.

The search engine and the classifier both use TF-IDF features, which convert
text into numerical vectors.

## 2. What Does The Project Do?

The full pipeline is:

1. Load the 20 Newsgroups dataset.
2. Preprocess the text:
   - lowercase
   - remove punctuation and numbers
   - tokenize
   - remove stop words
3. Convert documents into TF-IDF vectors.
4. Train two classifiers:
   - Multinomial Naive Bayes
   - Logistic Regression
5. Evaluate the classifiers using accuracy, precision, recall, F1-score, and
   confusion matrix.
6. Save the trained model, vectorizer, metrics, class names, and search index
   using joblib.
7. Provide a FastAPI backend for search and prediction.
8. Provide a Streamlit GUI for searching, classifying text, viewing metrics,
   and viewing labels.

## 3. The Most Important TA Questions

### TA: Where is tokenization in the code?

Show:

```text
ml/preprocessing.py
```

Important lines:

```python
TOKEN_PATTERN = re.compile(r"[a-z]{2,}")

def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = TOKEN_PATTERN.findall(text)
    return [token for token in tokens if token not in ENGLISH_STOP_WORDS]
```

Explanation:

The `tokenize` function receives raw text. First it converts it to lowercase.
Then `TOKEN_PATTERN.findall(text)` extracts only alphabetic words with at least
two letters. This removes punctuation, numbers, and one-letter tokens. Finally,
the list comprehension removes English stop words such as "the", "is", and
"and".

### TA: Where is preprocessing?

Show:

```text
ml/preprocessing.py
```

Important lines:

```python
def preprocess_text(text: str) -> str:
    return " ".join(tokenize(text))
```

Explanation:

`preprocess_text` calls `tokenize`, then joins the cleaned tokens back into one
string. This cleaned string is passed to the TF-IDF vectorizer.

### TA: Where is TF-IDF?

Show:

```text
ml/vectorizer.py
```

Important lines:

```python
return TfidfVectorizer(
    preprocessor=preprocess_text,
    tokenizer=str.split,
    token_pattern=None,
    lowercase=False,
    max_features=max_features,
    ngram_range=(1, 2),
)
```

Explanation:

`TfidfVectorizer` converts text into numerical vectors. The vectorizer uses the
custom preprocessing function first, then splits the cleaned text into tokens.
It uses unigrams and bigrams, meaning single words and two-word phrases.

### TA: Where is the dataset loaded?

Show:

```text
ml/data_loader.py
```

Important lines:

```python
train_data = fetch_20newsgroups(subset="train", ...)
test_data = fetch_20newsgroups(subset="test", ...)
```

Explanation:

The project loads the standard training and testing splits from scikit-learn.
Headers, footers, and quoted replies are removed to reduce noise.

### TA: Where is the classifier trained?

Show:

```text
ml/train.py
```

Important lines:

```python
classifier.fit(x_train, train_data.target)
predictions = classifier.predict(x_test)
```

Explanation:

`fit` trains the classifier using TF-IDF vectors and their correct class labels.
`predict` tests the classifier on unseen test data.

### TA: Which classifiers are used?

Show:

```text
ml/classifier.py
```

Important lines:

```python
"multinomial_naive_bayes": MultinomialNB(alpha=0.5)
"logistic_regression": OneVsRestClassifier(LogisticRegression(...))
```

Explanation:

The project compares two classical ML classifiers. Logistic Regression is wrapped
with One-vs-Rest because the dataset has 20 classes.

### TA: Where is evaluation?

Show:

```text
ml/evaluation.py
```

Important lines:

```python
accuracy_score(...)
precision_score(..., average="macro")
recall_score(..., average="macro")
f1_score(..., average="macro")
confusion_matrix(...)
```

Explanation:

The project calculates accuracy, precision, recall, F1-score, and confusion
matrix. Macro averaging means each class contributes equally.

### TA: Where is cosine similarity search?

Show:

```text
app/services/search_service.py
```

Important lines:

```python
query_vector = vectorizer.transform([query])
scores = cosine_similarity(query_vector, document_vectors).ravel()
top_indices = scores.argsort()[::-1][:top_k]
```

Explanation:

The query is converted into a TF-IDF vector using the same vectorizer. Then
cosine similarity compares the query vector with every document vector. The
documents are sorted by similarity score, and the top `k` results are returned.

### TA: Where are the model and vectorizer saved?

Show:

```text
ml/train.py
```

Important lines:

```python
joblib.dump(best_model, ARTIFACTS_DIR / "classifier.joblib")
joblib.dump(vectorizer, ARTIFACTS_DIR / "vectorizer.joblib")
```

Explanation:

The trained classifier and fitted TF-IDF vectorizer are saved using joblib so
they can be loaded later by FastAPI and Streamlit without retraining.

### TA: Where are saved artifacts loaded?

Show:

```text
app/services/artifact_loader.py
```

Important lines:

```python
vectorizer = joblib.load(...)
classifier = joblib.load(...)
class_names = joblib.load(...)
search_index = joblib.load(...)
```

Explanation:

This file loads the saved artifacts from the `artifacts/` folder. The
`@lru_cache(maxsize=1)` decorator means the artifacts are loaded once and reused
for later API requests.

### TA: Where is FastAPI defined?

Show:

```text
app/main.py
```

Important lines:

```python
app = FastAPI(...)
app.include_router(search.router)
app.include_router(predict.router)
```

Explanation:

This creates the FastAPI application and connects the route files for health,
search, prediction, labels, and metrics.

### TA: Where is Streamlit?

Show:

```text
ui/streamlit_app.py
```

Important lines:

```python
st.tabs(["Search", "Classify", "Metrics", "Labels"])
```

Explanation:

The Streamlit app gives the user a GUI. It allows searching documents,
classifying text, viewing model metrics, and viewing class labels.

## 4. Important Files Explained

### ml/preprocessing.py

- Imports `re` for regular expressions.
- Imports `ENGLISH_STOP_WORDS` from scikit-learn.
- Defines `TOKEN_PATTERN = re.compile(r"[a-z]{2,}")`.
- `tokenize` lowercases text, extracts valid word tokens, and removes stop words.
- `preprocess_text` joins the cleaned tokens back into a string.

### ml/vectorizer.py

- Imports `TfidfVectorizer`.
- Imports the custom `preprocess_text` function.
- Defines `build_tfidf_vectorizer`.
- The vectorizer uses custom preprocessing.
- `tokenizer=str.split` splits the cleaned string into tokens.
- `ngram_range=(1, 2)` uses words and two-word phrases.
- `max_features=30000` limits the vocabulary size.

### ml/data_loader.py

- Imports `fetch_20newsgroups`.
- Defines `load_20newsgroups_data`.
- Loads the training subset.
- Loads the testing subset.
- Removes headers, footers, and quotes.
- Returns both dataset splits.

### ml/classifier.py

- Imports Logistic Regression, Multinomial Naive Bayes, and One-vs-Rest.
- Defines `build_classifiers`.
- Returns a dictionary of two models.
- Naive Bayes is fast and common for text classification.
- Logistic Regression is a strong classical linear classifier.

### ml/evaluation.py

- Imports metric functions from scikit-learn.
- `evaluate_classifier` computes accuracy, precision, recall, F1-score, and a
  classification report.
- `save_confusion_matrix` creates and saves the confusion matrix image.

### ml/train.py

- Defines the project root and artifacts folder.
- Loads train/test data.
- Builds and fits the TF-IDF vectorizer.
- Transforms train and test documents into vectors.
- Trains both classifiers.
- Evaluates both classifiers.
- Selects the best model using macro F1-score.
- Saves the confusion matrix image.
- Saves the model, vectorizer, class names, search index, and metrics.

### app/services/search_service.py

- Loads saved artifacts.
- Converts the query into a TF-IDF vector.
- Compares query vector to document vectors using cosine similarity.
- Sorts documents by similarity score.
- Returns rank, document id, score, label, and snippet.

### app/services/model_service.py

- Loads the saved vectorizer, classifier, and class names.
- Converts new text into TF-IDF features.
- Predicts the class index.
- Converts the class index into a readable class label.
- Also returns labels and metrics.

### app/routes/*.py

- Each file defines one API route group.
- `health.py` handles `/health`.
- `search.py` handles `/search`.
- `predict.py` handles `/predict`.
- `labels.py` handles `/labels`.
- `metrics.py` handles `/metrics`.

### ui/streamlit_app.py

- Creates a browser-based GUI.
- Search tab calls `/search`.
- Classify tab calls `/predict`.
- Metrics tab calls `/metrics` and displays the confusion matrix.
- Labels tab calls `/labels`.
- If FastAPI is not running, it can load local artifacts directly.

## 5. Short Presentation Script

This project builds a simple search engine and classifier using the 20
Newsgroups dataset. The dataset contains text documents from 20 different
categories. First, the text is cleaned by lowercasing, removing punctuation and
numbers, tokenizing, and removing stop words. Then TF-IDF converts the text into
numerical vectors.

For search, the user query is converted into a TF-IDF vector and compared with
all document vectors using cosine similarity. The highest scoring documents are
returned as the most relevant results.

For classification, the project trains two models: Multinomial Naive Bayes and
Logistic Regression. Both are evaluated using accuracy, precision, recall,
F1-score, and confusion matrix. The best model is saved using joblib and used by
the FastAPI backend and Streamlit frontend.

## 6. Commands To Remember

Train:

```bash
python -m ml.train
```

Run FastAPI:

```bash
uvicorn app.main:app --reload
```

Run Streamlit:

```bash
streamlit run ui/streamlit_app.py
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Open Streamlit:

```text
http://localhost:8501
```
