# Simple Search Engine: 20 Newsgroups

This project builds a classical NLP / Information Retrieval system using the
20 Newsgroups dataset from scikit-learn. It includes:

- TF-IDF based document retrieval with cosine similarity
- Text preprocessing: lowercasing, punctuation/number removal, tokenization, stop-word removal
- Classifier training and evaluation
- Comparison between Multinomial Naive Bayes and Logistic Regression
- Saved artifacts using `joblib`
- FastAPI backend for deployment bonus marks
- Streamlit frontend for GUI bonus marks

## Dataset Justification

The 20 Newsgroups dataset is a good fit for this assignment because it contains
real text documents grouped into 20 topic categories. This makes it suitable for
both information retrieval, where a query is matched against documents, and text
classification, where a model predicts the category of a document.

Dataset reference:
[scikit-learn 20 Newsgroups dataset](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)

## Tools Used and Why

This project uses classical NLP, machine learning, API, and GUI tools. Each tool
has a specific purpose in the project.

| Tool | Why it was used |
|---|---|
| Python | Main programming language for the whole project. |
| scikit-learn | Used to load the dataset, build TF-IDF features, train classifiers, and calculate evaluation metrics. |
| `fetch_20newsgroups` | Loads the 20 Newsgroups dataset directly from scikit-learn. |
| Regular expressions `re` | Used during preprocessing to extract valid word tokens and remove punctuation/numbers. |
| `ENGLISH_STOP_WORDS` | Removes common English stop words such as "the", "is", and "and". |
| `TfidfVectorizer` | Converts text documents into numerical TF-IDF vectors. |
| Cosine similarity | Measures similarity between the user query vector and document vectors for search. |
| `MultinomialNB` | A classical Naive Bayes classifier commonly used for text classification. |
| `LogisticRegression` | A strong classical linear classifier used for comparison. |
| `OneVsRestClassifier` | Allows Logistic Regression to handle the 20-class classification problem. |
| Accuracy, Precision, Recall, F1-score | Required metrics used to evaluate classifier performance. |
| Confusion Matrix | Shows correct and incorrect predictions for each class. |
| matplotlib | Saves the confusion matrix as `artifacts/confusion_matrix.png`. |
| joblib | Saves and loads the trained classifier, fitted vectorizer, labels, and search index. |
| JSON | Saves evaluation results in `artifacts/metrics.json`. |
| FastAPI | Provides a backend API so the trained system can be used through HTTP endpoints. |
| Pydantic | Validates FastAPI request and response data. |
| Uvicorn | Runs the FastAPI server locally. |
| Streamlit | Provides a simple browser GUI for search, classification, metrics, and labels. |
| requests | Allows the Streamlit app to call the FastAPI endpoints. |
| Git/GitHub | Used for version control and project submission. |

### Why FastAPI Was Used

FastAPI is used as the deployment layer for the project. After training, the
classifier, vectorizer, metrics, labels, and search index are saved as artifacts.
FastAPI loads these saved artifacts and exposes them through API endpoints.

This means another program can send requests such as:

```text
POST /search
POST /predict
GET /metrics
GET /labels
```

and receive JSON responses without retraining the model.

In simple words:

> FastAPI turns the trained machine learning system into a usable backend
> service.

### Why Streamlit Was Used

Streamlit is used to create a simple graphical user interface. Instead of using
curl commands or API docs, a user can open a browser page and:

- enter a search query
- choose the number of results
- classify raw text
- view evaluation metrics
- view the confusion matrix
- view all class labels

In simple words:

> FastAPI is mainly for programs and APIs, while Streamlit is for humans using a
> browser.

### Why joblib Was Used

Training the model every time would be slow and unnecessary. The project uses
joblib to save the trained model and fitted vectorizer after training.

Saved artifacts include:

- `artifacts/classifier.joblib`
- `artifacts/vectorizer.joblib`
- `artifacts/class_names.joblib`
- `artifacts/search_index.joblib`

FastAPI and Streamlit load these files directly, so the project can make
predictions and search documents immediately after startup.

### Why TF-IDF Was Used

Machine learning models cannot directly understand raw text, so text must be
converted into numbers. TF-IDF gives higher weight to important words and lower
weight to very common words.

This makes TF-IDF useful for both:

- document retrieval
- text classification

### Why Cosine Similarity Was Used

The search engine converts both the user query and all documents into TF-IDF
vectors. Cosine similarity measures how close the query vector is to each
document vector.

Documents with higher cosine similarity scores are considered more relevant and
are returned first.

### Why Two Classifiers Were Used

The assignment requires at least one classifier, but this project trains two for
bonus comparison:

- Multinomial Naive Bayes
- Logistic Regression

Both models are evaluated, and the one with the best macro F1-score is saved as
the final classifier.

## Project Structure

```text
app/
  main.py
  routes/
    health.py
    labels.py
    metrics.py
    predict.py
    search.py
  schemas/
    requests.py
    responses.py
  services/
    artifact_loader.py
    model_service.py
    search_service.py
ml/
  data_loader.py
  preprocessing.py
  vectorizer.py
  classifier.py
  evaluation.py
  train.py
ui/
  streamlit_app.py
artifacts/
  classifier.joblib
  vectorizer.joblib
  metrics.json
  confusion_matrix.png
  class_names.joblib
  search_index.joblib
report/
  technical_report.md
requirements.txt
README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model and Build the Search Index

Run this command from the project root:

```bash
python -m ml.train
```

This creates the following files under `artifacts/`:

- `classifier.joblib`
- `vectorizer.joblib`
- `class_names.joblib`
- `search_index.joblib`
- `metrics.json`
- `confusion_matrix.png`

## Run the FastAPI Server

```bash
uvicorn app.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

### Windows Port Error Fix

If you see this error:

```text
[WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions
```

it usually means Uvicorn cannot use the requested port. The most common causes
are that another process is already using the port, or Windows has blocked that
port.

Check whether port `8000` is already in use:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen
```

If a process is listed, stop it by replacing `<PID>` with the `OwningProcess`
value:

```powershell
Stop-Process -Id <PID> -Force
```

Then run FastAPI again:

```bash
uvicorn app.main:app --reload
```

Alternatively, run the API on another port:

```bash
uvicorn app.main:app --reload --port 8001
```

## Run the Streamlit App

In another terminal, run:

```bash
streamlit run ui/streamlit_app.py
```

The Streamlit app tries to call the FastAPI endpoints first. If FastAPI is not
running, it falls back to loading the saved local artifacts directly.

## Run on the Local Network

By default, `127.0.0.1` means "this computer only". If you want another device
on the same Wi-Fi or LAN to open the project, run the servers on `0.0.0.0`.

First, find your computer's local IP address.

Windows PowerShell:

```powershell
Get-NetIPAddress -AddressFamily IPv4
```

Look for an address similar to:

```text
192.168.x.x
```

In this example, assume the computer IP is:

```text
192.168.100.59
```

Run FastAPI for network access:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run Streamlit for network access:

```bash
streamlit run ui/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Then open these URLs from another device on the same network:

```text
FastAPI docs: http://192.168.100.59:8000/docs
Streamlit app: http://192.168.100.59:8501
```

If the browser cannot connect, Windows Firewall may be blocking the ports. Allow
Python through Windows Firewall, or create firewall rules for ports `8000` and
`8501`.

PowerShell as Administrator:

```powershell
New-NetFirewallRule -DisplayName "NLP FastAPI 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
New-NetFirewallRule -DisplayName "NLP Streamlit 8501" -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow
```

Important note: this is local network deployment, not public internet hosting.
Only devices connected to the same network can access it.

## API Endpoints

### Health

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{"status":"ok"}
```

### Search

```bash
curl -X POST http://127.0.0.1:8000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"space shuttle nasa orbit\",\"top_k\":3}"
```

Linux/macOS:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"space shuttle nasa orbit","top_k":3}'
```

### Predict

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"The graphics card renders 3D images quickly.\"}"
```

Linux/macOS:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"The graphics card renders 3D images quickly."}'
```

### Labels

```bash
curl http://127.0.0.1:8000/labels
```

### Metrics

```bash
curl http://127.0.0.1:8000/metrics
```

## Example Search Queries

Try these in Streamlit or through `/search`:

- `space shuttle nasa orbit`
- `windows graphics card driver`
- `baseball team season game`
- `religion belief christian god`
- `encryption privacy security key`

## Example Output

A search request returns ranked documents:

```json
{
  "query": "space shuttle nasa orbit",
  "top_k": 3,
  "results": [
    {
      "rank": 1,
      "document_id": 8570,
      "similarity_score": 0.42,
      "predicted_label": "sci.space",
      "snippet": "..."
    }
  ]
}
```

A prediction request returns a class index and label:

```json
{
  "class_index": 1,
  "class_label": "comp.graphics"
}
```

The exact results and metric values may vary slightly depending on installed
library versions.

Current saved evaluation summary:

```text
Selected model: logistic_regression
Multinomial Naive Bayes accuracy: 0.6832, macro F1: 0.6568
Logistic Regression accuracy: 0.6905, macro F1: 0.6767
```
