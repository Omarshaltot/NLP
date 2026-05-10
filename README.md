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
