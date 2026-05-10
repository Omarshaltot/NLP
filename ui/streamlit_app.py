"""Streamlit GUI for the 20 Newsgroups search engine project."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "artifacts" / "confusion_matrix.png"


def get_api_base_url() -> str:
    default_api_url = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
    return st.sidebar.text_input("FastAPI URL", value=default_api_url).rstrip("/")


def call_api(method: str, endpoint: str, api_base_url: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(
        method=method,
        url=f"{api_base_url}{endpoint}",
        timeout=10,
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


def local_search(query: str, top_k: int) -> dict[str, Any]:
    from app.services.search_service import search_documents

    return {
        "query": query,
        "top_k": top_k,
        "results": search_documents(query=query, top_k=top_k),
    }


def local_predict(text: str) -> dict[str, Any]:
    from app.services.model_service import predict_category

    return predict_category(text)


def local_labels() -> dict[str, Any]:
    from app.services.model_service import get_labels

    return {"labels": get_labels()}


def local_metrics() -> dict[str, Any]:
    from app.services.model_service import get_metrics

    return {"metrics": get_metrics()}


def api_or_local(
    api_base_url: str,
    method: str,
    endpoint: str,
    local_function,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        return call_api(method, endpoint, api_base_url, **kwargs)
    except requests.RequestException:
        return local_function()


def main() -> None:
    st.set_page_config(page_title="20 Newsgroups Search Engine", layout="wide")

    st.title("Simple Search Engine")
    st.write(
        "TF-IDF document retrieval and classical ML text classification "
        "using the 20 Newsgroups dataset."
    )

    api_base_url = get_api_base_url()

    search_tab, predict_tab, metrics_tab, labels_tab = st.tabs(
        ["Search", "Classify", "Metrics", "Labels"]
    )

    with search_tab:
        query = st.text_input("Search query", value="space shuttle nasa orbit")
        top_k = st.slider("Top results", min_value=1, max_value=20, value=5)

        if st.button("Run search", type="primary"):
            if not query.strip():
                st.warning("Enter a search query.")
            else:
                payload = {"query": query, "top_k": top_k}
                try:
                    data = call_api("POST", "/search", api_base_url, json=payload)
                    st.caption("Results loaded from FastAPI.")
                except requests.RequestException:
                    data = local_search(query=query, top_k=top_k)
                    st.caption("FastAPI unavailable; results loaded from local artifacts.")

                for result in data["results"]:
                    st.subheader(
                        f"#{result['rank']} - {result['predicted_label']} "
                        f"({result['similarity_score']:.4f})"
                    )
                    st.write(result["snippet"])

    with predict_tab:
        text = st.text_area(
            "Text to classify",
            value="The graphics card driver renders 3D images very quickly.",
            height=160,
        )

        if st.button("Predict category", type="primary"):
            if not text.strip():
                st.warning("Enter text to classify.")
            else:
                payload = {"text": text}
                try:
                    prediction = call_api("POST", "/predict", api_base_url, json=payload)
                    st.caption("Prediction loaded from FastAPI.")
                except requests.RequestException:
                    prediction = local_predict(text)
                    st.caption("FastAPI unavailable; prediction loaded from local artifacts.")

                st.metric("Predicted label", prediction["class_label"])
                st.write(f"Class index: `{prediction['class_index']}`")

    with metrics_tab:
        try:
            metrics_data = call_api("GET", "/metrics", api_base_url)
            st.caption("Metrics loaded from FastAPI.")
        except requests.RequestException:
            metrics_data = local_metrics()
            st.caption("FastAPI unavailable; metrics loaded from local artifacts.")

        metrics = metrics_data["metrics"]
        selected = metrics.get("selected_model")
        selected_metrics = metrics.get("model_comparison", {}).get(selected, {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{selected_metrics.get('accuracy', 0):.4f}")
        col2.metric("Precision", f"{selected_metrics.get('precision_macro', 0):.4f}")
        col3.metric("Recall", f"{selected_metrics.get('recall_macro', 0):.4f}")
        col4.metric("F1-score", f"{selected_metrics.get('f1_macro', 0):.4f}")

        st.write(f"Selected model: `{selected}`")
        st.json(json.loads(json.dumps(metrics.get("model_comparison", {}))))

        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix")
        else:
            st.info("Run `python -m ml.train` to generate the confusion matrix image.")

    with labels_tab:
        try:
            labels_data = call_api("GET", "/labels", api_base_url)
            st.caption("Labels loaded from FastAPI.")
        except requests.RequestException:
            labels_data = local_labels()
            st.caption("FastAPI unavailable; labels loaded from local artifacts.")

        st.write(labels_data["labels"])


if __name__ == "__main__":
    main()
