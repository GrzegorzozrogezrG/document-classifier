from typing import List

import pandas as pd
from pycaret.clustering import assign_model, create_model, setup
from sklearn.feature_extraction.text import TfidfVectorizer


def _build_text_features(df: pd.DataFrame, max_features: int = 60) -> pd.DataFrame:
    summary_col = "opis_zawartosci" if "opis_zawartosci" in df.columns else "content_summary"
    corpus: List[str] = (
        df[summary_col].fillna("").astype(str)
        + " "
        + df["tags"].fillna("").astype(str)
        + " "
        + df["document_type"].fillna("").astype(str)
        + " "
        + df["primary_content"].fillna("").astype(str)
    ).tolist()

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    tfidf_df = pd.DataFrame(matrix.toarray(), columns=feature_names, index=df.index)
    return tfidf_df


def cluster_documents(df: pd.DataFrame, num_clusters: int = 4) -> pd.DataFrame:
    if df.empty:
        return df

    output = df.copy()
    if "opis_zawartosci" not in output.columns:
        output["opis_zawartosci"] = output.get("content_summary", "")
    if "content_summary" not in output.columns:
        output["content_summary"] = output["opis_zawartosci"]
    if "tags" not in output.columns:
        output["tags"] = ""
    if "document_type" not in output.columns:
        output["document_type"] = "other"
    if "primary_content" not in output.columns:
        output["primary_content"] = "mixed"

    output["summary_len"] = output["opis_zawartosci"].fillna("").astype(str).str.len()
    output["tag_count"] = output["tags"].fillna("").astype(str).apply(lambda s: 0 if not s else len(s.split(",")))
    output["text_percentage"] = pd.to_numeric(output["text_percentage"], errors="coerce").fillna(50)

    text_features = _build_text_features(output)

    numeric = output[["text_percentage", "summary_len", "tag_count"]].reset_index(drop=True)
    model_input = pd.concat([numeric, text_features.reset_index(drop=True)], axis=1)

    sample_count = len(model_input)
    safe_clusters = max(1, min(int(num_clusters), sample_count))

    setup(data=model_input, session_id=42, verbose=False, html=False, normalize=True)
    model = create_model("kmeans", num_clusters=safe_clusters)
    assigned = assign_model(model)

    output["cluster_label"] = assigned["Cluster"].astype(str)
    output["user_group"] = output["cluster_label"]
    return output