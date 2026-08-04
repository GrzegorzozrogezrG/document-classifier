import io
import base64
import hashlib
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_analyzer import analyze_document, detect_api_provider, validate_api_key, name_clusters_with_ai
from src.grouping import cluster_documents
from src.packaging import create_7z_archive


st.set_page_config(page_title="Document Classifier", page_icon="📄", layout="wide")

QDRANT_COLLECTION = "session_state_store"
QDRANT_VECTOR = [0.0, 0.0, 0.0, 0.0]


DOC_TYPE_PL = {
    "invoice": "faktura",
    "contract": "umowa",
    "id_card": "dowod tozsamosci",
    "report": "raport",
    "article": "artykul",
    "form": "formularz",
    "receipt": "paragon",
    "presentation": "prezentacja",
    "other": "inne",
    "unknown": "nieznany",
}

PRIMARY_CONTENT_PL = {
    "text": "tekst",
    "image": "obraz",
    "mixed": "mieszany",
}


def _translate_classification_to_polish(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "document_type" in output.columns:
        output["document_type"] = (
            output["document_type"].astype(str).str.strip().str.lower().map(DOC_TYPE_PL).fillna(output["document_type"])
        )
    if "primary_content" in output.columns:
        output["primary_content"] = (
            output["primary_content"].astype(str).str.strip().str.lower().map(PRIMARY_CONTENT_PL).fillna(output["primary_content"])
        )
    return output


def _init_state() -> None:
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if "loaded_session_id" not in st.session_state:
        st.session_state.loaded_session_id = ""
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gemini-2.5-flash"
    if "num_clusters" not in st.session_state:
        st.session_state.num_clusters = 4
    if "archive_name" not in st.session_state:
        st.session_state.archive_name = "classified_documents.7z"


def _get_qdrant_client() -> Optional[QdrantClient]:
    """Create Qdrant client from Streamlit secrets; return None when not configured."""
    try:
        qdrant_url = st.secrets.get("QDRANT_URL", st.secrets.get("qdrant_url", ""))
        qdrant_api_key = st.secrets.get("QDRANT_API_KEY", st.secrets.get("qdrant_api_key", ""))
    except Exception:
        return None
    if not qdrant_url or not qdrant_api_key:
        return None
    try:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    except Exception:
        return None


def _ensure_qdrant_collection(client: QdrantClient) -> None:
    try:
        exists = client.collection_exists(QDRANT_COLLECTION)
    except Exception:
        exists = False
    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=len(QDRANT_VECTOR), distance=Distance.COSINE),
        )


def _session_id(provider: str, api_key: str) -> str:
    raw = f"{provider}:{api_key.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _serialize_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for doc in documents:
        item = dict(doc)
        raw_bytes = item.get("raw_bytes", b"")
        item["raw_bytes"] = base64.b64encode(raw_bytes).decode("ascii") if isinstance(raw_bytes, (bytes, bytearray)) else ""
        serialized.append(item)
    return serialized


def _deserialize_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deserialized: List[Dict[str, Any]] = []
    for doc in documents:
        item = dict(doc)
        raw_b64 = item.get("raw_bytes", "")
        try:
            item["raw_bytes"] = base64.b64decode(raw_b64.encode("ascii")) if raw_b64 else b""
        except Exception:
            item["raw_bytes"] = b""
        deserialized.append(item)
    return deserialized


def _normalize_state_schema(saved: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate old keys (content_summary) to new key (opis_zawartosci)."""
    migrated = dict(saved)
    results = [dict(row) for row in migrated.get("results", [])]
    documents = [dict(row) for row in migrated.get("documents", [])]

    for row in results:
        if "opis_zawartosci" not in row:
            row["opis_zawartosci"] = row.get("content_summary", "")
    for row in documents:
        if "opis_zawartosci" not in row:
            row["opis_zawartosci"] = row.get("content_summary", "")

    migrated["results"] = results
    migrated["documents"] = documents
    return migrated


def _load_state_from_qdrant(sid: str) -> Optional[Dict[str, Any]]:
    client = _get_qdrant_client()
    if client is None:
        return None
    try:
        _ensure_qdrant_collection(client)
        points = client.retrieve(collection_name=QDRANT_COLLECTION, ids=[sid], with_payload=True)
        if not points:
            return None
        state = points[0].payload.get("state") if points[0].payload else None
        if not isinstance(state, dict):
            return None
        return _normalize_state_schema(state)
    except Exception:
        return None


def _save_state_to_qdrant(sid: str, state: Dict[str, Any]) -> bool:
    client = _get_qdrant_client()
    if client is None:
        return False
    try:
        _ensure_qdrant_collection(client)
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=sid,
                    vector=QDRANT_VECTOR,
                    payload={
                        "session_id": sid,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "state": state,
                    },
                )
            ],
        )
        return True
    except Exception:
        return False


def _restore_state_for_key(provider: str, api_key: str) -> None:
    if provider not in {"gemini", "openai"}:
        return
    sid = _session_id(provider, api_key)
    if st.session_state.loaded_session_id == sid:
        return

    saved = _load_state_from_qdrant(sid)
    st.session_state.loaded_session_id = sid

    if not saved:
        return

    st.session_state.documents = _deserialize_documents(saved.get("documents", []))
    restored_df = pd.DataFrame(saved.get("results", []))
    if "opis_zawartosci" not in restored_df.columns and "content_summary" in restored_df.columns:
        restored_df["opis_zawartosci"] = restored_df["content_summary"]
    st.session_state.results_df = restored_df
    st.session_state.model_name = saved.get("model_name", st.session_state.model_name)
    st.session_state.num_clusters = int(saved.get("num_clusters", st.session_state.num_clusters))
    st.session_state.archive_name = saved.get("archive_name", st.session_state.archive_name)
    st.toast("Przywrocono ostatnia sesje dla tego klucza API.")


def _persist_state_for_key(provider: str, api_key: str, model_name: str, num_clusters: int, archive_name: str) -> None:
    if provider not in {"gemini", "openai"}:
        return
    sid = _session_id(provider, api_key)
    state = {
        "provider": provider,
        "model_name": model_name,
        "num_clusters": int(num_clusters),
        "archive_name": archive_name,
        "results": st.session_state.results_df.to_dict(orient="records") if not st.session_state.results_df.empty else [],
        "documents": _serialize_documents(st.session_state.documents),
    }
    _save_state_to_qdrant(sid, state)


def _build_records(files: List[Any], api_key: str, model_name: str, provider: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    progress = st.progress(0, text="Przygotowywanie...")
    status_box = st.empty()

    def _on_rate_limit(filename: str, wait: float, attempt: int) -> None:
        for remaining in range(int(wait), 0, -1):
            status_box.warning(
                f"⏳ Limit API (429) dla '{filename}' — "
                f"proba {attempt}/{5}, czekam {remaining}s..."
            )
            import time as _t; _t.sleep(1)
        status_box.empty()

    for idx, file in enumerate(files):
        progress.progress(idx / len(files), text=f"Analizuje: {file.name}")
        file_bytes = file.getvalue()
        try:
            analysis = analyze_document(
                api_key=api_key,
                model_name=model_name,
                file_bytes=file_bytes,
                mime_type=file.type or "application/octet-stream",
                filename=file.name,
                on_rate_limit=_on_rate_limit,
            )
        except Exception as exc:
            st.error(f"Blad analizy '{file.name}': {exc}")
            raise

        record = {
            "doc_id": idx,
            "filename": file.name,
            "mime_type": file.type or "image/png",
            "document_type": analysis.get("document_type", "unknown"),
            "primary_content": analysis.get("primary_content", "mixed"),
            "text_percentage": int(analysis.get("text_percentage", 50)),
            "opis_zawartosci": analysis.get("opis_zawartosci", analysis.get("content_summary", "")),
            "tags": ", ".join(analysis.get("tags", [])),
            "suggested_group": analysis.get("suggested_group", "general"),
            "raw_bytes": file_bytes,
        }
        records.append(record)
    progress.progress(1.0, text="Zakończono.")
    return records


def main() -> None:
    _init_state()

    st.title("Klasyfikacja i grupowanie dokumentow obrazowych")
    st.caption("Streamlit + Gemini/OpenAI + Pandas + PyCaret + eksport 7z")

    with st.sidebar:
        st.header("Ustawienia")
        api_key = st.text_input(
            "API key (Gemini lub OpenAI)",
            type="password",
            help="Gemini: AIza... | OpenAI: sk-...",
        )

        provider = detect_api_provider(api_key) if api_key.strip() else "unknown"
        if provider == "gemini":
            st.success("Wykryto klucz: Gemini")
        elif provider == "openai":
            st.success("Wykryto klucz: OpenAI")
        elif api_key.strip():
            st.warning("Nie rozpoznano dostawcy klucza API.")

        if _get_qdrant_client() is not None:
            st.caption("Session storage: Qdrant ✓")
        else:
            st.caption("Session storage: niedostepny")

        if api_key.strip() and provider in {"gemini", "openai"}:
            _restore_state_for_key(provider, api_key)

        if provider == "openai":
            model_options = ["gpt-4o-mini", "gpt-4.1-nano"]
        else:
            model_options = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]

        saved_model = st.session_state.get("model_name", model_options[0])
        selected_index = model_options.index(saved_model) if saved_model in model_options else 0
        model_name = st.selectbox(
            "Model",
            options=model_options,
            index=selected_index,
            help="Model dobierany do wykrytego dostawcy API.",
        )

        num_clusters = st.slider(
            "Liczba klastrow",
            min_value=2,
            max_value=10,
            value=int(st.session_state.get("num_clusters", 4)),
        )
        archive_name = st.text_input("Nazwa archiwum", value=st.session_state.get("archive_name", "classified_documents.7z"))

        st.session_state.model_name = model_name
        st.session_state.num_clusters = num_clusters
        st.session_state.archive_name = archive_name

    uploaded_files = st.file_uploader(
        "Dodaj pliki dokumentow (obrazy, PDF, tekst)",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "pdf", "txt", "csv", "md"],
        accept_multiple_files=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        analyze_clicked = st.button("1) Analizuj dokumenty", use_container_width=True)
    with col_b:
        cluster_clicked = st.button("2) Grupuj podobne", use_container_width=True)

    if analyze_clicked:
        if not api_key:
            st.error("Podaj API key w panelu bocznym.")
            st.stop()
        try:
            validate_api_key(api_key, provider=provider)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        if not uploaded_files:
            st.error("Dodaj co najmniej jeden plik.")
            st.stop()

        try:
            with st.spinner(f"Analizuje dokumenty przez {provider}..."):
                docs = _build_records(uploaded_files, api_key, model_name, provider=provider)
        except Exception:
            st.stop()

        st.session_state.documents = docs
        base_df = pd.DataFrame(docs).drop(columns=["raw_bytes"])
        base_df["group"] = base_df["suggested_group"]
        st.session_state.results_df = base_df
        st.success(f"Przeanalizowano {len(docs)} plikow.")

        _persist_state_for_key(provider, api_key, model_name, num_clusters, archive_name)

    if cluster_clicked:
        if st.session_state.results_df.empty:
            st.error("Najpierw uruchom analize dokumentow.")
        else:
            sample_count = len(st.session_state.results_df)
            safe_clusters = max(1, min(int(num_clusters), sample_count))
            if safe_clusters != int(num_clusters):
                st.info(
                    f"Zmieniono liczbe klastrow z {num_clusters} na {safe_clusters}, "
                    f"bo liczba dokumentow wynosi {sample_count}."
                )
            with st.spinner("Uruchamiam grupowanie PyCaret..."):
                clustered_df = cluster_documents(st.session_state.results_df.copy(), num_clusters=safe_clusters)
            # AI generuje nazwy klastrów
            desc_col = "opis_zawartosci"
            cluster_descs: dict = {}
            for lbl in clustered_df["cluster_label"].unique():
                descs = clustered_df[clustered_df["cluster_label"] == lbl][desc_col].dropna().tolist()
                cluster_descs[str(lbl)] = [str(d) for d in descs]
            with st.spinner("AI generuje nazwy grup..."):
                cluster_names = name_clusters_with_ai(api_key, model_name, cluster_descs)
            clustered_df["group"] = (
                clustered_df["cluster_label"].astype(str).map(cluster_names).fillna(clustered_df["cluster_label"].astype(str))
            )
            st.session_state.results_df = clustered_df
            st.success("Grupowanie zakonczone.")
            _persist_state_for_key(provider, api_key, model_name, num_clusters, archive_name)

    if not st.session_state.results_df.empty:
        st.subheader("Wyniki klasyfikacji")
        full_df = st.session_state.results_df.copy()

        # Zapewnij kolumne group
        if "group" not in full_df.columns:
            full_df["group"] = full_df.get("suggested_group", "ogolne")

        editable_cols = [
            "doc_id",
            "filename",
            "document_type",
            "primary_content",
            "text_percentage",
            "tags",
            "group",
        ]

        df_to_show = full_df.copy()
        df_to_show = _translate_classification_to_polish(df_to_show)
        for col in editable_cols:
            if col not in df_to_show.columns:
                df_to_show[col] = ""

        edited_df = st.data_editor(
            df_to_show[editable_cols],
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "doc_id": st.column_config.NumberColumn("ID dokumentu", format="%d", disabled=True),
                "filename": st.column_config.TextColumn("Nazwa pliku", disabled=True),
                "document_type": st.column_config.TextColumn("Typ dokumentu", disabled=True),
                "primary_content": st.column_config.TextColumn("Glowna zawartosc", disabled=True),
                "text_percentage": st.column_config.NumberColumn("Procent tekstu", format="%d", disabled=True),
                "tags": st.column_config.TextColumn("Tagi", disabled=True),
                "group": st.column_config.TextColumn("Grupa", help="Mozesz recznie nadpisac grupe"),
            },
        )

        # Aktualizuj tylko kolumne group z edytora
        group_map = edited_df.set_index("doc_id")["group"].to_dict()
        full_df["group"] = full_df["doc_id"].map(group_map).fillna(full_df["group"])
        st.session_state.results_df = full_df

        # Filtrowanie wg grupy (dostepne po klasteryzacji)
        if "cluster_label" in full_df.columns:
            unique_groups = sorted(full_df["group"].dropna().astype(str).unique().tolist())
            filter_options = ["(wszystkie)"] + unique_groups
            selected_filter = st.selectbox(
                "Filtruj wyniki wedlug grupy",
                options=filter_options,
                key="cluster_filter",
            )
            if selected_filter != "(wszystkie)":
                filtered_view = _translate_classification_to_polish(
                    full_df[full_df["group"].astype(str) == selected_filter].copy()
                )
                display_cols = [c for c in editable_cols if c in filtered_view.columns]
                st.dataframe(filtered_view[display_cols], use_container_width=True, hide_index=True)

        st.subheader("Opis zawartosci dokumentu")
        if "opis_zawartosci" not in df_to_show.columns:
            df_to_show["opis_zawartosci"] = df_to_show.get("content_summary", "")
        summary_df = df_to_show[["doc_id", "filename", "opis_zawartosci"]].copy()
        summary_df["summary_label"] = summary_df.apply(
            lambda row: f"[{int(row['doc_id'])}] {row['filename']}", axis=1
        )
        selected_summary = st.selectbox(
            "Wybierz dokument",
            options=summary_df["summary_label"].tolist(),
            index=0,
        )
        selected_row = summary_df[summary_df["summary_label"] == selected_summary].iloc[0]
        st.text_area(
            "Opis zawartosci",
            value=str(selected_row["opis_zawartosci"]),
            height=180,
            disabled=True,
        )

        all_groups = sorted([str(g) for g in full_df["group"].fillna("ogolne").unique()])
        selected_groups = st.multiselect(
            "Wybierz grupy do eksportu",
            options=all_groups,
            default=all_groups,
        )

        if api_key.strip() and provider in {"gemini", "openai"}:
            _persist_state_for_key(provider, api_key, model_name, num_clusters, archive_name)

        if st.button("3) Pobierz archiwum .7z", use_container_width=True):
            selected = full_df[full_df["group"].astype(str).isin(selected_groups)]
            by_id = {int(row["doc_id"]): row for _, row in selected.iterrows()}

            export_items: List[Dict[str, Any]] = []
            for doc in st.session_state.documents:
                doc_id = int(doc["doc_id"])
                if doc_id not in by_id:
                    continue
                export_items.append(
                    {
                        "filename": doc["filename"],
                        "raw_bytes": doc["raw_bytes"],
                        "user_group": str(by_id[doc_id]["group"]),
                    }
                )

            if not export_items:
                st.error("Brak plikow do eksportu. Wybierz przynajmniej jedna grupe.")
                st.stop()

            archive_bytes = create_7z_archive(export_items, archive_name=archive_name)
            st.download_button(
                label="Kliknij aby pobrac 7z",
                data=io.BytesIO(archive_bytes),
                file_name=archive_name,
                mime="application/x-7z-compressed",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()