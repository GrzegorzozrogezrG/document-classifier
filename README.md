# Klasyfikator dokumentów (Streamlit + AI + PyCaret)

Aplikacja porządkuje dokumenty na podstawie ich treści. Użytkownik przesyła obrazy
lub pliki PDF, model AI rozpoznaje ich typ i opisuje zawartość, a następnie
PyCaret grupuje podobne materiały. Wyniki można ręcznie poprawić i pobrać jako
archiwum `.7z` z osobnym katalogiem dla każdej grupy.

## Cel projektu

Projekt ma skrócić czas potrzebny na ręczne przeglądanie i segregowanie większej
liczby dokumentów. Pierwsza wersja produktu obejmuje pełny przepływ:

1. przesłanie dokumentów,
2. analizę treści przez Gemini lub OpenAI,
3. automatyczne grupowanie podobnych plików,
4. ręczną korektę przypisanych grup,
5. eksport uporządkowanych dokumentów.

## Features

- Klasyfikacja obrazów i dokumentów PDF przy użyciu Gemini lub OpenAI
- Metadata table in Pandas / Streamlit
- Similarity grouping with PyCaret (KMeans clustering)
- Manual group override by user
- Export grouped files as `.7z`

## Stack

- Streamlit
- Pandas
- PyCaret
- Google Gemini (`google-generativeai`)
- Py7zr

## Run

1. Create or activate your environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

4. W panelu bocznym wklej klucz API Gemini lub OpenAI. Kluczy i innych sekretów
   nie zapisuj w repozytorium.

## Notes

- Supported file types: png, jpg, jpeg, webp, bmp, tiff, pdf
- `user_group` can be edited before export
- Export uses a folder-per-group structure inside the `.7z` archive

## Current Status (app.py)

- `app.py` starts correctly with the project virtual environment.
- The previous interpreter issue (`No Python at ...`) was caused by a stale venv base interpreter path and is resolved after recreating `.venv`.
- The application should be launched with `streamlit run app.py` (running `python app.py` only shows Streamlit "bare mode" warnings).
- Main flow currently works: upload -> analyze -> cluster -> manual group edit -> export `.7z`.

## To Do

### MVP (Next)

- Migrate from deprecated `google-generativeai` to the newer `google-genai` SDK.
- Improve runtime error reporting in the UI for external API/network failures.
- Add dependency/version pinning strategy to reduce resolver backtracking and environment drift.

### Later (Stabilization)

- Add automated tests for provider detection, document analysis fallback paths, clustering output, and archive export.
- Add CI checks (lint + basic smoke test) to validate that the app starts and core imports work.
## Project status

Development work is tracked on feature branches and reviewed through pull requests.
