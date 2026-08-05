# Document Classifier (Streamlit + AI + PyCaret)

This application organizes documents based on their content. Users upload images
or PDF files, an AI model identifies their type and describes their content, and
PyCaret groups similar materials. Users can manually adjust the results and
download them as a `.7z` archive with a separate directory for each group.

## Project Goal

The project aims to reduce the time required to manually review and sort large
collections of documents. The first product version covers the complete workflow:

1. Upload documents.
2. Analyze their content with Gemini or OpenAI.
3. Automatically group similar files.
4. Manually adjust the assigned groups.
5. Export the organized documents.

## MVP Scope

The MVP is intended for users who need to organize a small or medium-sized batch
of mixed business documents without building classification rules manually.

### Included

- Upload PNG, JPG, JPEG, WebP, BMP, TIFF, and PDF files.
- Analyze documents with a user-provided Gemini or OpenAI API key.
- Return a document type, content description, tags, and suggested group.
- Group similar documents automatically with KMeans clustering.
- Display analysis results in an editable table.
- Allow users to correct group assignments before export.
- Export selected documents to a `.7z` archive organized by group.
- Show actionable errors for invalid files, API failures, and malformed AI responses.

### Not Included

- User accounts, roles, or shared workspaces.
- Permanent document storage or document-history management.
- Training custom classification models.
- OCR or classification guarantees for handwritten or low-quality documents.
- Automatic processing from email, cloud drives, or external systems.
- Production-scale batch processing and background job queues.

## Success Criteria

The MVP is successful when all of the following conditions are met:

- A user can complete the upload-to-export workflow without editing source code.
- At least 90% of supported, valid test files are processed without an application error.
- Every successfully analyzed document receives all required metadata fields.
- Users can review and change every automatically assigned group.
- The exported archive contains every selected file exactly once in its chosen group.
- Invalid files and external API failures produce clear, actionable messages.
- A representative batch of 20 documents can be processed and exported in one session.

## Features

- Image and PDF document classification using Gemini or OpenAI
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

4. Paste your Gemini or OpenAI API key into the sidebar. Do not store API keys
   or other secrets in the repository.

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
## Project Status

Development work is tracked on feature branches and reviewed through pull requests.
