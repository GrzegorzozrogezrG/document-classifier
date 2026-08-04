import io
import json
import re
import time
import base64
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import google.generativeai as genai
from openai import OpenAI

_MAX_RETRIES = 5
_BASE_WAIT = 30  # sekund — domyslny czas oczekiwania gdy API nie poda retry_delay


ANALYSIS_PROMPT = (
    "Analyze the provided document. "
    "Return ONLY valid JSON - no markdown, no code fences, no extra keys - "
    "with exactly this schema: "
    '{"document_type":"invoice|contract|id_card|report|article|form|receipt|presentation|other",'
    '"primary_content":"text|image|mixed",'
    '"text_percentage":<integer 0-100>,'
    '"opis_zawartosci":"krotkie zdanie po POLSKU opisujace zawartosc dokumentu",'
    '"tags":["tag1_po_polsku","tag2_po_polsku"],'
    '"suggested_group":"nazwa_grupy_po_polsku_bez_spacji"}'
    " Values for opis_zawartosci, tags and suggested_group MUST be in Polish."
)


TEXT_ANALYSIS_PROMPT_TEMPLATE = (
    "Analyze the provided document text content. "
    "Return ONLY valid JSON - no markdown, no code fences, no extra keys - "
    "with exactly this schema: "
    '{"document_type":"invoice|contract|id_card|report|article|form|receipt|presentation|other",'
    '"primary_content":"text|image|mixed",'
    '"text_percentage":<integer 0-100>,'
    '"opis_zawartosci":"krotkie zdanie po POLSKU opisujace zawartosc dokumentu",'
    '"tags":["tag1_po_polsku","tag2_po_polsku"],'
    '"suggested_group":"nazwa_grupy_po_polsku_bez_spacji"}'
    " Values for opis_zawartosci, tags and suggested_group MUST be in Polish."
    "\n\nDocument text:\n{content}"
)

CLUSTER_NAME_PROMPT = (
    "Na podstawie ponizszych opisow dokumentow nadaj krotka, trafna nazwe grupy "
    "w jezyku polskim (2-4 slowa, uzywaj podkreslnikow zamiast spacji, "
    "bez polskich znakow diakrytycznych). "
    "Zwroc TYLKO nazwe grupy, bez zadnych dodatkowych slow ani znakow.\n\nOpisy:\n{descriptions}"
)


def _parse_retry_delay(error_text: str) -> float:
    """Wyciaga retry_delay z komunikatu bledu 429, lub zwraca wartosc domyslna."""
    match = re.search(r"retry[_\s]delay[^0-9]*(\d+(?:\.\d+)?)", str(error_text), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0  # +2s margines
    return float(_BASE_WAIT)


def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc) or "quota" in str(exc).lower() or "rate" in str(exc).lower()


def detect_api_provider(api_key: str) -> str:
    key = (api_key or "").strip()
    if key.startswith("AIza"):
        return "gemini"
    if key.startswith("sk-"):
        return "openai"
    return "unknown"


def _extract_json_block(raw_text: str) -> Dict[str, Any]:
    # Proba parsowania calego tekstu jako JSON
    try:
        result = json.loads(raw_text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Szukaj zachlannie od pierwszego { do ostatniego }
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(raw_text[start : end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Ostatnia szansa: jesli tekst wyglada jak pola JSON bez nawiasow, otocz je
    stripped = raw_text.strip()
    if stripped and not stripped.startswith("{"):
        try:
            result = json.loads("{" + stripped.rstrip(",") + "}")
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return {}


def _normalize_tags(tags: Any) -> List[str]:
    if isinstance(tags, list):
        return [str(tag).strip() for tag in tags if str(tag).strip()]
    if isinstance(tags, str):
        return [part.strip() for part in tags.split(",") if part.strip()]
    return []


def validate_api_key(api_key: str, provider: Optional[str] = None) -> None:
    """Raise ValueError if the API key looks invalid before making a call."""
    key = api_key.strip()
    provider_name = provider or detect_api_provider(key)

    if not key:
        raise ValueError("Klucz API jest pusty.")
    if len(key) < 20:
        raise ValueError(f"Klucz API wydaje sie za krotki ({len(key)} znakow). Sprawdz czy skopiowales caly klucz.")
    if " " in key or "\n" in key:
        raise ValueError("Klucz API zawiera niedozwolone biale znaki. Skopiuj klucz ponownie.")
    if provider_name == "gemini" and not key.startswith("AIza"):
        raise ValueError("Klucz nie wyglada jak Gemini API key (powinien zaczynac sie od 'AIza').")
    if provider_name == "openai" and not key.startswith("sk-"):
        raise ValueError("Klucz nie wyglada jak OpenAI API key (powinien zaczynac sie od 'sk-').")
    if provider_name == "unknown":
        raise ValueError("Nie rozpoznano dostawcy klucza API. Uzyj klucza Gemini (AIza...) lub OpenAI (sk-...).")


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Oczekiwano slownika JSON, otrzymano: {type(payload).__name__} = {repr(payload)[:200]}")
    opis = str(payload.get("opis_zawartosci", payload.get("content_summary", ""))).strip()
    text_percentage = payload.get("text_percentage", 50)
    try:
        text_percentage = int(text_percentage)
    except (ValueError, TypeError):
        text_percentage = 50
    text_percentage = min(max(text_percentage, 0), 100)

    return {
        "document_type": str(payload.get("document_type", "other")).strip().lower(),
        "primary_content": str(payload.get("primary_content", "mixed")).strip().lower(),
        "text_percentage": text_percentage,
        "opis_zawartosci": opis,
        "content_summary": opis,
        "tags": _normalize_tags(payload.get("tags", [])),
        "suggested_group": str(payload.get("suggested_group", "general")).strip().lower() or "general",
    }


def analyze_image_with_gemini(
    api_key: str,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
    filename: str,
    on_rate_limit: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """Analyze an image document with Gemini and return normalized payload."""
    clean_key = api_key.strip()
    validate_api_key(clean_key, provider="gemini")

    genai.configure(api_key=clean_key)

    model = genai.GenerativeModel(model_name.strip())

    image_part = {"mime_type": mime_type, "data": image_bytes}
    last_exc: Exception = RuntimeError("Nieznany blad")

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                [ANALYSIS_PROMPT, image_part],
                generation_config=genai.GenerationConfig(temperature=0.1, response_mime_type="application/json"),
            )
            raw_text = response.text or ""
            payload = _extract_json_block(raw_text)
            if not payload:
                raise ValueError(f"Gemini zwrocil nieoczekiwany format: {raw_text[:300]}")
            break  # sukces — wychodzimy z petli
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit(exc) and attempt < _MAX_RETRIES:
                wait = _parse_retry_delay(exc)
                if on_rate_limit:
                    on_rate_limit(filename=filename, wait=wait, attempt=attempt)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Blad Gemini dla '{filename}': {exc}") from exc
    else:
        raise RuntimeError(f"Blad Gemini dla '{filename}' po {_MAX_RETRIES} probach: {last_exc}") from last_exc

    return _normalize_payload(payload)


def analyze_image_with_openai(
    api_key: str,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
    filename: str,
    on_rate_limit: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """Analyze an image document with OpenAI and return normalized payload."""
    clean_key = api_key.strip()
    validate_api_key(clean_key, provider="openai")

    client = OpenAI(api_key=clean_key)
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    image_url = f"data:{mime_type};base64,{image_b64}"

    last_exc: Exception = RuntimeError("Nieznany blad")

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=model_name.strip(),
                temperature=0,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": ANALYSIS_PROMPT},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    }
                ],
            )

            raw_text = getattr(response, "output_text", "") or ""
            if not raw_text:
                raw_text = str(response)
            payload = _extract_json_block(raw_text)
            if not payload:
                raise ValueError(f"OpenAI zwrocil nieoczekiwany format: {raw_text[:300]}")
            return _normalize_payload(payload)
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit(exc) and attempt < _MAX_RETRIES:
                wait = _parse_retry_delay(exc)
                if on_rate_limit:
                    on_rate_limit(filename=filename, wait=wait, attempt=attempt)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Blad OpenAI dla '{filename}': {exc}") from exc

    raise RuntimeError(f"Blad OpenAI dla '{filename}' po {_MAX_RETRIES} probach: {last_exc}") from last_exc


def analyze_pdf_with_openai(
    api_key: str,
    model_name: str,
    pdf_bytes: bytes,
    filename: str,
    on_rate_limit: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """Analyze a PDF document with OpenAI using the Responses API (input_file).
    Falls back to text extraction if the model does not support file input.
    """
    clean_key = api_key.strip()
    validate_api_key(clean_key, provider="openai")
    client = OpenAI(api_key=clean_key)
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    last_exc: Exception = RuntimeError("Nieznany blad")

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=model_name.strip(),
                temperature=0,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": ANALYSIS_PROMPT},
                            {
                                "type": "input_file",
                                "filename": filename,
                                "file_data": f"data:application/pdf;base64,{pdf_b64}",
                            },
                        ],
                    }
                ],
            )
            raw_text = getattr(response, "output_text", "") or ""
            if not raw_text:
                raw_text = str(response)
            payload = _extract_json_block(raw_text)
            if not payload:
                raise ValueError(f"OpenAI zwrocil nieoczekiwany format: {raw_text[:300]}")
            return _normalize_payload(payload)
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit(exc) and attempt < _MAX_RETRIES:
                wait = _parse_retry_delay(exc)
                if on_rate_limit:
                    on_rate_limit(filename=filename, wait=wait, attempt=attempt)
                time.sleep(wait)
                continue
            # Jesli model nie obsluguje input_file, sprobuj ekstrakcji tekstu
            err_str = str(exc).lower()
            if "input_file" in err_str or "unsupported" in err_str or "file" in err_str:
                text = _pdf_to_text(pdf_bytes)
                if text:
                    return analyze_text_with_openai(api_key, model_name, text, filename, on_rate_limit)
            raise RuntimeError(f"Blad OpenAI PDF dla '{filename}': {exc}") from exc

    raise RuntimeError(f"Blad OpenAI PDF dla '{filename}' po {_MAX_RETRIES} probach: {last_exc}") from last_exc
    """Extract text from PDF using pypdf. Returns empty string on failure."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        return ""


def analyze_text_with_gemini(
    api_key: str,
    model_name: str,
    text_content: str,
    filename: str,
    on_rate_limit: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """Analyze plain text document with Gemini."""
    clean_key = api_key.strip()
    validate_api_key(clean_key, provider="gemini")
    genai.configure(api_key=clean_key)
    model = genai.GenerativeModel(model_name.strip())
    prompt = TEXT_ANALYSIS_PROMPT_TEMPLATE.format(content=text_content[:8000])
    last_exc: Exception = RuntimeError("Nieznany blad")
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                [prompt],
                generation_config=genai.GenerationConfig(temperature=0.1, response_mime_type="application/json"),
            )
            raw_text = response.text or ""
            payload = _extract_json_block(raw_text)
            if not payload:
                raise ValueError(f"Gemini zwrocil nieoczekiwany format: {raw_text[:300]}")
            return _normalize_payload(payload)
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit(exc) and attempt < _MAX_RETRIES:
                wait = _parse_retry_delay(exc)
                if on_rate_limit:
                    on_rate_limit(filename=filename, wait=wait, attempt=attempt)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Blad Gemini dla '{filename}': {exc}") from exc
    raise RuntimeError(f"Blad Gemini dla '{filename}' po {_MAX_RETRIES} probach: {last_exc}") from last_exc


def analyze_text_with_openai(
    api_key: str,
    model_name: str,
    text_content: str,
    filename: str,
    on_rate_limit: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """Analyze plain text document with OpenAI."""
    clean_key = api_key.strip()
    validate_api_key(clean_key, provider="openai")
    client = OpenAI(api_key=clean_key)
    prompt = TEXT_ANALYSIS_PROMPT_TEMPLATE.format(content=text_content[:8000])
    last_exc: Exception = RuntimeError("Nieznany blad")
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            # Proba z JSON mode; jesli model go nie obsluguje, fallback bez niego
            try:
                response = client.chat.completions.create(
                    model=model_name.strip(),
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "You are a document analysis assistant. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                )
            except Exception:
                response = client.chat.completions.create(
                    model=model_name.strip(),
                    temperature=0,
                    messages=[
                        {"role": "system", "content": "You are a document analysis assistant. Always respond with valid JSON only, no markdown."},
                        {"role": "user", "content": prompt},
                    ],
                )
            raw_text = (response.choices[0].message.content or "").strip()
            payload = _extract_json_block(raw_text)
            if not isinstance(payload, dict) or not payload:
                raise ValueError(
                    f"OpenAI zwrocil nieoczekiwany format dla '{filename}'.\n"
                    f"Model: {model_name}\nOdpowiedz (pierwsze 400 znakow):\n{raw_text[:400]}"
                )
            return _normalize_payload(payload)
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit(exc) and attempt < _MAX_RETRIES:
                wait = _parse_retry_delay(exc)
                if on_rate_limit:
                    on_rate_limit(filename=filename, wait=wait, attempt=attempt)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Blad OpenAI dla '{filename}': {exc}") from exc
    raise RuntimeError(f"Blad OpenAI dla '{filename}' po {_MAX_RETRIES} probach: {last_exc}") from last_exc


def name_clusters_with_ai(
    api_key: str,
    model_name: str,
    cluster_descriptions: Dict[str, List[str]],
) -> Dict[str, str]:
    """Generate human-readable Polish names for each cluster via AI.
    Returns {cluster_label: name}.
    """
    provider = detect_api_provider(api_key)
    clean_key = api_key.strip()
    names: Dict[str, str] = {}
    for label, descriptions in cluster_descriptions.items():
        sample = "\n".join(f"- {d}" for d in descriptions[:5] if str(d).strip())
        if not sample:
            names[str(label)] = f"klaster_{label}"
            continue
        prompt = CLUSTER_NAME_PROMPT.format(descriptions=sample)
        try:
            if provider == "gemini":
                genai.configure(api_key=clean_key)
                m = genai.GenerativeModel(model_name.strip())
                resp = m.generate_content(
                    [prompt],
                    generation_config=genai.GenerationConfig(temperature=0.3),
                )
                raw = (resp.text or "").strip().splitlines()[0].strip()
            elif provider == "openai":
                c = OpenAI(api_key=clean_key)
                resp = c.chat.completions.create(
                    model=model_name.strip(),
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = (resp.choices[0].message.content or "").strip().splitlines()[0].strip()
            else:
                raw = ""
            name = re.sub(r"[^\w]", "_", raw).strip("_") or f"klaster_{label}"
            names[str(label)] = name
        except Exception:
            names[str(label)] = f"klaster_{label}"
    return names


def analyze_document(
    api_key: str,
    model_name: str,
    file_bytes: bytes,
    mime_type: str,
    filename: str,
    on_rate_limit: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """Unified entrypoint: handles images, PDFs, and text files."""
    provider = detect_api_provider(api_key)
    ext = Path(filename).suffix.lower() if filename else ""
    is_image = mime_type.startswith("image/") or ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    is_pdf = mime_type == "application/pdf" or ext == ".pdf"
    is_text = ext in {".txt", ".md", ".csv", ".log"} or (mime_type.startswith("text/") and not is_image)

    if is_image:
        if provider == "gemini":
            return analyze_image_with_gemini(api_key, model_name, file_bytes, mime_type, filename, on_rate_limit)
        if provider == "openai":
            return analyze_image_with_openai(api_key, model_name, file_bytes, mime_type, filename, on_rate_limit)
    elif is_pdf:
        if provider == "gemini":
            return analyze_image_with_gemini(api_key, model_name, file_bytes, "application/pdf", filename, on_rate_limit)
        if provider == "openai":
            return analyze_pdf_with_openai(api_key, model_name, file_bytes, filename, on_rate_limit)
    elif is_text:
        text = file_bytes.decode("utf-8", errors="replace")
        if provider == "gemini":
            return analyze_text_with_gemini(api_key, model_name, text, filename, on_rate_limit)
        if provider == "openai":
            return analyze_text_with_openai(api_key, model_name, text, filename, on_rate_limit)

    raise ValueError(f"Nieobslugiwany typ pliku: {mime_type} ({filename})")


# backward-compatible alias
analyze_image_document = analyze_document