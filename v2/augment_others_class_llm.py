"""
Augment the \"Others\" class: rewrite medical-looking OCR text into non-medical styles via LLM.

Modes:
  jsonl  -- resume-friendly JSONL append (same as legacy data_aumented_others.py)
  tail   -- threaded run on the last N rows, persona + MEDICAL_OTHERS prompt, CSV out

Requires: pip install pandas requests tqdm
Set LONGCAT_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

LONGCAT_CHAT_COMPLETIONS_URL = "https://api.longcat.chat/openai/v1/chat/completions"
LLM_MODEL_NAME = "LongCat-Flash-Lite"

DEFAULT_INPUT_CSV = os.environ.get(
    "OTHERS_AUGMENT_INPUT_CSV",
    r"C:\Users\user\Downloads\140924_combined_reports_prescriptions_7600.csv",
)
DEFAULT_OUTPUT_JSONL = os.environ.get("OTHERS_AUGMENT_OUTPUT_JSONL", "augmented_data.jsonl")
DEFAULT_TAIL_OUTPUT_CSV = os.environ.get(
    "OTHERS_TAIL_OUTPUT_CSV",
    r"C:\Users\user\Documents\augmented_dataset_conversation_other_bottom.csv",
)

JSONL_MAX_WORKERS = 5
JSONL_FUTURE_TIMEOUT_SEC = 45

TAIL_MAX_WORKERS = 20
TAIL_ROW_COUNT = 3000
PERSONA_STREAM_ROW_LIMIT = 25_000

PROMPT_TEMPLATE_MEDICAL_OTHERS = """
You are a data augmentation assistant.

Given an input text that may look like a medical report, lab result, or prescription, your task is to transform it into formats that DO NOT resemble structured medical documents.

Your goal is to generate outputs that belong to the "other" class in a classification dataset.

### Instructions:
1. Convert the input into ONE of the following styles (randomly choose):
   - Casual conversation (2–3 people talking)
   - WhatsApp/chat messages
   - Story or narrative paragraph
   - General discussion or explanation
   - Blog-style writing
   - Question-answer dialogue
   - Personal note or diary entry

2. IMPORTANT:
   - Do NOT preserve structured medical formatting (no bullet lists like prescriptions, no "Name:", "Age:", etc.)
   - Do NOT keep it looking like a report
   - You MAY keep or slightly distort the meaning, but presentation must change completely
   - You MAY omit, reorder, or generalize details
   - You MAY introduce informal language, filler words, or conversational tone


3. Optional Noise Injection:
   - Add small talk, interruptions, or irrelevant details
   - Introduce ambiguity or partial information
   - Slightly alter names, numbers, or entities

4. - Do NOT preserve structured medical formatting  
  (no "Name:", "Age:", bullet lists, tables, or prescription layout)
- Break ordering of information (shuffle, drop, merge details)
- Use natural, human-like phrasing
- You MAY distort or generalize the meaning

5.  OCR Noise + Typo Injection (VERY IMPORTANT):

Introduce realistic OCR errors and human typing mistakes:

- Character substitutions:
  - o → 0, l → 1, e → c, a → @, s → 5
- Missing or extra spaces:
  - "bloodpressure" / "blood  pressure"
- Broken or merged words:
  - "medicationtaken" / "medi cation"
- Random capitalization:
  - "pAtient", "DOcTor"
- Spelling mistakes:
  - "fever" → "fevr", "tablet" → "tablct"
- Punctuation noise:
  - extra commas, missing full stops, "??", "..."
- Slight numeric distortions:
  - 500 → 50O, 25 → 2S
- Partial truncation:
  - cut words midway ("prescrip...", "medic...")

 Do NOT overdo noise — keep text readable but imperfect (like OCR output).

6. Output must:
   - Be natural and human-like
   - Clearly NOT resemble a medical document
   - Be suitable for classification as "other"

### Input:
{text}

OUTPUT FORMAT:


Return ONLY a JSON array of string:

[
"",

]


"""


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def load_persona_dataframe() -> pd.DataFrame:
    stream = load_dataset("nvidia/Nemotron-Personas-India", split="en_IN", streaming=True)
    sample = stream.take(PERSONA_STREAM_ROW_LIMIT)
    return pd.DataFrame(list(sample))


def sample_persona(persona_frame: pd.DataFrame) -> dict:
    row = persona_frame.sample(1).iloc[0]
    return {
        "name": row.get("name", ""),
        "age": row.get("age", ""),
        "gender": row.get("gender", ""),
        "city": row.get("city", ""),
    }


def call_llm(prompt: str) -> dict:
    api_key = os.environ.get("LONGCAT_API_KEY", "your_api_key_here")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.85,
    }
    response = requests.post(LONGCAT_CHAT_COMPLETIONS_URL, headers=headers, json=payload)
    return response.json()


_jsonl_file_lock = threading.Lock()


def append_jsonl_rows(output_path: str, entries: list[dict]) -> None:
    with _jsonl_file_lock:
        with open(output_path, "a", encoding="utf-8") as outfile:
            for entry in entries:
                outfile.write(json.dumps(entry) + "\n")


def load_processed_filenames(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    processed: set[str] = set()
    with open(output_path, "r", encoding="utf-8") as infile:
        for line in infile:
            try:
                record = json.loads(line)
                processed.add(record["filename"])
            except Exception:
                continue
    return processed


def process_row_jsonl(row_index: int, row: pd.Series) -> list[dict] | None:
    filename = row.get("filename", f"row_{row_index}")
    try:
        text = row["text"]
        label = row.get("label", "N/A")
        filename = row.get("filename", f"row_{row_index}")

        if pd.isna(text) or str(text).strip() == "":
            return None

        prompt_other = PROMPT_TEMPLATE_MEDICAL_OTHERS.format(text=text)
        response = call_llm(prompt_other)

        output = response["choices"][0]["message"]["content"]
        variations = json.loads(output)

        local_rows: list[dict] = []
        local_rows.append(
            {
                "filename": filename,
                "original_text": text,
                "augmented_text": text,
                "label": label,
                "status": "original",
            }
        )
        for variation in variations:
            local_rows.append(
                {
                    "filename": filename,
                    "original_text": text,
                    "augmented_text": variation,
                    "label": label,
                    "status": "augmented",
                }
            )
        return local_rows

    except Exception as exc:
        print(f"\n[ERROR] Row {row_index} ({filename}): {exc!s}")
        return None


def run_jsonl_pipeline(input_csv: str, output_jsonl: str, max_workers: int = JSONL_MAX_WORKERS) -> None:
    records_frame = pd.read_csv(input_csv)
    processed_filenames = load_processed_filenames(output_jsonl)
    print(f"Total rows: {len(records_frame)}")
    print(f"Already processed: {len(processed_filenames)}")

    pending_frame = records_frame[~records_frame["filename"].isin(processed_filenames)]
    print(f"Rows remaining: {len(pending_frame)}")

    if pending_frame.empty:
        print("Everything is already processed.")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filename = {
            executor.submit(process_row_jsonl, idx, row): row["filename"]
            for idx, row in pending_frame.iterrows()
        }

        for future in tqdm(
            as_completed(future_to_filename),
            total=len(future_to_filename),
            desc="Processing",
        ):
            source_filename = future_to_filename[future]
            try:
                result = future.result(timeout=JSONL_FUTURE_TIMEOUT_SEC)
                if result:
                    append_jsonl_rows(output_jsonl, result)
            except Exception as exc:
                print(f"\n[CRITICAL] {source_filename} timed out or failed: {exc!s}")

    print(f"\nDone. Results saved to {output_jsonl}")


def process_tail_row(
    row_index: int,
    row: pd.Series,
    persona_frame: pd.DataFrame,
) -> list[dict]:
    try:
        text = row["text"]
        label = row.get("label", None)
        filename = row.get("filename", None)

        if pd.isna(text) or str(text).strip() == "":
            return []

        persona = make_json_safe(sample_persona(persona_frame))
        prompt_other = PROMPT_TEMPLATE_MEDICAL_OTHERS.format(text=text)

        response = call_llm(prompt_other)
        output = response["choices"][0]["message"]["content"]
        variations = json.loads(output)

        local_rows: list[dict] = []
        local_rows.append(
            {
                "original_text": text,
                "augmented_text": text,
                "label": label,
                "filename": filename,
                "persona": json.dumps(persona),
                "status": "original",
            }
        )
        for variation in variations:
            local_rows.append(
                {
                    "original_text": text,
                    "augmented_text": variation,
                    "label": label,
                    "filename": filename,
                    "persona": json.dumps(persona),
                    "raw_response": output,
                    "status": "augmented",
                }
            )
        return local_rows

    except Exception as exc:
        print(f"Error processing row {row_index}: {exc!s}")
        return [
            {
                "original_text": row.get("text", ""),
                "augmented_text": "",
                "label": row.get("label", None),
                "filename": row.get("filename", None),
                "persona": "",
                "status": f"failed: {exc!s}",
            }
        ]


def run_tail_thread_pool(
    records_frame: pd.DataFrame,
    persona_frame: pd.DataFrame,
    tail_count: int,
    max_workers: int,
) -> list[dict]:
    tail_frame = records_frame.tail(tail_count)
    augmented_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_tail_row, idx, row, persona_frame)
            for idx, row in tail_frame.iterrows()
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting data (tail)"):
            batch = future.result()
            if batch:
                augmented_rows.extend(batch)
    return augmented_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Others-class LLM augmentation")
    parser.add_argument(
        "mode",
        choices=("jsonl", "tail"),
        help="jsonl: resume JSONL pipeline; tail: last N rows to CSV with personas",
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Source CSV path")
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL, help="JSONL output (jsonl mode)")
    parser.add_argument("--output-csv", default=DEFAULT_TAIL_OUTPUT_CSV, help="CSV output (tail mode)")
    parser.add_argument("--tail-rows", type=int, default=TAIL_ROW_COUNT, help="Rows from tail (tail mode)")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override worker count (defaults: jsonl=5, tail=20)",
    )
    args = parser.parse_args()

    if args.mode == "jsonl":
        max_workers = args.max_workers if args.max_workers is not None else JSONL_MAX_WORKERS
        print(f"Using max_workers={max_workers} for jsonl mode")
        run_jsonl_pipeline(args.input_csv, args.output_jsonl, max_workers=max_workers)
        return

    max_workers = args.max_workers if args.max_workers is not None else TAIL_MAX_WORKERS
    print("Loading persona reference data...")
    persona_frame = load_persona_dataframe()
    records_frame = pd.read_csv(args.input_csv)
    augmented = run_tail_thread_pool(records_frame, persona_frame, args.tail_rows, max_workers)
    pd.DataFrame(augmented).to_csv(args.output_csv, index=False)
    print(f"Wrote {len(augmented)} augmented records to {args.output_csv}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("jsonl")
    main()
