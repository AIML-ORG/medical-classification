"""
LLM augmentation for prescription vs lab-report rows (label 1 -> lab prompt, else prescription).

Requires: pip install datasets pandas requests tqdm
Set LONGCAT_API_KEY in the environment for the Bearer token.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

LONGCAT_CHAT_COMPLETIONS_URL = "https://api.longcat.chat/openai/v1/chat/completions"
LLM_MODEL_NAME = "LongCat-Flash-Lite"

# Default paths; override with env or edit before running.
DEFAULT_INPUT_CSV = os.environ.get(
    "MEDICAL_AUGMENT_INPUT_CSV",
    r"C:\Users\user\Downloads\140924_combined_reports_prescriptions_7600.csv",
)
DEFAULT_OUTPUT_CSV = os.environ.get(
    "MEDICAL_AUGMENT_OUTPUT_CSV",
    "augmented_prescription_lab_reports.csv",
)

PERSONA_STREAM_ROW_LIMIT = 25_000
MAX_WORKERS = 20

PROMPT_TEMPLATE_PRESCRIPTION = """
You are a data augmentation assistant for information extraction models on medical prescriptions.
Your task is to generate synthetic variations of a given OCR medical record.

INPUT:
You will receive a medical record in text strings.

GOAL:
Generate 3 new variations of the record.

AUGMENTATION RULES:

1. Patient Information

* Patient demographic information will be provided from an external persona dataset (NVIDIA Nemotron Personas India).
* Use the provided persona values to fill or replace the patient-related fields such as:
  patient_name, patient_age, patient_sex/gender, and optionally patient_location or city.
* Do NOT invent random names or demographics. Only use the persona information when available.

2. Medicine Variations

* Modify medicine related fields to create realistic prescription diversity.
* You may change medicine names, dosage amounts, or instructions while keeping them medically plausible.
  Examples of variations include:
* changing dosage values (10 mg → 25 mg → 50 mg)
* changing instruction phrases (Before meals, After meals, Once daily, Twice daily)
* replacing a medicine with another common medicine.

3. Structural Changes

* Shuffle the order of fields.
* Remove 1–3 fields randomly (for example patient_age, clinic_address, signature).
* Sometimes add a new optional field like patient_sex or patient_location.
for example, if the original record does not have patient_sex, you can add it in one of the variations using the persona data.
consider this .'BioSphere  Advanced  Pathology  LABORATORY  TEST  REPORT  PatientID  368599  Collected  2025-05-27  Total  WBC:308.07fL  Potassium-187.65  fL  Glucose  Random(196.63cells/cumm)  Serum  Creatinine(322.21cells/cumm)  Hemoglobin-370.55  %  Blood  Urea:397.53%  Lab  Internal  QC  OK '
here you can remove the laboratory name, or the patient id, or the collected date, or the total wbc, or the potassium, or the glucose, or the serum creatinine, or the hemoglobin, or the blood urea, or the lab internal qc. you can also change the values of these fields. you can also add new fields like patient sex or patient location using the persona data.
and you can change values and shuffle the order of these fields.

4. ADDITION OF random senetences AND OCR ERRORS
   Introduce random words or phrases that could realistically appear in the context of a medical record. Additionally, simulate OCR errors by altering characters in the text.as generally the ocr is a bit long and not that short as we have considered 

5. Formatting Variations
   Change formatting such as:

Age
35
35 yrs
35 years

Date
2024-12-16
16/12/2024
Dec 16 2024

Doctor Name
Dr. A. Smith
Dr A Smith
Dr.A.Smith

Gender
Male
M
Sex: Male

6. Important Constraints

* Do NOT explain anything.
* Only output the generated records.
* Ensure the 3 generated records are structurally different from each other.

7. CHANGE the name of labs and doctors to create more diversity.
lets say if the original record has a lab name "BioSphere Advanced Pathology LABORATORY TEST REPORT" you can change it to "HealthFirst Diagnostic Center" or "MediLab Pathology Services" in the variations. Similarly, for doctor names, if the original record mentions "Dr. A. Smith", you can create variations like "Dr. R. Kumar" or "Dr. S. Patel". This will help in creating more diverse and realistic synthetic records.

8. INTRODUCTION OF TYPOS AND JUSTIFIED OCR ERRORS
introduce typos in the variations to mimic real-world OCR errors. For example, "WBC" could be written as "W8C" or "Glucose" could be "Glocose". This will help in making the augmented data more robust for training OCR models.

OUTPUT FORMAT:

Return ONLY a JSON array of 3 strings.

Example:
[
"",
"",
""
]

INPUT RECORD:
{text}

PERSONA DATA:
{persona}
"""


PROMPT_TEMPLATE_NON_PRESCRIPTION = """
You are a data augmentation assistant for OCR-based information extraction models.

Your task is to generate synthetic variations of laboratory reports / diagnostic records (non-prescription data).

INPUT:
You will receive a medical record in text strings.

GOAL:
Generate 3 new variations of the record.

AUGMENTATION RULES:

1. Report & Entity Variations

* Change laboratory / hospital names to realistic alternatives
  (e.g., "AIG Hospitals Laboratory" → "HealthFirst Diagnostics", "MediCare Labs", "PrimePath Labs")
* Modify or replace **report titles**
  (e.g., "LABORATORY TEST REPORT" → "Diagnostic Report", "Investigation Summary")


 2. Test Parameter Variations

* Modify test values while keeping formats realistic:

  * Change numeric values (e.g., 134.66 → 120.45 → 298.12)
  * Slightly alter units (IU/L, g/dL, mmol/L, %, fL, cells/cumm, sec)
* You may:

  * Remove some test parameters
  * Add new common lab parameters such as:

    * Platelet Count
    * RBC
    * Sodium / Potassium
    * Urea
    * Creatinine
    * Hemoglobin
    * CRP
* Keep values noisy/unstructured like OCR outputs

 3. Structural Changes

* Shuffle field order randomly
* Remove 1–4 fields (e.g., PatientID, Collected date, specific tests)
* Optionally add new fields:

  * Patient Name
  * Age
  * Gender / Sex
  * Location
* Add or remove sections like:

  * "Medication Advice"
  * "Lab Internal QC OK"
  * "End Of Report"

  
4. ADDITION OF random senetences AND OCR ERRORS
   Introduce random words or phrases that could realistically appear in the context of a medical record. Additionally, simulate OCR errors by altering characters in the text.as generally the ocr is a bit long and not that short as we have considered 

5. Formatting Variations

Introduce formatting diversity:

Dates:

* 2025-02-11
* 11/02/2025
* Feb 11 2025

PatientID:

* PatientID 257136
* ID: 257136
* PID-257136

Test entries:

* RDW-134.66 IU/L
* RDW : 134.66 IU/L
* RDW(134.66IU/L)

Spacing / casing:

* RANDOM CAPS
* inconsistent spacing
* merged words

6. OCR Noise & Typos (IMPORTANT)

Introduce realistic OCR errors:

* Character swaps:
 * O ↔ 0
  * I ↔ 1
  * B ↔ 8
* Example:
 * WBC → W8C
  * Sodium → Sodiurn
  * Glucose → Glocose
* Broken tokens:

  * "Creatinine" → "Creatin ine"
  * "Hemoglobin" → "Hem0globin"

 7. Optional Medication Section

* Sometimes include a Medication Advice section
* Modify medicine names and dosage patterns:

  * Tab Paracetamol 1-0-1
  * Cap Azithromycin 0-1-0
* Or remove this section entirely

 8. Important Constraints

* Do NOT explain anything
* Output ONLY the generated records
* Ensure all 3 outputs are structurally different
* Keep text in raw OCR-like format (no JSON structure inside strings).

 OUTPUT FORMAT:

Return ONLY a JSON array of 3 strings:

[
"",
"",
""
]

---

INPUT RECORD:
{text}

PERSONA DATA:
{persona}

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


def process_prescription_or_lab_row(
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
        persona_json = json.dumps(persona, indent=2)

        prompt_prescription = PROMPT_TEMPLATE_PRESCRIPTION.format(text=text, persona=persona_json)
        prompt_lab_report = PROMPT_TEMPLATE_NON_PRESCRIPTION.format(text=text, persona=persona_json)

        if label == 1:
            response = call_llm(prompt_lab_report)
        else:
            response = call_llm(prompt_prescription)

        output = response["choices"][0]["message"]["content"]
        variations = json.loads(output)

        result_rows: list[dict] = []
        result_rows.append(
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
            result_rows.append(
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
        return result_rows

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


def run_thread_pool_augmentation(
    records_frame: pd.DataFrame,
    persona_frame: pd.DataFrame,
    max_workers: int = MAX_WORKERS,
) -> list[dict]:
    augmented_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_prescription_or_lab_row, idx, row, persona_frame)
            for idx, row in records_frame.iterrows()
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting data"):
            batch = future.result()
            if batch:
                augmented_rows.extend(batch)
    return augmented_rows


def main() -> None:
    print("Loading persona reference data...")
    persona_frame = load_persona_dataframe()
    print(f"Persona rows: {len(persona_frame)}")

    records_frame = pd.read_csv(DEFAULT_INPUT_CSV)
    print(f"Input columns: {list(records_frame.columns)}")

    augmented_rows = run_thread_pool_augmentation(records_frame, persona_frame)
    output_frame = pd.DataFrame(augmented_rows)
    output_frame.to_csv(DEFAULT_OUTPUT_CSV, index=False)
    print(f"Wrote {len(output_frame)} rows to {DEFAULT_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
