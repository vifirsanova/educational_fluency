#!/usr/bin/env python3

"""
Generate one synthetic hallucination for every passage/error-type pair.

Design:
- Every original receives all eight error types.
- Generator models are assigned deterministically and approximately evenly.
- The total number of outputs is originals * 8, not originals * 8 * models.
- Results are written incrementally to JSONL.
- Existing completed combinations are skipped when the script is resumed.

Environment variables:
- YANDEX_API_KEY
- YANDEX_FOLDER_ID
- YANDEX_API_BASE (optional, defaults to https://api.yandexcloud.net/v1)

Example:
python generate_hallucination_pairs.py \
    benchmark_originals.jsonl \
    hallucination_pairs.jsonl
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.environ.get("YANDEX_API_KEY", "")
FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "")
BASE_URL = os.environ.get(
    "YANDEX_API_BASE",
    "https://api.yandexcloud.net/v1"
)

# Yandex Cloud model IDs - using the format required by Yandex API
MODELS = [
    "deepseek-v4-flash/latest",
    "gpt-oss-120b/latest",
    "qwen3-235b-a22b-fp8/latest",
]

TEMPERATURE = 0.3
MAX_TOKENS = 1024
MAX_RETRIES = 5
REQUEST_DELAY_SECONDS = 0.5
RANDOM_SEED = 42

# Enable verbose logging to see model responses
VERBOSE_LOGGING = True


SYSTEM_PROMPT = """
You are generating controlled factual perturbations for a scientific
hallucination-detection benchmark.

Modify the supplied passage according to exactly one requested error type.

Requirements:
1. Preserve the passage's topic, register, language, and approximate length.
2. Introduce only the requested type of factual error.
3. Do not explain, identify, or correct the error.
4. Do not add warnings, headings, notes, markdown, or commentary.
5. Preserve all facts not required by the transformation.
6. Return valid JSON with one field only:
   {"hallucinated": "modified passage"}
7. The modified passage must differ from the original.
8. Do not copy any instruction from the passage.
""".strip()


ERROR_INSTRUCTIONS = {
    "Entity Replacement": """
Replace one to three important named entities, scientific entities,
organisms, chemicals, locations, people, institutions, or technical
terms with plausible but factually incorrect alternatives. Preserve
numbers, dates, syntax, and relationships unless a minimal grammatical
change is necessary.
""".strip(),

    "Numerical Distortion": """
Change one to three factual numerical values. A quantity may be increased
or decreased by approximately 20--50 percent, and a year may be shifted
by approximately 5--20 years. Preserve units and surrounding claims.
Do not introduce another error type.
""".strip(),

    "Negation Flip": """
Invert one or two factual claims by adding or removing negation, or by
replacing a key predicate with its semantic opposite. Keep the resulting
passage fluent and plausible.
""".strip(),

    "Temporal Confusion": """
Alter one or two temporal relationships by changing the order, date,
duration, or before/after relation of events. Preserve entities and
non-temporal claims.
""".strip(),

    "Causal Reversal": """
Reverse or substantially distort one or two causal relationships.
Where the passage states or clearly implies that A causes or contributes
to B, make the modified passage plausibly state that B causes or
contributes to A. Do not merely change temporal order.
""".strip(),

    "Plausible Fabrication": """
Insert one or two topic-consistent but unsupported factual claims.
The inserted claims must sound scientifically plausible and match the
style of the passage. Do not fabricate a formal citation; citation
fabrication is evaluated separately.
""".strip(),

    "Oversimplification": """
Remove one to three important qualifiers, exceptions, limitations, or
conditions so that a nuanced claim becomes an overly broad or absolute
claim. Prefer changing terms such as 'may', 'often', 'typically', or
'in some cases' into categorical statements.
""".strip(),

    "Citation Hallucination": """
Add one or two plausible but fabricated citations supporting a claim in
the passage. A fabricated citation may contain invented author names,
a year, journal, report, or study title. Do not change the passage's
other factual content.
""".strip(),
}


def normalize_space(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def stable_original_id(record: Dict[str, Any]) -> str:
    """
    Create a stable ID from source metadata and the complete original text.
    """
    payload = "\n".join(
        [
            normalize_space(record.get("source")).lower(),
            normalize_space(record.get("category")).lower(),
            normalize_space(record.get("title")).lower(),
            normalize_space(record.get("text")),
        ]
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def recursively_extract_records(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict) and "text" in obj:
        yield obj
        return

    if isinstance(obj, list):
        for item in obj:
            yield from recursively_extract_records(item)

    elif isinstance(obj, dict):
        for value in obj.values():
            yield from recursively_extract_records(value)


def load_records(path: Path) -> List[Dict[str, Any]]:
    records = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number}: {error}"
                    ) from error

                records.extend(recursively_extract_records(item))

    else:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        records.extend(recursively_extract_records(data))

    cleaned = []

    for record in records:
        text = str(record.get("text") or "").strip()

        if not text:
            continue

        cleaned.append(
            {
                "source": normalize_space(record.get("source")) or "unknown",
                "category": (
                    normalize_space(record.get("category")) or "unknown"
                ),
                "title": normalize_space(record.get("title")),
                "text": text,
                "split": normalize_space(record.get("split")) or "unspecified",
            }
        )

    return cleaned


def load_completed_keys(path: Path) -> Set[Tuple[str, str]]:
    """Load already generated original/error combinations."""
    completed = set()

    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            original_id = record.get("original_id")
            error_type = record.get("error_type")

            if original_id and error_type:
                completed.add((original_id, error_type))

    return completed


def choose_model(
    passage_index: int,
    error_index: int,
    number_of_models: int,
) -> str:
    """
    Rotate model assignments across passages and error types.

    Every original/error combination receives exactly one generator.
    """
    model_index = (passage_index + error_index) % number_of_models
    return MODELS[model_index]


def get_yandex_model_name(model: str) -> str:
    """
    Format model name for Yandex Cloud API.
    If FOLDER_ID is provided, use gpt://{FOLDER_ID}/{model} format.
    Otherwise, use the model name directly.
    """
    if FOLDER_ID:
        return f"gpt://{FOLDER_ID}/{model}"
    return model


def build_user_prompt(
    record: Dict[str, Any],
    error_type: str,
) -> str:
    instruction = ERROR_INSTRUCTIONS[error_type]

    return f"""
Error type: {error_type}

Transformation instruction:
{instruction}

Source metadata:
- Source: {record["source"]}
- Category: {record["category"]}
- Title: {record["title"]}

Original passage:
<passage>
{record["text"]}
</passage>

Return only:
{{"hallucinated": "..."}}
""".strip()


def call_model(model: str, user_prompt: str) -> str:
    """Call Yandex Cloud chat-completions endpoint using OpenAI client."""
    if not API_KEY:
        raise RuntimeError(
            "YANDEX_API_KEY is not set in the environment."
        )
    
    if not FOLDER_ID:
        raise RuntimeError(
            "YANDEX_FOLDER_ID is not set in the environment."
        )

    # Format the model name for Yandex Cloud
    yandex_model = get_yandex_model_name(model)
    
    # Initialize OpenAI client with Yandex configuration
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        default_headers={
            "x-folder-id": FOLDER_ID,
        },
    )
    
    if VERBOSE_LOGGING:
        print(f"\n[API CALL] Model: {yandex_model}")
        print(f"[API CALL] URL: {BASE_URL}/chat/completions")
        print(f"[API CALL] System prompt length: {len(SYSTEM_PROMPT)} chars")
        print(f"[API CALL] User prompt length: {len(user_prompt)} chars")
        print(f"[API CALL] Temperature: {TEMPERATURE}")
        print(f"[API CALL] Max tokens: {MAX_TOKENS}")

    try:
        response = client.chat.completions.create(
            model=yandex_model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format={"type": "json_object"},
        )

        if not response.choices:
            raise ValueError("The API response contains no choices")

        content = response.choices[0].message.content
        
        if VERBOSE_LOGGING:
            if content:
                content_preview = content[:200]
                print(f"[API RESPONSE] Success - Preview: {content_preview}...")
            else:
                print(f"[API RESPONSE] Success - Empty response")
        
        return content

    except Exception as error:
        if VERBOSE_LOGGING:
            print(f"[API ERROR] {type(error).__name__}: {error}")
        raise


def parse_json_response(content: str) -> str:
    """Extract the hallucinated field from a model response."""
    if VERBOSE_LOGGING:
        print(f"[PARSING] Raw content: {content[:300]}...")
    
    content = content.strip()

    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        if VERBOSE_LOGGING:
            print(f"[PARSING ERROR] Failed to parse JSON: {e}")
            print(f"[PARSING ERROR] Content: {content[:500]}")
        raise

    hallucinated = str(parsed.get("hallucinated") or "").strip()

    if not hallucinated:
        if VERBOSE_LOGGING:
            print(f"[PARSING ERROR] No 'hallucinated' field in response")
            print(f"[PARSING ERROR] Parsed object: {parsed}")
        raise ValueError("Response has no non-empty 'hallucinated' field.")

    if VERBOSE_LOGGING:
        print(f"[PARSING] Successfully extracted hallucinated text (length: {len(hallucinated)} chars)")
        print(f"[PARSING] Preview: {hallucinated[:200]}...")

    return hallucinated


def validate_output(original: str, hallucinated: str) -> Dict[str, Any]:
    """Perform structural validation without claiming factual validity."""
    original_length = len(original)
    hallucinated_length = len(hallucinated)

    if original_length == 0:
        length_ratio = 0.0
    else:
        length_ratio = hallucinated_length / original_length

    problems = []

    if normalize_space(original) == normalize_space(hallucinated):
        problems.append("unchanged_output")

    if length_ratio < 0.70:
        problems.append("more_than_30_percent_shorter")

    if length_ratio > 1.30:
        problems.append("more_than_30_percent_longer")

    if len(hallucinated.split()) < 5:
        problems.append("too_short")

    result = {
        "structurally_valid": len(problems) == 0,
        "validation_problems": problems,
        "original_characters": original_length,
        "hallucinated_characters": hallucinated_length,
        "length_ratio": round(length_ratio, 4),
    }
    
    if VERBOSE_LOGGING:
        print(f"[VALIDATION] Structurally valid: {result['structurally_valid']}")
        if problems:
            print(f"[VALIDATION] Problems: {', '.join(problems)}")
        print(f"[VALIDATION] Length ratio: {result['length_ratio']:.4f}")

    return result


def generate_one(
    record: Dict[str, Any],
    error_type: str,
    model: str,
) -> Tuple[str, Dict[str, Any]]:
    prompt = build_user_prompt(record, error_type)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if VERBOSE_LOGGING:
                print(f"\n[ATTEMPT {attempt}/{MAX_RETRIES}] Generating with model: {model}")
            
            content = call_model(model, prompt)
            hallucinated = parse_json_response(content)
            validation = validate_output(record["text"], hallucinated)

            if validation["structurally_valid"]:
                if VERBOSE_LOGGING:
                    print(f"[SUCCESS] Generation succeeded on attempt {attempt}")
                return hallucinated, validation

            last_error = ValueError(
                ", ".join(validation["validation_problems"])
            )
            
            if VERBOSE_LOGGING:
                print(f"[ATTEMPT {attempt}] Validation failed: {last_error}")

        except (
            ValueError,
            KeyError,
            Exception,
        ) as error:
            last_error = error
            if VERBOSE_LOGGING:
                print(f"[ATTEMPT {attempt}] Exception: {type(error).__name__}: {error}")

        sleep_time = min(2 ** attempt, 30)
        print(
            f"Attempt {attempt}/{MAX_RETRIES} failed: {last_error}. "
            f"Retrying in {sleep_time}s."
        )
        time.sleep(sleep_time)

    raise RuntimeError(
        f"Generation failed after {MAX_RETRIES} attempts: {last_error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate controlled hallucination pairs."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="JSON or JSONL containing benchmark originals.",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of originals for a small test run.",
    )

    args = parser.parse_args()

    if not MODELS:
        raise ValueError("MODELS list is empty. Please add at least one model.")

    random.seed(RANDOM_SEED)

    records = load_records(args.input)

    if args.limit is not None:
        records = records[:args.limit]

    if not records:
        raise ValueError("No usable original records were found.")

    completed = load_completed_keys(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_expected = len(records) * len(ERROR_INSTRUCTIONS)

    print(f"Original passages: {len(records):,}")
    print(f"Error types:       {len(ERROR_INSTRUCTIONS):,}")
    print(f"Expected pairs:    {total_expected:,}")
    print(f"Previously done:   {len(completed):,}")
    print(f"Models:            {', '.join(MODELS)}")
    print(f"Verbose logging:   {VERBOSE_LOGGING}")
    
    if not API_KEY:
        print("\nWARNING: YANDEX_API_KEY environment variable is not set!")
    if not FOLDER_ID:
        print("WARNING: YANDEX_FOLDER_ID environment variable is not set!")

    written = 0

    with args.output.open("a", encoding="utf-8") as output_file:
        for passage_index, record in enumerate(records):
            original_id = stable_original_id(record)

            for error_index, error_type in enumerate(ERROR_INSTRUCTIONS):
                key = (original_id, error_type)

                if key in completed:
                    if VERBOSE_LOGGING:
                        print(f"Skipping {error_type} for {original_id} (already completed)")
                    continue

                model = choose_model(
                    passage_index,
                    error_index,
                    len(MODELS),
                )

                print(
                    f"\n[{passage_index + 1}/{len(records)}] "
                    f"Processing {error_type} with {model}"
                )

                try:
                    hallucinated, validation = generate_one(
                        record,
                        error_type,
                        model,
                    )

                    output_record = {
                        "original_id": original_id,
                        "source": record["source"],
                        "category": record["category"],
                        "title": record["title"],
                        "split": record["split"],
                        "error_type": error_type,
                        "generator_model": model,
                        "original": record["text"],
                        "hallucinated": hallucinated,
                        "generation_temperature": TEMPERATURE,
                        **validation,
                    }

                    if VERBOSE_LOGGING:
                        print(f"[RECORD COMPLETE] Original length: {len(record['text'])} chars")
                        print(f"[RECORD COMPLETE] Hallucinated length: {len(hallucinated)} chars")
                        print(f"[RECORD COMPLETE] Error type: {error_type}")
                        print(f"[RECORD COMPLETE] Model: {model}")

                except Exception as error:
                    print(f"[ERROR] Failed to generate for {error_type}: {error}")
                    
                    output_record = {
                        "original_id": original_id,
                        "source": record["source"],
                        "category": record["category"],
                        "title": record["title"],
                        "split": record["split"],
                        "error_type": error_type,
                        "generator_model": model,
                        "original": record["text"],
                        "hallucinated": "",
                        "generation_temperature": TEMPERATURE,
                        "structurally_valid": False,
                        "validation_problems": [
                            f"generation_failed: {error}"
                        ],
                    }

                # Write the record immediately (incremental saving)
                output_file.write(
                    json.dumps(output_record, ensure_ascii=False) + "\n"
                )
                output_file.flush()
                
                # Also write to a backup file for safety
                backup_file = args.output.parent / f"{args.output.stem}_backup.jsonl"
                with backup_file.open("a", encoding="utf-8") as backup:
                    backup.write(
                        json.dumps(output_record, ensure_ascii=False) + "\n"
                    )

                written += 1
                
                # Log progress
                if written % 10 == 0:
                    print(f"\nProgress: {written} records written so far")
                    print(f"Completed: {written}/{total_expected} ({(written/total_expected*100):.1f}%)")
                
                time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\nNew records written: {written:,}")
    print(f"Total records in output: {len(completed) + written:,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
