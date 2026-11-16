from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# --------------------------------------------------
# Folders
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "output" / "lmmg"
OUTPUT_DIR = BASE_DIR / "output_with_token_counts"/ "lmmg"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache tokenizers so each model is loaded only once.
tokenizer_cache = {}


def get_tokenizer(model_name, revision):
    """Load and cache the tokenizer, without loading the language model."""

    revision_key = None if pd.isna(revision) else str(revision)
    key = (model_name, revision_key)

    if key not in tokenizer_cache:
        print(f"\nLoading tokenizer: {model_name}, revision={revision_key}")

        kwargs = {}
        if revision_key is not None:
            kwargs["revision"] = revision_key

        tokenizer_cache[key] = AutoTokenizer.from_pretrained(
            model_name,
            **kwargs
        )

    return tokenizer_cache[key]


def add_final_period(sentence):
    """Reproduce the sentence construction in the original lmmg.py."""

    if pd.isna(sentence):
        raise ValueError("A sentence is missing from the CSV.")

    return str(sentence) + "."


def count_tokens(sentence, tokenizer):
    """Count tokens without adding special tokens."""

    tokens = tokenizer(
        sentence,
        add_special_tokens=False
    )

    return len(tokens["input_ids"])


# --------------------------------------------------
# Process existing CSV files
# --------------------------------------------------

csv_files = list(INPUT_DIR.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

print(f"Found {len(csv_files)} CSV files.")

for csv_file in csv_files:

    print(f"\nProcessing: {csv_file.name}")

    df = pd.read_csv(csv_file)

    required_columns = [
        "sentence1",
        "sentence2",
        "model",
        "revision"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        print(f"Skipping {csv_file.name}: missing columns {missing}")
        continue

    token_counts_1 = []
    token_counts_2 = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        model_name = row["model"]
        revision = row["revision"]

        tokenizer = get_tokenizer(model_name, revision)

        # Match the original lmmg.py, which appended a full stop.
        sentence1 = add_final_period(row["sentence1"])
        sentence2 = add_final_period(row["sentence2"])

        token_counts_1.append(
            count_tokens(sentence1, tokenizer)
        )

        token_counts_2.append(
            count_tokens(sentence2, tokenizer)
        )

    # Add the new columns.
    df["token_count_sentence1"] = token_counts_1
    df["token_count_sentence2"] = token_counts_2

    df["token_count_difference"] = (
        df["token_count_sentence1"] -
        df["token_count_sentence2"]
    )

    df["absolute_token_count_difference"] = (
        df["token_count_difference"].abs()
    )

    # Save in a separate folder, preserving the original subfolder structure.
    relative_path = csv_file.relative_to(INPUT_DIR)
    output_file = OUTPUT_DIR / relative_path

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")

print("\nFinished adding token counts.")