#!/usr/bin/env python

import os
import sys
import click
import importlib.util
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq

load_dotenv()  # Ensure GROQ_API_KEY is set

MODEL_NAME   = "llama-3.3-70b-versatile"
EXPECTED_CSV = "data/icici/result.csv"
PARSER_FILE  = "parser.py"

def generate_parser_code(sample_rows: list[dict]) -> str:
    """
    Ask Groq to generate a parser that:
      - uses pdfplumber to open the PDF and extract all tables,
      - detects the header row matching Date, Description, Debit Amt, Credit Amt, Balance,
      - parses each row converting blank Debits/Credits to np.nan and others to float,
      - returns a pandas.DataFrame with columns ['Date','Description','Debit','Credit','Balance'].
    We provide sample_rows as guidance but do NOT include function signature in the prompt.
    """
    user_prompt = f"""
We have a bank-statement PDF laid out in five columns: Date, Description, Debit Amt, Credit Amt, Balance.

Here are the first three rows of the *expected* output DataFrame (as Python dicts):

{sample_rows}

Please write complete Python code that:
1. Imports pdfplumber, pandas as pd, and numpy as np.
2. Defines a function `parse_bank_statement(pdf_path)` that:
   - Opens the given PDF with pdfplumber.
   - Extracts all tables from every page.
   - Finds the header row by matching column names case-insensitively.
   - Parses each subsequent row into dicts, converting blank numeric cells to `np.nan` and non-blank to `float()`.
   - Returns a pandas.DataFrame with exactly those five columns.
3. At the end, under `if __name__ == '__main__':`, calls `parse_bank_statement` on a `pdf_path` variable and prints the resulting DataFrame.

Do not hard-code the sample_rows; write generic code that will reproduce them and all other rows. Output only the Python code (no additional explanation).
"""
    llm = ChatGroq(model=MODEL_NAME, api_key=os.getenv("GROQ_API_KEY"), temperature=0.0)
    response = llm.invoke([
        {"role": "system", "content": "You are a Python coding assistant."},
        {"role": "user",   "content": user_prompt}
    ])
    return response.content

def clean_markdown(code: str) -> str:
    """
    Strip backtick fences and any leading commentary, returning clean Python.
    """
    lines = [ln for ln in code.splitlines() if not ln.strip().startswith("```")]
    # drop everything before the first import/def
    for i, ln in enumerate(lines):
        if ln.startswith(("import ", "from ", "def ")):
            lines = lines[i:]
            break
    return "\n".join(lines).rstrip() + "\n"

def write_parser(code: str):
    cleaned = clean_markdown(code)
    with open(PARSER_FILE, "w") as f:
        f.write(cleaned)
    click.echo(f"✔️  Wrote parser to {PARSER_FILE}")

def load_and_run(pdf_path: str) -> pd.DataFrame:
    """
    Dynamically import parser.py and run parse_bank_statement.
    """
    spec = importlib.util.spec_from_file_location("parser", os.path.abspath(PARSER_FILE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_bank_statement(pdf_path)

@click.command()
@click.argument("pdf_path", type=click.Path(exists=True))
def main(pdf_path):
    """
    1) Load expected CSV and sample first 3 rows.
    2) Generate parser.py via Groq (one-shot), without embedding a function signature in the prompt.
    3) Load and run the parser on the full PDF.
    4) Compare the entire output to the expected CSV and report success or diff.
    """
    # 1. Load expected CSV
    try:
        expected = pd.read_csv(EXPECTED_CSV)
    except Exception as e:
        click.echo(f"❌ Failed to read expected CSV: {e}")
        sys.exit(1)

    sample_rows = expected.head(3).to_dict(orient="records")

    # 2. Generate parser
    click.echo("⏳ Generating parser via Groq…")
    raw_code = generate_parser_code(sample_rows)
    write_parser(raw_code)

    # 3. Run parser and compare
    click.echo("📦 Running parser and comparing full output…")
    try:
        df = load_and_run(pdf_path)
    except Exception as e:
        click.echo(f"❌ Parser error: {e}")
        sys.exit(1)

    # 4. Compare DataFrames
    if df.equals(expected):
        click.echo("✅ Success! Output matches expected CSV exactly.")
        df.to_csv("result_extracted.csv", index=False)
        click.echo("   Saved as result_extracted.csv")
    else:
        click.echo("❌ Mismatch: parser output did not match expected CSV.")
        diff = pd.concat([df, expected]).drop_duplicates(keep=False)
        click.echo("\nDifferences:\n" + diff.to_string(index=False))
        sys.exit(1)

if __name__ == "__main__":
    main()
