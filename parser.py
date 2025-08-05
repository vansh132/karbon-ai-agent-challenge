import pdfplumber
import pandas as pd
import numpy as np

def parse_bank_statement(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        tables = []
        for page in pdf.pages:
            tables.extend(page.extract_tables())
        
        # Find the table with the matching header row
        for table in tables:
            header = [cell.lower() for cell in table[0]]
            if set(header) == {'date', 'description', 'debit amt', 'credit amt', 'balance'}:
                break
        
        # Parse rows into dicts
        rows = []
        for row in table[1:]:
            parsed_row = {
                'Date': row[0],
                'Description': row[1],
                'Debit Amt': np.nan if row[2] == '' else float(row[2]),
                'Credit Amt': np.nan if row[3] == '' else float(row[3]),
                'Balance': float(row[4])
            }
            rows.append(parsed_row)
        
        return pd.DataFrame(rows)

if __name__ == '__main__':
    pdf_path = 'bank_statement.pdf'  # replace with your pdf file path
    df = parse_bank_statement(pdf_path)
    print(df)
