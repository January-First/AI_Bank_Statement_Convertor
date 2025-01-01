import pdfplumber
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import pandas as pd

# Path to the PDF file
file_path = "./statement.pdf"

# Extract text from PDF
with pdfplumber.open(file_path) as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
        
def clean_text(raw_text):
    cleaned_text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", raw_text)
    cleaned_text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", cleaned_text)
    cleaned_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned_text)
    cleaned_text = re.sub(r"\n+", "\n", cleaned_text)
    return cleaned_text

text = clean_text(text)
print("Cleaned Text:")
print(text[:500])  


tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0)  # Use device 0 for GPU or -1 for CPU
entities = ner_pipeline(text)


print("\nNER Entities:")
for entity in entities[:20]: 
    print(entity)

# Define regex pattern to match transactions
# Updated regex pattern to match transactions
transaction_pattern = re.compile(
    r"(\d{1,2} \w{3})\s+(.+?)\s+([\d,]+\.\d{2})", re.MULTILINE
)

transactions = []
for match in transaction_pattern.finditer(text):
    date, description, amount = match.groups()
    transactions.append({
        "Date": date.strip(),
        "Description": description.strip(),
        "Amount (SGD)": float(amount.replace(",", ""))  
    })
    

matches = transaction_pattern.findall(text)
print("\nRegex Matches:")
if matches:
    for match in matches:
        print(match)
else:
    print("No matches found. Check the regex or cleaned text.")


if transactions:
    df = pd.DataFrame(transactions)
    df.to_csv("extracted_transactions.csv", index=False)
    print("\nTransactions saved to 'extracted_transactions.csv'")
