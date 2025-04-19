# 📘 Bank Statement Text Extraction Script

## 🩾 Overview

This 🔰 Python script processes a 🏦 bank statement in 📝 PDF format to extract structured 🧹 transaction data. It combines 📜 text extraction, ✨ text cleaning, and pattern matching using 🔎 regular expressions (regex) and 🤖 Natural Language Processing (NLP) to identify and save transaction details into a 📊 CSV file.

---

## 🌟 Features

- 🐂 Extracts text from 📜 PDF files using `pdfplumber`.
- 🩹 Cleans and formats the extracted text to address irregularities.
- 🛠️ Uses a pre-trained 🤗 Hugging Face NER (Named Entity Recognition) model to identify entities.
- 🧾 Matches transactions using regex to extract:
  - 📅 Date
  - 🗋 Description
  - 💵 Amount
- Saves the extracted transactions into a 💾 CSV file for further analysis.

---

## 📦 Requirements

### 🛠️ Libraries

Install the following 🔰 Python libraries before running the script:

```bash
pip install pdfplumber transformers pandas
```
### 🧠 Model
The script uses a pre-trained 🤗 Hugging Face model: dbmdz/bert-large-cased-finetuned-conll03-english.

---

## 🔖 How to Use

1. Place Your 📜 PDF File:
Save your 🏦 bank statement as statement.pdf in the script's 🗂 directory.
2. Run the Script:
🏃 Execute the script in a 🔰 Python environment (e.g., terminal or Jupyter Notebook).
3. Review Output:
✅ Check the console for debugging outputs:
- "🩹 Cleaned Text": Displays a snippet of cleaned text.
- "🤖 NER Entities": Shows the first 20 entities identified by the NER model.
- "🔎 Regex Matches": Lists matched transactions.
- 🗂 Open extracted_transactions.csv for the saved transactions.

---
## 🖍️ Notes
This script is designed for structured 🏦 bank statements with consistent formatting.
For highly unstructured PDFs, additional 🛠️ preprocessing may be required.

## 🔒 License

This project is licensed under a proprietary license. See the [LICENSE.txt](./LICENSE.txt) file for details.
