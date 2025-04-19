from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import os
import flask
from werkzeug.utils import secure_filename
import uuid
import pandas as pd
import tempfile
from datetime import datetime, timedelta
import re
import pdfplumber
import json
import logging
import csv
import json
import pickle
from hashlib import md5

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls', 'pdf', 'txt'}

# Ensure temp directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.after_request
def add_header(response):
    """Add diagnostic headers to help debug rendering issues."""
    response.headers['X-Debug'] = 'True'
    return response

def create_temp_file_response(temp_path, download_name):
    """
    Create a file download response and schedule the temp file for cleanup
    """
    # Note the file path for cleanup after the request
    if not hasattr(flask.g, 'temp_files_to_remove'):
        flask.g.temp_files_to_remove = []
    flask.g.temp_files_to_remove.append(temp_path)
    
    # Return the file
    return send_file(temp_path, 
                    as_attachment=True,
                    download_name=download_name)


# Add this after_request handler at the app level, not inside route functions
@app.after_request
def cleanup_temp_files(response):
    """Remove temporary files after the request is done"""
    if hasattr(flask.g, 'temp_files_to_remove'):
        for temp_path in flask.g.temp_files_to_remove:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.error(f"Error removing temp file {temp_path}: {e}")
    return response

@app.route('/session-debug')
def session_debug():
    """Display session data for debugging."""
    session_data = dict(session)
    return render_template('debug_session.html', 
                          session_data=session_data,
                          template_folder=app.template_folder)
    
def store_result_data(data, filename):
    """Store result data in a file instead of session to avoid cookie size limitations"""
    # Create a unique ID based on the filename and timestamp
    id_string = f"{filename}_{datetime.now().timestamp()}"
    result_id = md5(id_string.encode()).hexdigest()
    
    # Create a file path for the results
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"results_{result_id}.json")
    
    # Store the data in the file
    with open(result_path, 'w') as f:
        json.dump(data, f)
    
    logger.info(f"Stored result data in {result_path}")
    return result_id

def get_result_data(result_id):
    """Retrieve result data from file"""
    if not result_id:
        return None
    
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"results_{result_id}.json")
    if not os.path.exists(result_path):
        logger.warning(f"Result file not found: {result_path}")
        return None
    
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading result data: {str(e)}")
        return None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clean_old_files():
    """Delete files older than 1 hour"""
    now = datetime.now()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(filepath):
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if now - file_time > timedelta(hours=1):
                os.remove(filepath)

def clean_text(raw_text):
    """Clean up text from PDF extraction"""
    if not raw_text:
        return ""
    cleaned_text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", raw_text)
    cleaned_text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", cleaned_text)
    cleaned_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned_text)
    cleaned_text = re.sub(r"\n+", "\n", cleaned_text)
    return cleaned_text

def process_pdf(filepath):
    """Process PDF file using pdfplumber"""
    try:
        logger.info(f"Processing PDF: {filepath}")
        
        # Extract text from PDF
        with pdfplumber.open(filepath) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Only add if text was extracted
                    text += page_text + "\n"
        
        if not text:
            logger.warning("No text extracted from PDF")
            return [{"date": "Warning", "description": "No text could be extracted from this PDF", "amount": 0}]
        
        # Clean the text
        text = clean_text(text)
        
        # Log a sample of the extracted text for debugging
        logger.debug(f"Extracted text sample: {text[:500]}...")
        
        # Try various patterns to match transaction data
        transactions = []
        
        # Pattern 1: Common date format with description and amount
        pattern1 = re.compile(
            r"(\d{1,2}[\/\-\.]\d{1,2}(?:[\/\-\.]\d{2,4})?|\d{1,2} \w{3}(?: \d{2,4})?)\s+(.+?)\s+([\d,]+\.\d{2}|-[\d,]+\.\d{2}|\([\d,]+\.\d{2}\))",
            re.MULTILINE
        )
        
        # Pattern 2: Alternative format often used in bank statements
        pattern2 = re.compile(
            r"(\d{1,2} \w{3}(?: \d{2,4})?|\w{3} \d{1,2}(?:, \d{4})?)\s+(.{10,}?)\s+([\d,]+\.\d{2}|-[\d,]+\.\d{2}|\([\d,]+\.\d{2}\))",
            re.MULTILINE
        )
        
        # Try first pattern
        for match in pattern1.finditer(text):
            date, description, amount = match.groups()
            
            # Process amount
            if "(" in amount and ")" in amount:
                # Handle negative amounts in parentheses
                amount = -float(amount.replace("(","").replace(")","").replace(",",""))
            elif amount.startswith("-"):
                amount = float(amount.replace(",",""))
            else:
                amount = float(amount.replace(",",""))
                
            transactions.append({
                "date": date.strip(),
                "description": description.strip(),
                "amount": amount
            })
        
        # If no transactions found with first pattern, try second pattern
        if not transactions:
            for match in pattern2.finditer(text):
                date, description, amount = match.groups()
                
                # Process amount
                if "(" in amount and ")" in amount:
                    amount = -float(amount.replace("(","").replace(")","").replace(",",""))
                elif amount.startswith("-"):
                    amount = float(amount.replace(",",""))
                else:
                    amount = float(amount.replace(",",""))
                    
                transactions.append({
                    "date": date.strip(),
                    "description": description.strip(),
                    "amount": amount
                })
        
        # If still no transactions, just extract potential line items
        if not transactions:
            logger.warning("No transactions found with standard patterns, using fallback")
            
            # Split text by lines and look for lines that might be transactions
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Look for lines with numbers that could be amounts
                amount_matches = re.findall(r'([\d,]+\.\d{2}|-[\d,]+\.\d{2}|\([\d,]+\.\d{2}\))', line)
                date_matches = re.findall(r'(\d{1,2}[\/\-\.]\d{1,2}(?:[\/\-\.]\d{2,4})?|\d{1,2} \w{3}(?: \d{4})?)', line)
                
                if amount_matches and date_matches:
                    # Found potential transaction line
                    date = date_matches[0]
                    amount = amount_matches[-1]  # Use last amount as transaction amount
                    
                    # Extract description by removing date and amount
                    description = line
                    for d in date_matches:
                        description = description.replace(d, '')
                    for a in amount_matches:
                        description = description.replace(a, '')
                    
                    # Process amount
                    if "(" in amount and ")" in amount:
                        amount = -float(amount.replace("(","").replace(")","").replace(",",""))
                    elif amount.startswith("-"):
                        amount = float(amount.replace(",",""))
                    else:
                        amount = float(amount.replace(",",""))
                        
                    transactions.append({
                        "date": date.strip(),
                        "description": description.strip(),
                        "amount": amount
                    })
        
        if not transactions:
            logger.warning("Failed to extract any transactions from PDF")
            # Return sample of the text for debugging
            return [{"date": "Debug", "description": f"Text sample: {text[:100]}...", "amount": 0}]
        
        logger.info(f"Successfully extracted {len(transactions)} transactions from PDF")
        return transactions
    
    except Exception as e:
        logger.exception(f"Error processing PDF: {str(e)}")
        return [{"date": "Error", "description": f"Failed to process PDF: {str(e)}", "amount": 0}]

def process_csv_excel(filepath, file_type):
    """Process CSV or Excel files"""
    try:
        logger.info(f"Processing {file_type} file: {filepath}")
        
        if file_type == 'csv':
            # Try different encodings
            for encoding in ['utf-8', 'latin1', 'ISO-8859-1']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return [{"date": "Error", "description": "Could not decode CSV file with supported encodings", "amount": 0}]
        else:  # Excel
            df = pd.read_excel(filepath)
        
        # Log dataframe info
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"DataFrame shape: {df.shape}")
        
        # Create a sample for debugging
        sample = df.head().to_dict() if not df.empty else {}
        logger.debug(f"DataFrame sample: {sample}")
        
        result = []
        
        # Try to identify columns based on common names
        date_col = None
        desc_col = None
        amount_col = None
        
        # Common column name patterns
        date_patterns = ['date', 'time', 'day', 'transaction date', 'posted date']
        desc_patterns = ['description', 'desc', 'narrative', 'details', 'transaction', 'particulars', 'memo']
        amount_patterns = ['amount', 'value', 'sum', 'balance', 'debit', 'credit', 'transaction amount']
        
        for col in df.columns:
            col_str = str(col).lower()
            
            if not date_col and any(pattern in col_str for pattern in date_patterns):
                date_col = col
                logger.debug(f"Found date column: {col}")
            
            elif not desc_col and any(pattern in col_str for pattern in desc_patterns):
                desc_col = col
                logger.debug(f"Found description column: {col}")
            
            elif not amount_col and any(pattern in col_str for pattern in amount_patterns):
                amount_col = col
                logger.debug(f"Found amount column: {col}")
        
        # If we couldn't identify all columns, try to use the first few columns
        if date_col is None and len(df.columns) > 0:
            date_col = df.columns[0]
            logger.debug(f"Using first column as date: {date_col}")
        
        if desc_col is None and len(df.columns) > 1:
            desc_col = df.columns[1]
            logger.debug(f"Using second column as description: {desc_col}")
        
        if amount_col is None:
            # Look for columns with numeric values
            for col in df.columns:
                if col != date_col and col != desc_col:
                    if pd.api.types.is_numeric_dtype(df[col]) or df[col].astype(str).str.replace(',', '').str.replace('-', '').str.replace('(', '').str.replace(')', '').str.replace('.', '').str.isdigit().any():
                        amount_col = col
                        logger.debug(f"Found numeric column to use as amount: {col}")
                        break
            
            # If still not found, use third column if available
            if amount_col is None and len(df.columns) > 2:
                amount_col = df.columns[2]
                logger.debug(f"Using third column as amount: {amount_col}")
        
        # If we have the necessary columns, process the data
        if date_col and desc_col and amount_col:
            for _, row in df.iterrows():
                date_val = str(row[date_col])
                desc_val = str(row[desc_col])
                
                # Process amount value
                amount_val = row[amount_col]
                if isinstance(amount_val, (int, float)):
                    amount_str = str(amount_val)
                else:
                    amount_str = str(amount_val)
                    # Clean up amount string
                    amount_str = amount_str.replace('$', '').replace(',', '')
                    # Handle parentheses for negative values
                    if '(' in amount_str and ')' in amount_str:
                        amount_str = '-' + amount_str.replace('(', '').replace(')', '')
                
                result.append({
                    "date": date_val,
                    "description": desc_val,
                    "amount": amount_str
                })
            
            logger.info(f"Successfully extracted {len(result)} transactions from {file_type}")
            return result
        else:
            missing_cols = []
            if not date_col: missing_cols.append("date")
            if not desc_col: missing_cols.append("description")
            if not amount_col: missing_cols.append("amount")
            
            logger.warning(f"Could not identify necessary columns: {', '.join(missing_cols)}")
            return [{"date": "Error", "description": f"Could not identify these columns: {', '.join(missing_cols)}", "amount": 0}]
    
    except Exception as e:
        logger.exception(f"Error processing {file_type}: {str(e)}")
        return [{"date": "Error", "description": f"Failed to process {file_type}: {str(e)}", "amount": 0}]

def process_txt(filepath):
    """Process text files with potential transaction data"""
    try:
        logger.info(f"Processing text file: {filepath}")
        result = []
        
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        # Try CSV-like format first
        lines = text.split('\n')
        csv_like = True
        
        for line in lines:
            if line.strip():
                parts = line.split(',')
                if len(parts) < 3:
                    csv_like = False
                    break
        
        if csv_like:
            logger.debug("Text file appears to be CSV-like")
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        result.append({
                            "date": parts[0].strip(),
                            "description": parts[1].strip(),
                            "amount": parts[2].strip()
                        })
        else:
            # Try bank statement-like format with regex
            logger.debug("Trying to extract transactions from text using regex")
            # Reuse PDF extraction logic for text files
            text = clean_text(text)
            
            # Try various patterns to match transaction data
            pattern = re.compile(
                r"(\d{1,2}[\/\-\.]\d{1,2}(?:[\/\-\.]\d{2,4})?|\d{1,2} \w{3}(?: \d{2,4})?)\s+(.+?)\s+([\d,]+\.\d{2}|-[\d,]+\.\d{2}|\([\d,]+\.\d{2}\))",
                re.MULTILINE
            )
            
            for match in pattern.finditer(text):
                date, description, amount = match.groups()
                
                # Process amount
                amount_str = amount
                if "(" in amount and ")" in amount:
                    amount_str = '-' + amount.replace("(","").replace(")","")
                
                result.append({
                    "date": date.strip(),
                    "description": description.strip(),
                    "amount": amount_str.replace(",", "")
                })
        
        if not result:
            logger.warning("No transactions found in text file")
            return [{"date": "Warning", "description": "No transactions found in text file", "amount": 0}]
        
        logger.info(f"Successfully extracted {len(result)} transactions from text file")
        return result
    
    except Exception as e:
        logger.exception(f"Error processing text file: {str(e)}")
        return [{"date": "Error", "description": f"Failed to process text file: {str(e)}", "amount": 0}]

# Continue from where the previous file cut off
def process_file(filepath):
    """Process different file types to extract transaction data"""
    filename = os.path.basename(filepath)
    extension = filename.rsplit('.', 1)[1].lower()
    
    try:
        if extension == 'csv':
            return process_csv_excel(filepath, 'csv')
        elif extension in ['xlsx', 'xls']:
            return process_csv_excel(filepath, 'excel')
        elif extension == 'pdf':
            return process_pdf(filepath)
        elif extension == 'txt':
            return process_txt(filepath)
        else:
            logger.warning(f"Unsupported file extension: {extension}")
            return [{"date": "Error", "description": f"Unsupported file type: {extension}", "amount": 0}]
    except Exception as e:
        logger.exception(f"General error processing file: {str(e)}")
        return [{"date": "Error", "description": f"Failed to process file: {str(e)}", "amount": 0}]

@app.route('/')
def index():
    clean_old_files()
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part was found in the request', 'error')  # Added category 'error'
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')  # Added category 'error'
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        filename = f"{unique_id}_{original_filename}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File saved: {filepath}")
        
        # Process the file
        result_data = process_file(filepath)
        
        # Store results in file instead of session
        result_id = store_result_data(result_data, original_filename)
        
        # Store only the ID and filename in session
        session['result_id'] = result_id
        session['original_filename'] = original_filename
        session['filepath'] = filepath  # Store for debugging
        
        # Add success message
        flash('File uploaded and processed successfully!', 'success')
        
        return redirect(url_for('results'))
    else:
        allowed_extensions = ', '.join(app.config["ALLOWED_EXTENSIONS"])
        flash(f'File type not allowed. Supported types: {allowed_extensions}', 'error')
        return redirect(url_for('index'))


@app.route('/results')
def results():
    """Display the results of file processing."""
    result_id = session.get('result_id')
    if not result_id:
        flash('No results found. Please upload a file first.', 'error')
        return redirect(url_for('index'))
    
    # Get data from file
    result_data = get_result_data(result_id)
    if not result_data:
        flash('Results data not found. Please upload a file again.', 'error')
        return redirect(url_for('index'))
    
    original_filename = session.get('original_filename', 'unknown file')
    
    # Log data for debugging
    logger.debug(f"Results route called with {len(result_data)} items for file: {original_filename}")
    logger.debug(f"First result item: {result_data[0] if result_data else 'No items'}")
    
    # Process the result data for display
    processed_results = []
    has_errors = False
    
    # Calculate summary stats for UI
    total_amount = 0
    total_positive = 0
    total_negative = 0
    
    for item in result_data:
        # Check if this is an error/warning row
        if item.get('date') and any(status in str(item['date']) for status in ['Error', 'Warning', 'Debug']):
            has_errors = True
            processed_results.append(item)
            continue
        
        # Try to ensure amount is numeric if possible
        try:
            amount = item.get('amount', 0)
            if isinstance(amount, str):
                # Clean up amount string
                amount = amount.replace('$', '').replace(',', '')
                if '(' in amount and ')' in amount:
                    amount = '-' + amount.replace('(', '').replace(')', '')
                amount = float(amount)
            else:
                amount = float(amount)
            
            # Update totals for summary stats
            if amount > 0:
                total_positive += amount
            else:
                total_negative += amount
            total_amount += amount
            
            processed_results.append({
                'date': item.get('date', ''),
                'description': item.get('description', ''),
                'amount': amount
            })
        except (ValueError, TypeError) as e:
            # If conversion fails, keep the original
            logger.warning(f"Failed to convert amount for item: {item}. Error: {str(e)}")
            processed_results.append(item)
    
    # Update the stored result data with processed results
    store_result_data(processed_results, original_filename)
    
    # Try/except to catch template rendering errors
    try:
        # First try the enhanced template with summary data
        return render_template('results.html', 
                            results=processed_results, 
                            filename=original_filename,
                            has_errors=has_errors,
                            total_amount=total_amount,
                            total_positive=total_positive,
                            total_negative=total_negative)
    except Exception as e:
        # If that fails, fall back to simple template
        logger.error(f"Error rendering results template: {str(e)}")
        try:
            return render_template('simple_results.html', 
                                results=processed_results, 
                                filename=original_filename)
        except Exception as e2:
            # If all else fails, show a basic response
            logger.error(f"Error rendering simple results template: {str(e2)}")
            return f"<h1>Results for {original_filename}</h1><p>Found {len(processed_results)} transactions</p><a href='/'>Back to Home</a>"

# Modify your download functions to use file storage
# This route was missing (it was called "download" but should be "download_text" to match the URL in templates)
@app.route('/download_text')
def download_text():
    result_id = session.get('result_id')
    if not result_id:
        flash('No data found to download', 'error')
        return redirect(url_for('index'))
    
    result_data = get_result_data(result_id)
    if not result_data:
        flash('Results data not found. Please upload a file again.', 'error')
        return redirect(url_for('index'))
    
    original_filename = session.get('original_filename', 'bank_data')
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as temp:
        for item in result_data:
            temp.write(f"Date: {item['date']}\n")
            temp.write(f"Description: {item['description']}\n")
            temp.write(f"Amount: {item['amount']}\n")
            temp.write("-" * 40 + "\n")
        temp_path = temp.name
    
    # Return the file with cleanup scheduled
    return create_temp_file_response(
        temp_path,
        f"{original_filename.split('.')[0]}_converted.txt"
    )

@app.route('/download_csv')
def download_csv():
    result_id = session.get('result_id')
    if not result_id:
        flash('No data found to download', 'error')
        return redirect(url_for('index'))
    
    result_data = get_result_data(result_id)
    if not result_data:
        flash('Results data not found. Please upload a file again.', 'error')
        return redirect(url_for('index'))
    
    original_filename = session.get('original_filename', 'bank_data')
    
    # Create temp file using csv module for proper handling of special characters
    import csv
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.csv', newline='') as temp:
        writer = csv.writer(temp)
        # Write header
        writer.writerow(['Date', 'Description', 'Amount'])
        # Write data
        for item in result_data:
            writer.writerow([
                str(item['date']),
                str(item['description']),
                str(item['amount'])
            ])
        temp_path = temp.name
    
    # Return the file with cleanup scheduled
    return create_temp_file_response(
        temp_path,
        f"{original_filename.split('.')[0]}_converted.csv"
    )

@app.route('/download_excel')
def download_excel():
    result_id = session.get('result_id')
    if not result_id:
        flash('No data found to download', 'error')
        return redirect(url_for('index'))
    
    result_data = get_result_data(result_id)
    if not result_data:
        flash('Results data not found. Please upload a file again.', 'error')
        return redirect(url_for('index'))
    
    original_filename = session.get('original_filename', 'bank_data')
    
    try:
        # Create a pandas DataFrame
        df = pd.DataFrame(result_data)
        
        # Create a temporary Excel file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp:
            temp_path = temp.name
        
        # Check if we need the openpyxl extra formatting
        try:
            from openpyxl.styles import Font, Alignment, PatternFill
            use_styling = True
        except ImportError:
            use_styling = False
            
        # Save DataFrame to Excel with basic formatting
        writer = pd.ExcelWriter(temp_path, engine='openpyxl')
        df.to_excel(writer, sheet_name='Transactions', index=False)
        
        # Apply formatting if available
        if use_styling:
            # Auto-adjust column widths
            for column in df:
                column_width = max(df[column].astype(str).map(len).max(), len(column))
                col_idx = df.columns.get_loc(column)
                writer.sheets['Transactions'].column_dimensions[chr(65 + col_idx)].width = column_width + 2
            
            # Format header
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            header_alignment = Alignment(horizontal="center", vertical="center")
            
            for cell in writer.sheets['Transactions'][1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_alignment
        
        # Save the workbook
        writer.close()
        
        # Return the file with cleanup scheduled
        return create_temp_file_response(
            temp_path,
            f"{original_filename.split('.')[0]}_converted.xlsx"
        )
    
    except Exception as e:
        logger.exception(f"Error creating Excel file: {str(e)}")
        flash(f"Error creating Excel file: {str(e)}", 'error')
        return redirect(url_for('results'))

# Modify clean_old_files to also clean up result files
def clean_old_files():
    """Delete files older than 1 hour"""
    now = datetime.now()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(filepath):
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if now - file_time > timedelta(hours=1):
                os.remove(filepath)

@app.route('/debug')
def debug_info():
    """Debug endpoint to see what's happening with file processing"""
    if not app.debug:
        return redirect(url_for('index'))
    
    debug_info = {
        'session': {k: v for k, v in session.items() if k != 'csrf_token'},
        'upload_dir': os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else [],
    }
    
    # If a file was processed, add extra debug info
    if 'filepath' in session and os.path.exists(session['filepath']):
        filepath = session['filepath']
        filename = os.path.basename(filepath)
        extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        debug_info['file'] = {
            'path': filepath,
            'size': os.path.getsize(filepath),
            'type': extension
        }
        
        # For text-based files, show sample content
        if extension in ['txt', 'csv']:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    debug_info['file']['sample'] = f.read(1000)
            except Exception as e:
                debug_info['file']['sample_error'] = str(e)
    
    return render_template('debug.html', debug_info=debug_info)

if __name__ == '__main__':
    app.run(debug=True , port=5049)