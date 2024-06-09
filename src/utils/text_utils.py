#src/utils/text_utils.py
import re

def clean_text(generated_content):
    """
    Clean the text by removing URLs, dates, names, and emails.

    :param generated_content: The text to be cleaned.
    :return: Cleaned text.
    """
    # Regex patterns to match sensitive information
    url_pattern = r'https?://\S+|www\.\S+'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b'


    # Remove or replace sensitive information
    generated_content = re.sub(url_pattern, '', generated_content)
    generated_content = re.sub(email_pattern, '', generated_content)
    generated_content = re.sub(date_pattern, '', generated_content)


    return generated_content
