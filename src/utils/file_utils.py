#src/utils/file_utils.py
import os
import datetime

from src.configuration.config import Config
from src.utils.text_utils import clean_text


def save_generated_text(generated_text, optional_name=None):
    """
    Save generated text to a file with an optional name and a timestamp in the /content/story folder.

    :param generated_text: The text to be saved.
    :param optional_name: An optional name to include in the filename.
    """
    # Clean the generated text
    cleaned_text = clean_text(generated_text)

    # Define the folder path
    folder_path = Config.PDF_DIRECTORY

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the filename
    if optional_name:
        file_name = f"{optional_name}_{timestamp}.txt"
    else:
        file_name = f"generated_text_{timestamp}.txt"

    # Complete file path
    file_path = os.path.join(folder_path, file_name)

    # Write the generated text to a file
    with open(file_path, 'w') as file:
        file.write(cleaned_text)

    print(f"Text has been saved to {file_path}")
