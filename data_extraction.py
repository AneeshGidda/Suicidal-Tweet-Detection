# Import the 'shutil' module, which provides file operations and utilities.
import shutil

# Define a function called 'unzip_file' that takes two parameters:
# - 'zip_file_path': The path to the compressed archive file.
# - 'destination_directory': The directory where the contents of the archive will be extracted.
def unzip_file(zip_file_path, destination_directory):
    # Use the 'shutil.unpack_archive' function to extract the contents of the archive.
    # It takes the 'zip_file_path' as the source file and 'destination_directory' as the extraction location.
    shutil.unpack_archive(zip_file_path, destination_directory)

# Define the path to the compressed archive file you want to unzip.
zip_file_path = r"C:\Users\Aneesh\Downloads\archive (1).zip"

# Define the directory where the extracted contents will be placed.
destination_directory = r"C:\Users\Aneesh\Codes\BERT Sentiment Analysis"

# Call the 'unzip_file' function with the specified 'zip_file_path' and 'destination_directory'.
# This will extract the contents of the archive to the specified directory.
unzip_file(zip_file_path, destination_directory)
