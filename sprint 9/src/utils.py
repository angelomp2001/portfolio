
import hashlib
import os

def calculate_file_hash(file_content: bytes) -> str:
    """
    Calculates SHA-256 hash of file content.
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(file_content)
    return sha256_hash.hexdigest()

def check_duplicate(new_file_content: bytes, data_dir: str = "data") -> tuple[bool, str]:
    """
    Checks if the uploaded file content matches any existing file in data_dir.
    Returns (is_duplicate, matching_filename).
    """
    new_hash = calculate_file_hash(new_file_content)
    
    if not os.path.exists(data_dir):
        return False, ""
        
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "rb") as f:
                existing_content = f.read()
                existing_hash = calculate_file_hash(existing_content)
                
                if new_hash == existing_hash:
                    return True, filename
                    
    return False, ""
