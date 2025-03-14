def ensure_directory(path):
    """Checks if directory exists, creates it if it does not exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)