import json
import logging

def load_jsonl(filepath):
    """
    Helper function to load a JSONL file safely.
    """
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        logging.info(f"Successfully loaded {len(data)} records from {filepath}")
    except Exception as e:
        logging.error(f"Failed to load JSONL from {filepath}: {e}")
    return data
