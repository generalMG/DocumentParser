import json
from collections import Counter
from datetime import datetime
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from functools import partial
import argparse

def process_chunk(lines):
    year_counter = Counter()
    version_counter = Counter()
    errors = 0
    
    for line in lines:
        try:
            arxiv_file = json.loads(line)
            
            if arxiv_file.get('versions') and len(arxiv_file['versions']) > 0:
                create_date_str = arxiv_file['versions'][0]['created']
                create_date = datetime.strptime(create_date_str, "%a, %d %b %Y %H:%M:%S %Z")
                year_counter[create_date.year] += 1
                version_counter[len(arxiv_file['versions'])] += 1
        
        except:
            errors += 1
    
    return year_counter, version_counter, errors

def read_in_chunks(file_path, chunk_size=50000):
    with open(file_path, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process arXiv metadata.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the arXiv metadata JSON file')
    args = parser.parse_args()
    file_path = args.file_path
    
    print(f"Using {cpu_count() // 4} CPU cores")
    
    with Pool(processes=cpu_count() // 4) as pool:
        results = list(pool.imap_unordered(process_chunk, read_in_chunks(file_path=file_path), chunksize=1))
    
    year_counts = Counter()
    version_counts = Counter()
    total_errors = 0
    
    for year_counter, version_counter, errors in results:
        year_counts.update(year_counter)
        version_counts.update(version_counter)
        total_errors += errors
    
    total_articles = sum(year_counts.values())
    
    threshold = total_articles * 0.001
    excluded_years = {v: c for v, c in version_counts.items() if c }