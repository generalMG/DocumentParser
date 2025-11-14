import json
from collections import Counter
from datetime import datetime
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from functools import partial
import argparse
import os

def process_chunk(lines):
    """Process a chunk of lines and return counters"""
    year_counter = Counter()
    version_counter = Counter()
    errors = 0

    for line in lines:
        try:
            article = json.loads(line)

            if article.get('versions') and len(article['versions']) > 0:
                # Extract year
                created_date_str = article['versions'][0]['created']
                created_date = datetime.strptime(created_date_str, "%a, %d %b %Y %H:%M:%S %Z")
                year_counter[created_date.year] += 1

                # Count versions
                version_counter[len(article['versions'])] += 1
        except:
            errors += 1

    return year_counter, version_counter, errors

def read_in_chunks(file_path, chunk_size=50000):
    """Generator to read file in chunks"""
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
    parser = argparse.ArgumentParser(description='Analyze arXiv metadata JSON file')
    parser.add_argument('file_path', nargs='?', default='./arxiv-metadata-oai-snapshot.json',
                        help='Path to the arXiv metadata JSON file (default: ./arxiv-metadata-oai-snapshot.json)')
    parser.add_argument('--cpu_count', type=int, default=cpu_count() // 4,
                        help='Number of CPU cores to use (default: quarter of available cores)')
    args = parser.parse_args()

    file_path = args.file_path

    print(f"Analyzing file: {file_path}")
    print(f"Using {args.cpu_count} CPU cores")
    print("Processing data in parallel...")

    # Process in parallel using imap_unordered (no pre-loading chunks)
    with Pool(processes=args.cpu_count) as pool:
        results = list(pool.imap_unordered(process_chunk, read_in_chunks(file_path), chunksize=1))

    # Merge results
    print("Merging results...")
    year_counts = Counter()
    version_counts = Counter()
    total_errors = 0

    for year_counter, version_counter, errors in results:
        year_counts.update(year_counter)
        version_counts.update(version_counter)
        total_errors += errors

    total_articles = sum(year_counts.values())

    # Filter out version counts < 0.1% threshold
    threshold = total_articles * 0.001
    excluded_versions = {v: c for v, c in version_counts.items() if c <= threshold}
    filtered_versions = {v: c for v, c in version_counts.items() if c > threshold}

    print(f"\n{'='*60}")
    print(f"Total articles: {total_articles:,}")
    print(f"Parse errors: {total_errors}")
    print(f"Threshold: {threshold:,.0f} articles")
    if excluded_versions:
        print(f"Excluded version counts (>{threshold:,.0f}): {sorted(excluded_versions.keys())}")
    print(f"{'='*60}")

    # Year analysis
    print("\nYEAR ANALYSIS")
    print("-" * 60)
    sorted_years = sorted(year_counts.items())
    for year, count in sorted_years[-10:]:  # Last 10 years
        print(f"{year}: {count:,} articles")

    # Version analysis (filtered)
    print("\nVERSION ANALYSIS (filtered)")
    print("-" * 60)
    sorted_versions = sorted(filtered_versions.items())
    for num_versions, count in sorted_versions:
        pct = (count / total_articles) * 100
        print(f"{num_versions} version(s): {count:,} ({pct:.2f}%)")

    # Generate plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Year histogram
    years = [y for y, _ in sorted_years]
    year_article_counts = [c for _, c in sorted_years]
    ax1.bar(years, year_article_counts, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Number of Articles', fontsize=12)
    ax1.set_title('Articles by Year', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Version histogram (filtered)
    version_nums = [v for v, _ in sorted_versions]
    version_article_counts = [c for _, c in sorted_versions]
    version_article_counts_percentages = []
    for v in version_article_counts:
        pct = (v / total_articles) * 100
        version_article_counts_percentages.append(pct)


    ax2.barh(version_nums, version_article_counts_percentages, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_ylabel('Number of Versions', fontsize=12)
    ax2.set_xlabel('Percentage of Total Articles (%)', fontsize=12)
    ax2.set_title('Articles by Version Count (filtered <0.1%)', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    if version_nums:
        ax2.set_yticks(version_nums)

    plt.tight_layout()

    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)

    output_path = 'outputs/arxiv_analysis_fast.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot as '{output_path}'")

    # Summary stats
    print(f"\nSUMMARY")
    print("-" * 60)
    print(f"Year range: {min(years)} - {max(years)}")
    print(f"Peak year: {max(sorted_years, key=lambda x: x[1])[0]}")
    print(f"Max versions (filtered): {max(version_nums)}")
    multi_version = sum(c for v, c in sorted_versions if v > 1)
    print(f"Multi-version articles (filtered): {multi_version:,} ({(multi_version/total_articles)*100:.2f}%)")

    #plt.show()
