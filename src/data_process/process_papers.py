"""
Process paper data under the Re2_data/papers directory.
Find all ultimate folders that contain the Initial_manuscript_md subfolder.
"""

import os
from pathlib import Path
from typing import List, Generator
import logging
from datetime import datetime
from tqdm import tqdm
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Token-related constants
MAX_TOKENS = 25000
ENCODING_NAME = "cl100k_base"  # Encoding used by GPT-3.5 / GPT-4


def count_and_truncate_tokens(text: str, max_tokens: int = MAX_TOKENS) -> tuple[str, int, int]:
    """
    Count the number of tokens in the text and truncate it to a maximum length.

    Args:
        text: Input text
        max_tokens: Maximum number of tokens allowed

    Returns:
        tuple: (truncated_text, original_token_count, truncated_token_count)
    """
    try:
        # Get tokenizer
        encoding = tiktoken.get_encoding(ENCODING_NAME)

        # Count original tokens
        tokens = encoding.encode(text)
        original_token_count = len(tokens)

        # Truncate if exceeding maximum length
        if original_token_count > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            truncated_text = encoding.decode(truncated_tokens)
            truncated_token_count = len(truncated_tokens)

            logger.info(
                f"Text truncated: original {original_token_count} tokens -> {truncated_token_count} tokens"
            )
            return truncated_text, original_token_count, truncated_token_count
        else:
            return text, original_token_count, original_token_count

    except Exception as e:
        logger.error(f"Error during token processing: {e}")
        # If tiktoken fails, return original text
        return text, 0, 0


def find_ultimate_folders(papers_root: str) -> Generator[Path, None, None]:
    """
    Recursively find all ultimate folders that contain an Initial_manuscript_md subfolder.

    Args:
        papers_root: Root path of the papers directory

    Yields:
        Path: Folder path that contains Initial_manuscript_md
    """
    papers_path = Path(papers_root)

    if not papers_path.exists() or not papers_path.is_dir():
        logger.error(f"Papers directory does not exist or is not a directory: {papers_root}")
        return

    # Use os.walk for depth-first traversal (more efficient)
    for root, dirs, files in os.walk(papers_path):
        # Check whether the current directory contains Initial_manuscript_md
        if 'Initial_manuscript_md' in dirs:
            ultimate_folder = Path(root)
            yield ultimate_folder

            # Once Initial_manuscript_md is found, do not descend further
            # This avoids redundant checks in confirmed ultimate folders
            dirs.clear()  # Stop further traversal into this directory


def get_all_ultimate_folders(papers_root: str) -> List[Path]:
    """
    Get a list of all ultimate folders.

    Args:
        papers_root: Root path of the papers directory

    Returns:
        List[Path]: List of all ultimate folder paths
    """
    ultimate_folders = list(find_ultimate_folders(papers_root))
    logger.info(f"Total ultimate folders found: {len(ultimate_folders)}")
    return ultimate_folders


def analyze_ultimate_folders(papers_root: str) -> dict:
    """
    Analyze the distribution of ultimate folders.

    Args:
        papers_root: Root path of the papers directory

    Returns:
        dict: Dictionary containing analysis statistics
    """
    ultimate_folders = get_all_ultimate_folders(papers_root)

    # Statistics grouped by conference/journal
    conference_stats = {}

    for folder in ultimate_folders:
        # Extract conference/journal information
        # (Assumed path structure: papers/Conference/Year/...)
        parts = folder.relative_to(papers_root).parts
        if len(parts) >= 1:
            conference = parts[0]
            if conference not in conference_stats:
                conference_stats[conference] = 0
            conference_stats[conference] += 1

    analysis = {
        'total_ultimate_folders': len(ultimate_folders),
        'conference_distribution': conference_stats,
        'folder_paths': [str(folder) for folder in ultimate_folders]
    }

    return analysis


def print_analysis_report(papers_root: str):
    """
    Print the analysis report.

    Args:
        papers_root: Root path of the papers directory
    """
    analysis = analyze_ultimate_folders(papers_root)

    print("\n" + "=" * 60)
    print("Ultimate Folder Analysis Report")
    print("=" * 60)
    print(f"Total: {analysis['total_ultimate_folders']} ultimate folders")
    print("\nDistribution by conference/journal:")

    # Sort and display by count
    sorted_conferences = sorted(
        analysis['conference_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for conference, count in sorted_conferences:
        print(f"  {conference}: {count}")

    print("\nFirst 10 example folder paths:")
    for i, path in enumerate(analysis['folder_paths'][:10]):
        print(f"  {i + 1}. {path}")

    if len(analysis['folder_paths']) > 10:
        print(f"  ... and {len(analysis['folder_paths']) - 10} more folders")


def main():
    """Main entry point"""
    import sys

    # Path to the papers directory
    papers_root = "/data/ph16/Graph_of_Persuasion/data/raw_data/Re2_data/papers"

    print_analysis_report(papers_root)
    folders = get_all_ultimate_folders(papers_root)

    papers = {}

    for folder in tqdm(folders, desc="Processing folders"):
        paper_id = str(folder).split('/')[-1]
        paper_file = os.path.join(
            folder, "Initial_manuscript_md", "Initial_manuscript.md"
        )

        try:
            with open(paper_file, "r", encoding='utf-8') as f:
                paper_content = f.read()

            # Locate the References section
            ref_pos = paper_content.find("\n## References\n\n")
            if ref_pos == -1:
                ref_pos = paper_content.find("\n## REFERENCES\n\n")
            if ref_pos == -1:
                ref_pos = paper_content.find("\nReferences\n\n")
            if ref_pos == -1:
                ref_pos = paper_content.find("\nREFERENCES\n\n")

            # Truncate content before References
            if ref_pos != -1:
                paper_content = paper_content[:ref_pos]
            else:
                pass
                # logger.info(f"References section not found, content not truncated: {paper_id}")

            # Count tokens and truncate if necessary
            truncated_content, original_tokens, final_tokens = count_and_truncate_tokens(
                paper_content
            )

            papers[paper_id] = truncated_content

        except FileNotFoundError:
            logger.warning(f"Paper file not found: {paper_file}")
            continue
        except Exception as e:
            logger.error(f"Error processing paper {paper_id}: {e}")
            continue

    # Save results to a JSON file
    import json
    output_file = "data/processed_papers.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(papers, f, indent=4, ensure_ascii=False)

    print(f"\nProcessing completed! Results saved to {output_file}")


if __name__ == "__main__":
    main()
