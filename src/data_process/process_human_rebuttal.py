'''
Extract "Perspectives" from human rebuttals using LLMs.
'''
import json
import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common_tools import api_batch_inference, process_json

import re

def has_table(text: str) -> bool:
    # Check for HTML-style tables
    if "<table>" in text and "</table>" in text:  # markdown/HTML table
        return True

    # Remove code blocks before checking for tables
    text_no_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    lines = text_no_code.splitlines()
    count = 0

    for line in lines:
        # Check whether the line looks like a table row (at least two vertical bars)
        if line.strip().count('|') >= 2:
            count += 1
            # At least two consecutive lines matching the condition
            if count >= 2:
                return True
        else:
            count = 0

    return False


SYS_PROMPT = '''You are a helpful assistant. You will receive an author's response to an academic paper review. You will also receive a set of questions.
Your task is to decompose the rebuttal structure. For each question, you need to find:
    1. Does the response address this question directly? (Yes/No)
    2. If the first question is Yes, extract the relevant paragraph(s) of the response that addresses the question. Otherwise, leave it EMPTY.
        - You shouldn't paraphrase or summarize the response, but directly copy the relevant part.
        - You should provide a complete paragraph instead of a sentence or phrase.
    3. If the question is affressed, you should summary the PERSPECTIVE of the response into one single point, categorized as either "Justification" or "Clarification".
            - Clarification means the author points out FACTUAL ERRORS or misunderstandings in the reviewer's question.
            - Justification means the author (potentially) admits the reviewer's comment is factually correct, but provides REASONS or ARGUMENTS to support their original design/decision.
    4. The summarized perspective should be a high-level description, not specific details.
        - Some examples are:
            - "Clarification: The method used in the paper is proved by ablation studies."
            - "Clarification: the choice of image size and resolution was deliberate and based on the specific requirements of our methodology."
            - "Justification: the paper's contributions are multifaceted, and the remaining content is a crucial aspect of the overall narrative."
            - "Justification: while not achieving optimal performance, the method used in the paper is widely adopted in prior works." 
[
    {
        "question": "<the question>",
        "addressed": "<Yes/No>",
        "response": "...",
        "perspective": "...".
    },
    ...
]
'''

def main():
    parser = argparse.ArgumentParser(description='Process human rebuttal data')
    parser.add_argument(
        '--review_file',
        type=str,
        default="data/test.json",
        help='Path to the review file (default: data/test.json)'
    )
    parser.add_argument(
        '--responses_file',
        type=str,
        default="data/revised_score/test_real_real.json",
        help='Path to the responses file (default: data/revised_score/test_real_real.json)'
    )

    args = parser.parse_args()

    review_file = args.review_file
    responses_file = args.responses_file

    reviews = json.load(open(review_file, "r"))
    responses = json.load(open(responses_file, "r"))["reviews"]

    all_messages = []

    for paper_idx, paper in enumerate(reviews):
        for review_idx, review in enumerate(paper["reviews"]):
            if responses[paper_idx]['reviews'][review_idx]["discussion"] == []:
                continue
            questions = "".join(
                [f'"{element}"\n' for idx, element in enumerate(review["decomposed_content"])]
            )
            human_rebuttal = responses[paper_idx]['reviews'][review_idx]["discussion"][0]["content"]
            all_messages.append([
                {"role": "system", "content": SYS_PROMPT},
                {
                    "role": "user",
                    "content": f"Questions:\n\n{questions}\n\nAuthor\'s Response:\n\n{human_rebuttal}"
                }
            ])

    print(f"There are total {len(all_messages)} messages prepared.")

    # Chunked processing mechanism: at most 6000 messages per chunk
    chunk_size = 6000
    total_messages = len(all_messages)
    num_chunks = (total_messages + chunk_size - 1) // chunk_size  # Ceiling division

    tmp_file = "1.tmp"
    responses = []

    # Try to load previously processed results from the temporary file
    if Path(tmp_file).exists():
        print(f"Found temporary file {tmp_file}, loading previous results...")
        with open(tmp_file, 'r', encoding='utf-8') as f:
            tmp_data = json.load(f)
            # Verify whether the total number of messages matches
            if tmp_data.get("total_messages") != total_messages:
                print(
                    f"Warning: Total messages mismatch! "
                    f"Previous: {tmp_data.get('total_messages')}, Current: {total_messages}"
                )
                print("Starting from scratch and backing up old temporary file...")
                Path(tmp_file).rename(f"{tmp_file}.backup")
                start_chunk = 0
                responses = []
            else:
                responses = tmp_data.get("responses", [])
                start_chunk = tmp_data.get("last_chunk", -1) + 1
                print(
                    f"Loaded {len(responses)} responses, "
                    f"resuming from chunk {start_chunk + 1}/{num_chunks}"
                )
    else:
        start_chunk = 0
        print("No temporary file found, starting from scratch.")

    # Process data chunk by chunk
    for chunk_idx in range(start_chunk, num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_messages)
        chunk_messages = all_messages[start_idx:end_idx]

        print(
            f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} "
            f"(messages {start_idx} to {end_idx - 1})..."
        )

        chunk_responses = api_batch_inference(
            chunk_messages,
            sampling_params={"temperature": 0.5, "max_tokens": 10000},
            model="meta/llama-3.3-70b-instruct",
            n_threads=6,
            progress=True
        )

        responses.extend(chunk_responses)

        # Save temporary results
        tmp_data = {
            "last_chunk": chunk_idx,
            "total_chunks": num_chunks,
            "total_messages": total_messages,
            "responses": responses
        }
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(tmp_data, f, ensure_ascii=False, indent=2)

        print(f"Chunk {chunk_idx + 1}/{num_chunks} completed and saved to {tmp_file}")

    print(f"\nAll chunks processed! Total responses: {len(responses)}")

    responses_processed = [process_json(response) for response in responses]

    # 1) Check whether the "response" field contains a table; if so, set it to empty
    for response_list in responses_processed:
        if isinstance(response_list, list):
            for item in response_list:
                if isinstance(item, dict) and "response" in item:
                    if has_table(item["response"]):
                        item["response"] = ""

    # 2) Add rebuttal_summary to each review in responses_file and save
    responses_data = json.load(open(responses_file, "r"))
    response_idx = 0

    for paper_idx, paper in enumerate(reviews):
        for review_idx, review in enumerate(paper["reviews"]):
            if responses_data["reviews"][paper_idx]['reviews'][review_idx]["discussion"] == []:
                # If there is no discussion, set an empty rebuttal_summary
                responses_data["reviews"][paper_idx]['reviews'][review_idx]["rebuttal_summary"] = []
            else:
                # Attach the corresponding rebuttal_summary
                new = responses_processed[response_idx]
                responses_data["reviews"][paper_idx]['reviews'][review_idx]["rebuttal_summary"] = new
                response_idx += 1

    # Save to a new file
    with open(responses_file + ".new", "w", encoding="utf-8") as f:
        json.dump(responses_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully processed {response_idx} rebuttals and saved to {responses_file}")

    # Remove temporary file
    if Path(tmp_file).exists():
        Path(tmp_file).unlink()
        print(f"Temporary file {tmp_file} has been removed.")


if __name__ == "__main__":
    main()
