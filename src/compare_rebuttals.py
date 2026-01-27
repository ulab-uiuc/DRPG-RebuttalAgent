from common_tools import *
import argparse
import random
import json
import os

paper_archive = json.load(open("data/processed_papers.json", "r"))


def find_last_digit(text: str):
    text = text[-100:]
    if "two responses are similar in quality" in text.lower():
        return -1
    for ch in reversed(text):
        if '0' <= ch <= '9':
            return int(ch)
    return -1



# ============================================================================
# recalc_only mode: fully independent, only read output file
# ============================================================================
def recalc_from_output_file(output_file):
    data = json.load(open(output_file, "r"))

    results = data["results"]

    total_reviews = 0
    total_reversed = 0
    judgement_counts = {1: 0, 2: 0, -1: 0}

    for paper in results:
        for review in paper["reviews"]:
            total_reviews += 1

            comment = review["comment"]
            reversed_position = review.get("reversed", False)

            judgement = find_last_digit(comment)

            # reverse back
            if reversed_position and judgement > 0:
                judgement = 3 - judgement

            review["judgement"] = judgement

            # stats
            judgement_counts[judgement] = judgement_counts.get(judgement, 0) + 1
            if reversed_position:
                total_reversed += 1

    # update stats
    data["stats"] = {
        "total_papers": len(results),
        "total_reviews": total_reviews,
        "total_reversed": total_reversed,
        "rebuttal_1_wins": judgement_counts.get(1, 0),
        "rebuttal_2_wins": judgement_counts.get(2, 0),
        "no_clear_preference": judgement_counts.get(-1, 0)
    }

    # save back
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Recalculation complete.")
    print(f"Results saved to {output_file}")
    return


# ============================================================================
# Normal mode: full pipeline (unchanged)
# ============================================================================
def gen_comparison_rebuttal_score(
    reviews_1,
    reviews_2,
    model='gpt-4o',
    save_file=None,
    seed=42,
):
    random.seed(seed)

    all_messages = []
    paper_review_mapping = []

    for paper_id in range(len(reviews_1)):
        paper = paper_archive.get(reviews_1[paper_id]["paper_id"], "No content found.")
        for review_id in range(len(reviews_1[paper_id]["reviews"])):
            review_content = reviews_1[paper_id]["reviews"][review_id]["review_content"]

            has_rebuttal_1 = (
                "discussion" in reviews_1[paper_id]["reviews"][review_id]
                and len(reviews_1[paper_id]["reviews"][review_id]["discussion"]) > 0
            )
            has_rebuttal_2 = (
                "discussion" in reviews_2[paper_id]["reviews"][review_id]
                and len(reviews_2[paper_id]["reviews"][review_id]["discussion"]) > 0
            )

            if not has_rebuttal_1 or not has_rebuttal_2:
                paper_review_mapping.append({
                    "paper_id": reviews_1[paper_id]["paper_id"],
                    "paper_index": paper_id,
                    "review_id": review_id,
                    "review_content": review_content,
                    "rebuttal_1": reviews_1[paper_id]["reviews"][review_id]["discussion"][0]["content"] if has_rebuttal_1 else "",
                    "rebuttal_2": reviews_2[paper_id]["reviews"][review_id]["discussion"][0]["content"] if has_rebuttal_2 else "",
                    "no_rebuttal": True
                })
                continue

            rebuttal_1 = reviews_1[paper_id]["reviews"][review_id]["discussion"][0]["content"]
            rebuttal_2 = reviews_2[paper_id]["reviews"][review_id]["discussion"][0]["content"]

            reversed_position = random.choice([True, False])

            if reversed_position:
                first_rebuttal = rebuttal_2
                second_rebuttal = rebuttal_1
            else:
                first_rebuttal = rebuttal_1
                second_rebuttal = rebuttal_2

            content = ""
            content += "$$$$ [The Review]:\n\n" + review_content + "\n\n"
            content += "$$$$ [Response 1]:\n\n" + first_rebuttal + "\n\n"
            content += "$$$$ [Response 2]:\n\n" + second_rebuttal + "\n\n"

            messages = [
                {"role": "system", "content": REVIEWER_COMPARE_SYS_PROMPT},
                {"role": "user", "content": content}
            ]

            all_messages.append(messages)
            paper_review_mapping.append({
                "paper_id": reviews_1[paper_id]["paper_id"],
                "paper_index": paper_id,
                "review_id": review_id,
                "review_content": review_content,
                "rebuttal_1": rebuttal_1,
                "rebuttal_2": rebuttal_2,
                "reversed": reversed_position,
                "no_rebuttal": False
            })

    # API inference
    if all_messages:
        responses = api_batch_inference(
            all_messages,
            sampling_params={"temperature": 0.6, "max_tokens": 5000},
            model=model,
            n_threads=10,
            progress=True
        )
    else:
        responses = []

    papers_dict = {}
    response_idx = 0

    for mapping in paper_review_mapping:
        paper_id = mapping["paper_id"]

        if paper_id not in papers_dict:
            papers_dict[paper_id] = {"paper_id": paper_id, "reviews": []}

        if mapping.get("no_rebuttal", False):
            papers_dict[paper_id]["reviews"].append({
                "review_content": mapping["review_content"],
                "rebuttal_1": mapping["rebuttal_1"],
                "rebuttal_2": mapping["rebuttal_2"],
                "comment": "",
                "judgement": -1,
                "reversed": False
            })
            continue

        response = responses[response_idx]
        response_idx += 1
        judgement = find_last_digit(response)
        reversed_position = mapping["reversed"]

        if reversed_position and judgement > 0:
            judgement = 3 - judgement

        papers_dict[paper_id]["reviews"].append({
            "review_content": mapping["review_content"],
            "rebuttal_1": mapping["rebuttal_1"],
            "rebuttal_2": mapping["rebuttal_2"],
            "comment": response,
            "judgement": judgement,
            "reversed": reversed_position
        })

    return list(papers_dict.values())


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rebuttal scores for reviews")
    parser.add_argument("--file_1", "-i", type=str)
    parser.add_argument("--file_2", "-j", type=str)
    parser.add_argument("--model", "-m", type=str, default="gpt-4o")
    parser.add_argument("--limit", "-l", type=int, default=-1)
    parser.add_argument("--recalc_only", action="store_true")
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()

    # ------------------------------------------------------------
    # recalc_only mode → skip everything else
    # ------------------------------------------------------------
    if args.recalc_only:
        recalc_from_output_file(args.output)
        exit()

    # ------------------------------------------------------------
    # normal mode
    # ------------------------------------------------------------
    reviews_1 = json.load(open(args.file_1, "r"))
    reviews_2 = json.load(open(args.file_2, "r"))
    if args.limit != -1:
        reviews_1 = reviews_1[:args.limit]
        reviews_2 = reviews_2[:args.limit]

    results = gen_comparison_rebuttal_score(
        reviews_1, reviews_2,
        model=args.model,
        save_file=args.output,
        seed=args.seed
    )

    # stats
    total_reviews = 0
    total_reversed = 0
    judgement_counts = {1: 0, 2: 0, -1: 0}
    for paper in results:
        total_reviews += len(paper["reviews"])
        for review in paper["reviews"]:
            j = review["judgement"]
            judgement_counts[j] = judgement_counts.get(j, 0) + 1
            if review.get("reversed", False):
                total_reversed += 1

    output_data = {
        "args": vars(args),
        "stats": {
            "total_papers": len(results),
            "total_reviews": total_reviews,
            "total_reversed": total_reversed,
            "rebuttal_1_wins": judgement_counts.get(1, 0),
            "rebuttal_2_wins": judgement_counts.get(2, 0),
            "no_clear_preference": judgement_counts.get(-1, 0)
        },
        "results": results
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output}")
