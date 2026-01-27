from common_tools import *
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
import json
import os


paper_archive = json.load(open("data/processed_papers.json", "r"))


def build_judge_prompt_content(content_json):
    return (
        "$$$$ [The Review]:\n"
        + content_json["review_content"]
        + "\n\n$$$$ [Initial Score]: "
        + content_json["initial_score"]
        + "\n\n$$$$ [The Author's Rebuttal]:\n"
        + content_json["rebuttal"]
    )


def extract_final_score_from_text(text: str):
    """Extract the last number between 0 and 10 from the text as final_score"""
    pattern = r'\b(10(?:\.\d+)?|[0-9](?:\.\d+)?)\b'
    matches = re.findall(pattern, text)

    valid_scores = []
    for m in matches:
        try:
            v = float(m)
            if 0 <= v <= 10:
                valid_scores.append(v)
        except:
            continue

    if valid_scores:
        return {
            "final_score": int(valid_scores[-1]),
            "comment": text
        }
    return {"comment": text}


def gen_rebuttal_score(
    input_data,
    save_file,
    model='gpt-4o',
    is_api=True,
    n_gpus=1,
    recalc_only=False
):
    """
    When recalc_only=True:
        - Ignore input_data
        - Load existing assistant responses directly from save_file
        - Re-parse scores from assistant responses
        - Update final_score and statistics
    """

    # ------------------------------------------------------------
    # recalc_only = True → load old results from save_file
    # ------------------------------------------------------------
    if recalc_only:
        if not os.path.exists(save_file):
            raise FileNotFoundError(
                f"recalc_only=True but save_file does not exist: {save_file}"
            )

        old = json.load(open(save_file, "r"))

        if "reviews" not in old:
            raise ValueError(
                "save_file does not contain the 'reviews' field; cannot run recalc_only mode."
            )

        reviews = old["reviews"]
        # Jump directly to the score recalculation logic below
    else:
        # ------------------------------------------------------------
        # Normal mode: input_data is the valid input
        # ------------------------------------------------------------
        reviews = input_data

    # ------------------------------------------------------------
    # Build API requests only when recalc_only=False
    # ------------------------------------------------------------
    all_messages = []
    review_indices = []

    if not recalc_only:
        if not is_api:
            llm = LLM(
                model=model,
                dtype='bfloat16',
                tensor_parallel_size=n_gpus,
                gpu_memory_utilization=0.75,
            )
            tokenizer = AutoTokenizer.from_pretrained(model)

        for pidx, paper in enumerate(reviews):
            for ridx, review in enumerate(paper["reviews"]):
                if (
                    review.get("discussion")
                    and len(review["discussion"]) > 0
                    and review.get("final_score") is None
                ):
                    content_json = {
                        "review_content": review["review_content"],
                        "initial_score": review["initial_score"],
                        "discussion": review["discussion"],
                    }
                    msgs = [
                        {"role": "system", "content": REVIEWER_SYS_PROMPT},
                        {
                            "role": "user",
                            "content": build_judge_prompt_content(content_json),
                        },
                    ]
                    all_messages.append(msgs)
                    review_indices.append((pidx, ridx))

    # ------------------------------------------------------------
    # Call API (skipped when recalc_only=True)
    # ------------------------------------------------------------
    responses_processed = []

    if not recalc_only and all_messages:
        if is_api:
            responses = api_batch_inference(
                all_messages,
                sampling_params={"temperature": 0, "max_tokens": 5000},
                model=model,
                n_threads=10,
                progress=True
            )
        else:
            formatted = [
                tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False
                )
                for msgs in all_messages
            ]
            params = SamplingParams(temperature=0, max_tokens=5000)
            responses = [
                out.outputs[0].text.strip()
                for out in llm.generate(formatted, params)
            ]

        responses_processed = [
            extract_final_score_from_text(r) for r in responses
        ]

    updated_reviews = [paper.copy() for paper in reviews]

    # ------------------------------------------------------------
    # Handle reviews without rebuttal: final_score = initial_score
    # ------------------------------------------------------------
    for paper in updated_reviews:
        for review in paper["reviews"]:
            if not review.get("discussion") or len(review["discussion"]) == 0:
                review["final_score"] = review["initial_score"]

    # ------------------------------------------------------------
    # Handle reviews with rebuttals
    # ------------------------------------------------------------
    for idx, (pidx, ridx) in enumerate(review_indices):
        review = updated_reviews[pidx]["reviews"][ridx]
        initial_score = review["initial_score"]

        # ============================================================
        # recalc_only mode → recover assistant response from output file
        # ============================================================
        if recalc_only:
            if (
                len(review["discussion"]) >= 2
                and review["discussion"][1]["role"] == "assistant"
            ):
                assistant_text = review["discussion"][1]["content"]
                parsed = extract_final_score_from_text(assistant_text)
                new_score = parsed.get("final_score", initial_score)
                review["final_score"] = new_score
            else:
                review["final_score"] = initial_score
            continue

        # ============================================================
        # Normal mode: use API output
        # ============================================================
        if idx < len(responses_processed):
            gpt_res = responses_processed[idx]
            new_score = gpt_res.get("final_score", initial_score)
            rebuttal = review["discussion"][-1]["content"]

            review["final_score"] = new_score
            review["discussion"].append(
                {"content": gpt_res.get("comment", ""), "role": "assistant"}
            )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------
    increased = decreased = unchanged = 0
    ini_scores = []
    fin_scores = []

    for paper in updated_reviews:
        for r in paper["reviews"]:
            ini = r["initial_score"]
            fin = r["final_score"]

            if fin > ini:
                increased += 1
            elif fin < ini:
                decreased += 1
                # fin = ini
                # r["final_score"] = ini
            else:
                unchanged += 1

            ini_scores.append(ini)
            fin_scores.append(fin)

    stats = {
        "total_reviews": len(ini_scores),
        "score_increased": increased,
        "score_decreased": decreased,
        "score_unchanged": unchanged,
        "initial_avg_score": sum(ini_scores) / len(ini_scores),
        "final_avg_score": sum(fin_scores) / len(fin_scores),
    }

    # ------------------------------------------------------------
    # Save results (save_file is overwritten even in recalc_only mode)
    # ------------------------------------------------------------
    json.dump(
        {"stats": stats, "reviews": updated_reviews},
        open(save_file, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2
    )

    return {"stats": stats, "updated_reviews": updated_reviews}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="data/rebuttal/test_real.json")
    parser.add_argument("--output", "-o", type=str, default="data/revised_score/test_real_gpt.json")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o")
    parser.add_argument("--is_api", action='store_true')
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--limit", "-l", type=int, default=-1)
    parser.add_argument("--recalc_only", action="store_true")

    args = parser.parse_args()

    # Load input file (only used when recalc_only=False)
    input_data = json.load(open(args.input, "r"))

    if not args.recalc_only and args.limit != -1:
        input_data = input_data[:args.limit]

    gen_rebuttal_score(
        input_data,
        save_file=args.output,
        model=args.model,
        is_api=args.is_api,
        n_gpus=args.n_gpus,
        recalc_only=args.recalc_only,
    )
