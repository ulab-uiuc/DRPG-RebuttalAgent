from common_tools import *
import argparse
import json
import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from planner.train_planner import SoftMatchingPlanner
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# Global resources
# =========================

paper_archive = json.load(open("data/processed_papers.json", "r"))
paper_paragraphs_archive = json.load(open("data/processed_paper_paragraphs.json", "r"))

llm = None
tokenizer = None
MAX_TOKENS = 27000


# =========================
# Utilities
# =========================

def clean_model_output(text):
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def ensure_llm_initialized(args, model_path=None):
    global llm, tokenizer
    model_path = model_path or args.model

    if llm is None:
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=args.n_gpus,
            gpu_memory_utilization=0.85,
            disable_sliding_window=False,
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except:
            pass
    return llm, tokenizer


# =========================
# Prompt builder (2.1 only)
# =========================

def build_rebuttal_prompt_content(
    paper_content,
    review_point,
    selected_perspective=None,
    conf_valid=True,
    sft_model=False
):
    content = ""

    paper_label = "$$$$ [Relevant Paragraphs]:\n\n" if sft_model else "$$$$ [Paper Excerpt]:\n\n"
    content += paper_label + paper_content + "\n\n"
    content += "$$$$ [The Review]:\n\n" + review_point + "\n\n"

    if selected_perspective and conf_valid:
        hint_label = "$$$$ [Perspective]:\n\n" if sft_model else "$$$$ [Hint]\n\n"
        content += hint_label
        content += (
            "You may consider constructing your response with the following perspective. "
            "However, you shall still focus on the paper content and make concrete responses.\n"
        )
        content += f"Perspective: {selected_perspective}\n\n"

    if tokenizer:
        tokens = tokenizer.encode(content, add_special_tokens=False)[:MAX_TOKENS]
        content = tokenizer.decode(tokens, skip_special_tokens=True)

    return content


# =========================
# Main pipeline (2.1)
# =========================

def gen_rebuttal(paper_reviews_without_rebuttal, args):

    # ---------- Load planner ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.planner_model, map_location=device)
    config = checkpoint.get("config", {})

    planner_model = SoftMatchingPlanner(
        pretrained_encoder=config.get("encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        mlp_hidden=config.get("mlp_hidden", [512]),
        freeze_encoder=config.get("freeze_encoder", False),
    ).to(device)

    planner_model.load_state_dict(checkpoint["model_state_dict"])
    planner_model.eval()

    print(f"Loaded planner model from {args.planner_model}")

    # ---------- Step 1: Decompose reviews ----------
    all_messages = []
    original_reviews = []

    for paper in paper_reviews_without_rebuttal:
        for review in paper["reviews"]:
            all_messages.append([
                {"role": "system", "content": REBUTTAL_DECOMPOSER_SYS_PROMPT},
                {"role": "user", "content": review["review_content"]},
            ])
            original_reviews.append(review["review_content"])

    llm, tokenizer = ensure_llm_initialized(args)
    all_messages = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        for m in all_messages
    ]

    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=8000)
    outputs = llm.generate(all_messages, params)

    review_points = []
    for i, out in enumerate(outputs):
        points = process_json(out.outputs[0].text)
        if not isinstance(points, list) or len(points) == 0:
            points = [original_reviews[i]]
        review_points.append(points)

    # ---------- Step 2: Retrieval ----------
    encoder = SentenceTransformer(args.encoder_model)

    review_idx = 0
    for paper in paper_reviews_without_rebuttal:
        paper_id = paper["paper_id"]
        paper_paragraphs = paper_paragraphs_archive.get(
            paper_id, sep_passage(paper_archive.get(paper_id, ""))
        )
        doc_embeddings = encoder.encode(paper_paragraphs, convert_to_tensor=True)

        for review in paper["reviews"]:
            review["relevant_paragraphs"] = []
            for point in review_points[review_idx]:
                retrieved = retrieve(
                    [point],
                    paper_paragraphs,
                    encoder,
                    doc_embeddings,
                    top_k=args.top_k
                )[0][0]
                review["relevant_paragraphs"].append(sorted(retrieved))
            review_idx += 1

    # ---------- Step 3: Perspective generation ----------
    perspective_messages = []
    for points in review_points:
        for p in points:
            perspective_messages.append([
                {"role": "system", "content": REBUTTAL_PERSPECTIVE_SYS_PROMPT},
                {"role": "user", "content": p},
            ])

    perspective_messages = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        for m in perspective_messages
    ]

    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=5000)
    outputs = llm.generate(perspective_messages, params)

    all_perspectives = []
    for out in outputs:
        pers = process_json(out.outputs[0].text)
        all_perspectives.append(pers if isinstance(pers, list) else [])

    # ---------- Step 4: Planner selection ----------
    planner_inputs = []
    positions = []

    idx = 0
    review_idx = 0
    for paper in paper_reviews_without_rebuttal:
        paper_id = paper["paper_id"]
        paper_paragraphs = paper_paragraphs_archive[paper_id]

        for review in paper["reviews"]:
            for i, point in enumerate(review_points[review_idx]):
                perspectives = all_perspectives[idx]
                if len(perspectives) > 0:
                    retrieved = review["relevant_paragraphs"][i][:args.top_k]
                    planner_inputs.append({
                        "paper_id": paper_id,
                        "passages": retrieved,
                        "passages_content": [paper_paragraphs[j] for j in retrieved],
                        "perspectives": [
                            str(p).replace("Justification: ", "").replace("Clarification: ", "")
                            for p in perspectives
                        ]
                    })
                    positions.append((review_idx, i, idx))
                idx += 1
            review_idx += 1

    results = []
    for i in tqdm(range(0, len(planner_inputs), 64)):
        batch = planner_inputs[i:i + 64]
        results.extend(planner_model.predict_best_view(batch, return_confidence=True))

    selected = {}
    for (r_idx, p_idx, g_idx), (best, conf) in zip(positions, results):
        selected[(r_idx, p_idx)] = (
            all_perspectives[g_idx][best],
            conf
        )

    # ---------- Step 5: Rebuttal generation ----------
    messages = []
    review_idx = 0
    ans_map = []

    for paper in paper_reviews_without_rebuttal:
        paper_id = paper["paper_id"]
        paragraphs = paper_paragraphs_archive[paper_id]

        for review in paper["reviews"]:
            for i, point in enumerate(review_points[review_idx]):
                retrieved = review["relevant_paragraphs"][i][:args.top_k]
                content = merge_passages([paragraphs[j] for j in retrieved])

                sel, conf = selected.get((review_idx, i), (None, 0.0))
                sel = None if sel is None else str(sel).replace("Justification: ", "").replace("Clarification: ", "")

                prompt = build_rebuttal_prompt_content(
                    content,
                    point,
                    sel,
                    conf > args.conf_threshold,
                    sft_model=not args.is_api
                )

                messages.append([
                    {"role": "system", "content": REBUTTAL_POINTWISE_SYS_PROMPT},
                    {"role": "user", "content": prompt},
                ])
                ans_map.append(review_idx)

            review_idx += 1

    messages = [
        tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        for m in messages
    ]

    outputs = llm.generate(messages, SamplingParams(max_tokens=8000))
    responses = [clean_model_output(o.outputs[0].text) for o in outputs]

    # ---------- Step 6: Assemble ----------
    ptr = 0
    review_idx = 0
    for paper in paper_reviews_without_rebuttal:
        for review in paper["reviews"]:
            content = (
                "Dear reviewer:\n"
                "We're very grateful for your constructive comments. "
                "Below are responses to your suggestions and concerns.\n\n"
            )
            for point in review_points[review_idx]:
                content += f"Question: {point}\n"
                content += f"Response: {responses[ptr]}\n\n"
                ptr += 1

            content += "We hope these responses address your concerns satisfactorily."
            review.setdefault("discussion", []).append({"role": "user", "content": content})
            review_idx += 1

    for paper in paper_reviews_without_rebuttal:
        for review in paper["reviews"]:
            review.pop("relevant_paragraphs", None)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(paper_reviews_without_rebuttal, f, ensure_ascii=False, indent=2)


# =========================
# Entry
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--encoder_model", type=str, required=True)
    parser.add_argument("--planner_model", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--conf_threshold", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--is_api", action="store_true")

    args = parser.parse_args()

    data = json.load(open(args.input, "r"))
    gen_rebuttal(data, args)
