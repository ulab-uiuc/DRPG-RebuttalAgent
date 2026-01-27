import json
import torch
import sys
import random
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common_tools import process_json

'''
Planner data format
Input:
    Candidate Perspectives
    Paper passages
Output:
    The ground truth perspective
'''


class PlannerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        para_file="data/processed_paper_paragraphs.json",
        rebuttal_file="data/revised_score/test_real_real.json",
        retrieval_file="data/test.json",
        persp_file="data/perspective/llama-3.3-70b-instruct/test.json",
        seed=42,
        cutoff=None,
        randomize=False,
        top_k=15
    ):
        random.seed(seed)

        para_file = json.load(open(para_file, 'r', encoding='utf-8'))
        rebuttal_file = json.load(open(rebuttal_file, 'r', encoding='utf-8'))["reviews"]
        retrieval_file = json.load(open(retrieval_file, 'r', encoding='utf-8'))
        persp_file = json.load(open(persp_file, 'r', encoding='utf-8'))
        self.data = []

        cnt = 0
        for paper_id in range(len(retrieval_file)):
            paragraphs = para_file[retrieval_file[paper_id]['paper_id']]
            for review_id in range(len(retrieval_file[paper_id]['reviews'])):
                good_rebuttal = (
                    rebuttal_file[paper_id]['reviews'][review_id]["final_score"]
                    > rebuttal_file[paper_id]['reviews'][review_id]["initial_score"]
                )

                # Check whether rebuttal_summary is a list of dictionaries
                rebuttal_summary = rebuttal_file[paper_id]['reviews'][review_id].get(
                    'rebuttal_summary', []
                )
                if not all(isinstance(e, dict) for e in rebuttal_summary):
                    cnt += len(
                        retrieval_file[paper_id]['reviews'][review_id]['decomposed_content']
                    )
                    continue

                complete_rebuttal = (
                    len(rebuttal_summary)
                    == len(
                        retrieval_file[paper_id]['reviews'][review_id]['decomposed_content']
                    )
                )
                if not good_rebuttal or not complete_rebuttal:
                    cnt += len(
                        retrieval_file[paper_id]['reviews'][review_id]['decomposed_content']
                    )
                    continue

                for point_id in range(
                    len(retrieval_file[paper_id]['reviews'][review_id]['decomposed_content'])
                ):
                    item = {}

                    item["paper_id"] = retrieval_file[paper_id]['paper_id']
                    item['perspectives'] = process_json(persp_file[cnt])
                    if item['perspectives'] == {}:
                        item['perspectives'] = []

                    item['perspectives'] = [
                        str(persp)
                        .replace("Clarification: ", "")
                        .replace("Justification: ", "")
                        for persp in item['perspectives']
                    ]

                    gt = rebuttal_summary[point_id]["perspective"]
                    gt = gt.replace("Clarification: ", "").replace("Justification: ", "")

                    item['ground_truth_pos'] = random.randint(
                        0, len(item['perspectives']) # randomize the position of ground truth
                    )
                    item['perspectives'].insert(item['ground_truth_pos'], gt)

                    if top_k == -1:
                        item['passages'] = retrieval_file[paper_id]['reviews'][review_id][
                            "relevant_paragraphs"
                        ][point_id]
                    else:
                        item['passages'] = retrieval_file[paper_id]['reviews'][review_id][
                            "relevant_paragraphs"
                        ][point_id][:max(top_k, 1)]

                    item['passages'] = sorted(set(item['passages']))
                    item['passages_content'] = [
                        paragraphs[pid] for pid in item['passages']
                    ]

                    addressed = rebuttal_summary[point_id]["addressed"] == "Yes"
                    valid_persp = len(gt) > 20 and len(item['perspectives']) > 1

                    if addressed and valid_persp:
                        self.data.append(item)

                    cnt += 1

        if cutoff is not None:
            self.data = self.data[:cutoff]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item