from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
import os
import json
import tiktoken
import re
import json5
from sentence_transformers import SentenceTransformer, util
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

REVIEWER_SYS_PROMPT = '''You are an experienced academic paper reviewer. You will receive a response from the authors addressing your review comments. Your task is to evaluate the response and decide whether to adjust your original score for the paper.

The scoring rubric is from 1 - 10 scale. Certain scores correspond to the following meanings:
- 1: The paper has serious flaws, lacks novelty, or is clearly unsuitable for acceptance.
- 3: The paper has significant weaknesses or insufficient contributions.
- 6: Top 25% of all submissions. The paper is slightly above the acceptance threshold, with generally solid work but some limitations.
- 8: Top 10% of all submissions. the paper has a good-quality paper with clear contributions and well-supported results.
- 10: Top 5% of all submissions. the paper makes exceptional contributions and are recommended for spotlight or oral presentation.

You should focus on the following criteria when assessing the author's response:
    - 1. Does the author's response validates their work with clear arguments and coherent logic?
    - 2. Does the author provide sufficient evidence or reasoning to support their claims?
    - 3. Is the author's response consistent with the content of the original paper?
In addition, please keep in mind that the goal of response is to CONVINCE the reviewer about the paper, instead of SUGGESTIONS for future work or ADMITTING weakness.
    - DO NOT consider suggestions, promises, or impacts for future work and revisions when evaluating the responses. Focus on this paper alone.
    - DO NOT consider tones or emotional appeals, as long as the content is professional. Focus on the logic and reasoning.
Then, you should decide whether to change your score based on the author's response:
    - You should be confident with your original review in most cases. You may increase your score only if the author provides sufficient reasoning that addresses your comments.
    - Do not increase your score based on minor corrections (e.g. typos) or promises on future revisions.
    - If the original score is low, you should be more lenient in increasing the score. If the original score is high, you should hold a higher standard.
    - In most cases, the score change will be small. Large changes, like 2 points, should be rare and well-justified.
As a conclusion, output "My final score is X" where X is your final score (an integer between 1 and 10).
'''

REVIEWER_SFT_SYS_PTOMPT = "You are a conference reviewer. Read the JSON and output ONLY a JSON object with keys 'comment' (string) and 'final_score' (integer)."

REVIEWER_COMPARE_SYS_PROMPT = '''You are an experienced academic paper reviewer. You will receive a review of an academic paper in computer science, and two responses from the authors.
Your task is to evaluate the responses and decide which response is better.

The response may address the reviewer's several comments. You should compare the responses to each comment individually.
When comparing the responses, you can refer to the following criteria:
    - 1. Does the author's response validates their work with clear arguments and coherent logic?
    - 2. Does the author provide sufficient evidence or reasoning to support their claims?
    - 3. Is the author's response consistent with the content of the original paper?
In addition, please keep in mind that the author isn't allowed to revise the paper afterwards. That is,  the goal of response is to CONVINCE the reviewer about the paper, instead of SUGGESTIONS for future work or ADMITTING weakness.
    - DO NOT consider suggestions, promises, or impacts for future work and revisions when evaluating the responses. Focus on this paper alone.
    - DO NOT consider tones or emotional appeals, as long as the content is professional. Focus on the logic and reasoning.
Please give concrete evidences while being concise. DO NOT repeat or summarize the responses' content or similarities; focus on their differences and YOUR ANALYSIS.
Output "I think response X (1 or 2) is better" or "I think two responses are similar in quality" at the end of your answer.
'''

# REVIEWER_COMPARE_SYS_PROMPT = '''You are an experienced computer science researcher. You serve as a reviewer in a top conference, and the author provided responses to your review.
# Compare the two responses. Which one is better? First, make a brief analysis, and then output your conclusion. '''


REBUTTAL_SYS_PROMPT = '''You are an experienced researcher in computer science. You have written a conference paper in the field of computer science or AI and received a review.
You need to write a rebuttal to address the reviewer's comments and convince them to increase their score.

Guidelines:
1. Be polite, concise, and professional. Make sure all responses are factual, respectful, and persuasive.
2. Address each comment point-by-point. It's recommended to format the main part of  the rebuttal as: "Question: ...Response: ...". For each point:
3. For each point, you should respond with clear reasoning, and evidence from the original paper, and your professional knowledge.
   - If the comment has misunderstood the paper or missed some content, clarify the point. If not, defend your choices and explain why this comment doesn't undermine your paper.
   - DO NOT propose suggestions or promises for future revision or future work.
4. Be confident with your paper. Try your best to explain and validate you work, and rebute the concerns raised by the reviewer.
5. Your rebuttal should be concise and no more than 1000 words. You should directly generate a passage without additional comments or thoughts.
'''

REBUTTAL_POINTWISE_SYS_PROMPT = '''You are an experienced researcher in computer science. You have written a conference paper in the field of computer science or AI and received a review.
You need to write a rebuttal to address the reviewer's comment and convince them to increase their score.

Guidelines:
1. Make sure your response is factual, respectful, and persuasive.
2. You should respond with clear reasoning, and evidence from the original paper, and your professional knowledge.
   - If the comment has misunderstood the paper or missed some content, clarify the point. If not, defend your choices and explain why this comment doesn't undermine your paper.
   - DO NOT propose suggestions or promises for future revision or future work.
3. Be confident with your paper. Try your best to explain and validate you work, and rebute the concerns raised by the reviewer.
4. Your rebuttal should be concise and no more than 200 words. You should directly generate a paragraph without additional comments or thoughts.
'''

REBUTTAL_DECOMPOSER_SYS_PROMPT = '''You are an experienced researcher in computer science. You have written a conference paper in the field of computer science or AI and received a review.
You need to analyse the reviewer's comments. Specifically, identify and list all the weakness points or confusions raised by the reviewer.
    - You may omit minor issues such as typos, but major comments should all be mentioned.
    - Preferably, extract sentences or words directly from the review. Do not oversimplify the comments.

Below is an example of the expected output format:
[
    "The paper introduced two modules, but lacks ablation study which includes only one of them.",
    "What does the author mean by PPO? Further explain will be helpful.",
    "The experimental results are only shown on 1 newly created environment."
]
'''

REBUTTAL_PERSPECTIVE_SYS_PROMPT = '''You are an experienced researcher in computer science.
You have received a review on a research paper. Your task is to propose up to 5 perspectives to address this point in the rebuttal.
    - The perspective should either show the reviewer's point wrong, or show that the work is valuable even though the review is correct. Specifically, You MUST consider the following two types of perspectives:
        - Clarification: The reviewer may have factual errors or misunderstood in the paper. For example, they may say something is missing when it's actually present in the paper, or say the methodology is wrong because of a misunderstanding.
        - Justification: Defend your choices and explain why the comment doesn't undermine your paper. For example, they may require an experiment which is unfeasible or unnecessary, or require empirical results for a theoretical paper.
    - DO NOT propose suggestions or promises for future revision or future work.
    - DO NOT mention specific locations in the paper since you won't be able to access it (e.g. "in section 3.2").
    
Below is an example of the expected output format:
Example:
Input: "The paper introduced two modules, but lacks ablation study which includes only one of them."
Output:
[
    "Clarification: we have actually included such experiment in the paper.",
    "Clarification: the two modules are dependent on each other and therefore cannot be separated.",
    "Justification: the ablation study is not necessary as each module has been individually validated in prior work."
]
'''


import re
import json5
import dirtyjson

def process_json(text: str):
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text, re.DOTALL)
    if not match:
        return {}
    json_str = match.group()
    json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)

    try:
        return dirtyjson.loads(json_str)
    except Exception as e:
        print(f"解析 JSON 失败: {e}")
        return {}



def sep_passage(text: str):
    paragraphs = [para for para in text.split('\n\n') if para.strip() != '']

    # 1. Merge paragraphs that start with "Table" with the following paragraph
    merged_paragraphs = []
    skip_next = False
    for i, para in enumerate(paragraphs):
        if skip_next:
            skip_next = False
            continue
        if para.strip().startswith('Table') and i + 1 < len(paragraphs):
            merged_paragraphs.append(para + '\n' + paragraphs[i + 1])
            skip_next = True
        else:
            merged_paragraphs.append(para)

    # 1.5 Merge formula blocks with surrounding paragraphs
    # Pattern: text -> $$formula$$ -> where clause / explanation
    formula_merged_paragraphs = []
    i = 0
    while i < len(merged_paragraphs):
        para = merged_paragraphs[i]
        stripped = para.strip()
        
        # If current paragraph is a formula block (starts and ends with $)
        if stripped.startswith('$') and stripped.endswith('$'):
            # Merge with previous paragraph (if exists and not the first one)
            if len(formula_merged_paragraphs) > 0:
                merged = formula_merged_paragraphs.pop() + '\n\n' + para
            else:
                merged = para
            
            # Check if next paragraph should be merged
            if i + 1 < len(merged_paragraphs):
                next_para = merged_paragraphs[i + 1]
                next_stripped = next_para.strip()
                
                # Determine if it's a continuation of the formula explanation
                # Conditions: starts with where/here/etc., or starts with lowercase and contains $, or another formula block
                is_continuation = (
                    re.match(r'^(where|here|and|thus|therefore|note that)\s', next_stripped, re.IGNORECASE) or
                    (next_stripped and next_stripped[0].islower() and '$' in next_stripped) or
                    (next_stripped.startswith('$') and next_stripped.endswith('$'))
                )
                
                if is_continuation:
                    merged += '\n\n' + next_para
                    i += 1  # Skip the next paragraph
                
            formula_merged_paragraphs.append(merged)
        else:
            formula_merged_paragraphs.append(para)
        
        i += 1

    # 2. Remove paragraphs that are pure image markdown links: ![...](...)
    filtered_paragraphs = []
    link_pattern = re.compile(r'^!\[.*?\]\(.*?\)$')
    for para in formula_merged_paragraphs:
        if not link_pattern.match(para.strip()):
            filtered_paragraphs.append(para)

    # 3. Remove all paragraphs before "## Abstract"
    start_idx = 0
    for i, para in enumerate(filtered_paragraphs):
        if para.strip().lower().startswith('## abstract'):
            start_idx = i
            break
    filtered_paragraphs = filtered_paragraphs[start_idx:]

    # 4. Handle section headings
    # Paragraphs starting with two or more # are headings; keep them but move to prefix of following content
    section_stack = []
    result_paragraphs = []
    for para in filtered_paragraphs:
        section_match = re.match(r'^(#{2,})\s*(.*)', para.strip())
        if section_match:
            # It's a heading → update section stack
            level = len(section_match.group(1))
            title = section_match.group(0)
            # Trim stack to maintain correct hierarchy
            while len(section_stack) >= level - 1:
                section_stack.pop()
            section_stack.append(title)
        else:
            # Not a heading → prepend all current section titles
            prefix = ''
            if section_stack:
                prefix = '\n'.join(section_stack) + '\n'
            result_paragraphs.append(prefix + para)

    # 5. Remove paragraphs shorter than k=20 characters (content only)
    k = 20
    final_paragraphs = []
    for para in result_paragraphs:
        # Separate section prefix and actual content
        lines = para.split('\n')
        section_lines = []
        content_lines = []
        for line in lines:
            if re.match(r'^#{2,}\s*', line.strip()):
                section_lines.append(line)
            else:
                content_lines.append(line)
        
        content = '\n'.join(content_lines)
        if len(content.strip()) >= k:
            final_paragraphs.append(para)
    
    return final_paragraphs


def merge_passages(paragraphs):
    """
    Merge passages that share the same section headings
    
    Args:
        paragraphs: list of str - subset of paragraphs from sep_passage (should be in order)
        
    Returns:
        str - merged text string
        
    How it works:
        - Extracts all heading prefixes for each paragraph (lines starting with ## or more #)
        - Ensures each heading (at any level) appears only once
        - Contents under different subheadings are joined with "..." when there are multiple segments
    
    Example:
        Input: ["## Method\n### XXX\ncontent1", "## Method\n### XXY\ncontent2"]
        Output: "## Method\n### XXX\ncontent1\n...\n### XXY\ncontent2"
    """
    if not paragraphs:
        return ""
    
    # Build hierarchical tree structure
    class Node:
        def __init__(self, title=''):
            self.title = title
            self.children = {}  # title -> Node
            self.contents = []  # content stored at this node
            self.first_seen = None  # first appearance order
    
    root = Node()
    global_order = [0]  # mutable counter via list
    
    for para in paragraphs:
        lines = para.split('\n')
        
        # Separate headings and content
        section_lines = []
        content_lines = []
        
        for line in lines:
            if re.match(r'^#{2,}\s*', line.strip()):
                section_lines.append(line)
            else:
                content_lines.append(line)
        
        content = '\n'.join(content_lines).strip()
        
        # Traverse/create path in tree according to headings
        current = root
        for title in section_lines:
            if title not in current.children:
                current.children[title] = Node(title)
                current.children[title].first_seen = global_order[0]
                global_order[0] += 1
            current = current.children[title]
        
        # Add content to leaf node
        if content:
            current.contents.append(content)
    
    # Recursively build merged text
    def build_text(node, depth=0):
        """
        Build text for current node and all descendants
        Returns complete text string
        """
        parts = []
        
        # Add contents of current node
        if node.contents:
            if len(node.contents) > 1:
                parts.append('\n...\n'.join(node.contents))
            else:
                parts.append(node.contents[0])
        
        # Process children in first-seen order
        if node.children:
            child_items = sorted(node.children.items(), key=lambda x: x[1].first_seen)
            
            for title, child_node in child_items:
                # Add child heading
                parts.append(title)
                
                # Recursively get child content
                child_text = build_text(child_node, depth + 1)
                if child_text:
                    parts.append(child_text)
        
        return '\n'.join(parts)
    
    # Build final result from all top-level sections
    result = []
    if root.children:
        top_level_items = sorted(root.children.items(), key=lambda x: x[1].first_seen)
        
        for title, node in top_level_items:
            text_parts = [title]
            child_text = build_text(node)
            if child_text:
                text_parts.append(child_text)
            
            result.append('\n'.join(text_parts))
    
    # Join all top-level sections with double newlines
    return '\n\n'.join(result)


def encode_papers(path, model, output_dir="./", batch_size=32):
    """
    Split papers into passages and encode them into embeddings
    
    Args:
        path: str - path to JSON file containing paper texts
        model: SentenceTransformer or str - embedding model instance or model name
        output_dir: str - directory to save processed files
        batch_size: int - batch size for encoding (default 32, larger = better GPU utilization)
    
    Returns:
        tuple: (paper_paragraphs, paper_embeddings)
            - paper_paragraphs: dict {paper_id: [list of passage strings]}
            - paper_embeddings: dict {paper_id: torch.Tensor (n_passages, embed_dim)}
              Note: all tensors are on CPU
    """
    # Load model if a string path/name is provided
    if isinstance(model, str):
        model = SentenceTransformer(model)
        
    print(f"Model loaded on device: {model.device}")
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Detected {num_gpus} GPUs, will use multi-process encoding")
    
    paper_archive = json.load(open(path, "r"))
    paper_paragraphs = {}
    
    # Step 1: Split all papers into passages
    print("Step 1/2: Splitting papers into paragraphs...")
    for paper_id, text in tqdm(list(paper_archive.items()), desc="Splitting paragraphs"):
        passages = sep_passage(text)
        paper_paragraphs[paper_id] = passages
    
    # Step 2: Batch encode all passages
    print(f"\nStep 2/2: Encoding all paragraphs (batch_size={batch_size})...")
    
    # Collect all passages and their metadata
    all_passages = []
    passage_to_paper = []      # which paper each passage belongs to
    passage_indices = []       # index within that paper
    
    for paper_id, passages in paper_paragraphs.items():
        for idx, passage in enumerate(passages):
            all_passages.append(passage)
            passage_to_paper.append(paper_id)
            passage_indices.append(idx)

    paper_embeddings = {}
    if len(all_passages) > 0:
        print(f"Encoding {len(all_passages)} paragraphs from {len(paper_paragraphs)} papers...")
        
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Using multi-process pool with {num_gpus} GPUs (progress bar not available in multi-process mode)")
            pool = model.start_multi_process_pool()
            all_embeddings = model.encode(
                all_passages,
                pool=pool,
                batch_size=batch_size,
                show_progress_bar=True
            )
            model.stop_multi_process_pool(pool)
            all_embeddings = torch.from_numpy(all_embeddings)
            print(f"✓ Completed encoding {len(all_passages)} paragraphs")
        else:
            all_embeddings = model.encode(
                all_passages, 
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=True
            )
        
        # Move to CPU
        all_embeddings = all_embeddings.cpu()
        
        # Distribute embeddings back to papers
        for paper_id in paper_paragraphs.keys():
            paper_embeddings[paper_id] = []
        
        for i, (paper_id, idx) in enumerate(zip(passage_to_paper, passage_indices)):
            paper_embeddings[paper_id].append(all_embeddings[i])
        
        for paper_id in paper_embeddings.keys():
            if len(paper_embeddings[paper_id]) > 0:
                paper_embeddings[paper_id] = torch.stack(paper_embeddings[paper_id])
            else:
                paper_embeddings[paper_id] = torch.tensor([])
    else:
        for paper_id in paper_paragraphs.keys():
            paper_embeddings[paper_id] = torch.tensor([])
    
    # Save processed paragraphs
    paragraphs_output_path = os.path.join(output_dir, "processed_paper_paragraphs.json")
    with open(paragraphs_output_path, "w", encoding="utf-8") as f:
        json.dump(paper_paragraphs, f, ensure_ascii=False, indent=2)
    print(f"\nSaved paper paragraphs to {paragraphs_output_path}")
    
    # Save embeddings
    embeddings_output_path = os.path.join(output_dir, "processed_paper_embeddings.pt")
    torch.save(paper_embeddings, embeddings_output_path)
    print(f"Saved paper embeddings to {embeddings_output_path}")
    
    return paper_paragraphs, paper_embeddings


def retrieve(queries, passages, model, passage_embeddings=None, top_k=10):
    """
    For each query, return indices and similarity scores of the top-k most similar passages
    
    Args:
        queries: str or list of str - query text(s)
        passages: list of str - candidate passage texts
        model: SentenceTransformer or str - embedding model
        passage_embeddings: optional pre-computed embeddings (saves recomputation)
        top_k: int - number of top results to return
    
    Returns:
        If single query: (indices, scores)
        If list of queries: [(indices, scores), ...]
        indices: array of shape (top_k,), scores: corresponding cosine similarities
    """
    # Load model if string is provided
    if isinstance(model, str):
        model = SentenceTransformer(model)
    
    # Normalize input to list
    single_query = isinstance(queries, str)
    if single_query:
        queries = [queries]
    
    # Encode queries and passages
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    if passage_embeddings is None:
        passage_embeddings = model.encode(passages, convert_to_tensor=True)
    
    # Compute cosine similarity matrix (queries × passages)
    similarities = util.cos_sim(query_embeddings, passage_embeddings)
    
    # Get top-k for each query
    results = []
    for i in range(len(queries)):
        sim_scores = similarities[i].cpu().numpy()
        
        top_k_actual = min(top_k, len(passages))
        top_indices = np.argsort(sim_scores)[::-1][:top_k_actual]
        top_scores = sim_scores[top_indices]
        results.append((top_indices, top_scores))
    
    return results[0] if single_query else results


def api_batch_inference(requests, sampling_params, model="gpt-4o", n_threads=8, progress=False):
    """
    Batch inference using multiple API keys / clients in parallel
    
    Args:
        requests: list of message lists (each is conversation history for one call)
        sampling_params: dict - generation parameters (temperature, max_tokens, etc.)
        model: str - model name ("gpt-4o", "gpt-4o-mini", or nvidia-hosted model)
        n_threads: int - maximum concurrent requests
        progress: bool - show tqdm progress bar
    
    Returns:
        list of str - generated responses (one per request)
    """
    params = sampling_params
    
    # Initialize API clients
    clients = []
    if model in ("gpt-4o", "gpt-4o-mini"):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        clients.append(OpenAI(
            api_key=openai_api_key,
            max_retries=5,
            timeout=60
        ))
    else:
        # Load NVIDIA API keys
        api_keys_data = json.load(open("api_keys.json", "r"))
        
        if isinstance(api_keys_data.get("nvidia_keys"), str):
            nvidia_keys = [api_keys_data["nvidia_keys"]]
        elif isinstance(api_keys_data.get("nvidia_keys"), list):
            nvidia_keys = api_keys_data["nvidia_keys"]
        else:
            raise ValueError("api_keys.json must contain 'nvidia_key' or 'nvidia_keys'")
        
        for key in nvidia_keys:
            clients.append(OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=key,
                max_retries=3,
                timeout=300
            ))
    
    class EmptyResult:
        def __init__(self):
            self.choices = [{"message": {"content": "This is an empty response due to an API error."}}]

    def get_completion(args):
        request, client_idx = args
        assert isinstance(request, list) and all(isinstance(turn, dict) for turn in request), \
            "Request format error: expected list of dicts"
        
        client = clients[client_idx % len(clients)]
        
        try:
            result = client.chat.completions.create(
                model=model,
                messages=request,
                **params
            )
            return result
        except Exception as e:
            print(f"API error: {e}")
            return EmptyResult()

    # Assign each request to a client (round-robin)
    requests_with_idx = [(req, idx) for idx, req in enumerate(requests)]
    
    with ThreadPoolExecutor(max_workers=min(len(requests), n_threads)) as executor:
        if progress:
            results = list(tqdm(
                executor.map(get_completion, requests_with_idx),
                total=len(requests),
                desc=f"Inference (Parallel, Model: {model}, {len(clients)} API Key(s))"
            ))
        else:
            results = list(executor.map(get_completion, requests_with_idx))

    # Extract content from responses
    processed_results = []
    for r in results:
        try:
            processed_results.append(r.choices[0].message.content)
        except:
            processed_results.append("This is an empty response due to an API error.")
    
    return processed_results