from common_tools import *
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from planner.train_planner import SoftMatchingPlanner


paper_archive = json.load(open("data/processed_papers.json", "r"))
paper_paragraphs_archive = json.load(open("data/processed_paper_paragraphs.json", "r"))

llm = None
tokenizer = None
MAX_TOKENS = 27000

def clean_model_output(text):
    '''
    Clean model output by removing think tags and similar artifacts
    
    Args:
        text: The raw model output text
    
    Returns:
        str: Cleaned text with think tags removed
    '''
    import re
    # Remove <think>...</think> tags and their content (non-greedy match)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove any leading/trailing whitespace
    text = text.strip()
    return text

def build_rebuttal_prompt_content(mode, paper_content, review_point, initial_score=None, perspectives=None, conf_valid=True, sft_model=False):
    '''
    Build the user prompt content for rebuttal generation
    
    Args:
        mode: Generation mode ("0.0", "0.1", "1.0", "2.0", "2.c", "2.j", "2.1", "jiu-jitsu")
        paper_content: Full paper content (mode 0.0, 0.1) or retrieved paper excerpt (mode 1.0, 2.x)
        review_point: The review content or decomposed point
        initial_score: Initial review score (only used in mode 0.0)
        perspectives: List of perspectives generated for the review point
    
    Returns:
        str: The formatted user prompt content
    '''
    content = ""
    
    if mode == "0.0":
        # Mode 0.0: Full paper + full review
        # paper_content = paper_content[:3000]
        content += "$$$$ [The Paper]:\n\n" + paper_content + "\n\n"
        content += "$$$$ [The Review]:\n\n" + review_point + "\n\n"
        if initial_score is not None:
            content += f"$$$$ [Score]: {str(initial_score)}"
    
    elif mode == "0.1":
        # Mode 0.1: Full paper + decomposed point
        paper_label = "$$$$ [Relevant Paragraphs]:\n\n" if sft_model else "$$$$ [The Paper]:\n\n"
        content += paper_label + paper_content + "\n\n"
        content += "$$$$ [The Review]:\n\n" + review_point
    
    elif mode in ["1.0", "2.0", "2.c", "2.j", "2.1"]:
        # Modes 1.0 and 2.x: Retrieved paper excerpt + decomposed point
        paper_label = "$$$$ [Relevant Paragraphs]:\n\n" if sft_model else "$$$$ [Paper Excerpt]:\n\n"
        content += paper_label + paper_content + "\n\n"
        content += "$$$$ [The Review]:\n\n" + review_point
        
        # Add perspective hints based on mode
        if perspectives and len(perspectives) > 0:
            if mode == "2.0":
                # Mode 2.0: Show all perspectives as suggestions
                hint_label = "$$$$ [Perspective]:\n" if sft_model else "$$$$ [Hint]\n"
                content += "\n" + hint_label
                if not sft_model:
                    content += "The following perspectives might be helpful for you to construct your response. However, you shall focus on the paper content and make concrete responses. The points are just suggestions, and you are to discern which point can be validated by the paper.\n"
                for perspective in perspectives:
                    pers = str(perspective).replace('Justification: ', '').replace('Clarification: ', '')
                    content += f"\t- {pers}\n"
                if sft_model:
                    content += "You may consider constructing your response with the following perspective. However, you shall still focus on the paper content and make concrete responses.\n"
            
            elif mode in ["2.c", "2.j"]:
                # Modes 2.c, 2.j: Show single selected perspective
                hint_label = "$$$$ [Perspective]:\n" if sft_model else "$$$$ [Hint]\n\n"
                content += "\n" + hint_label
                selected_pers = str(perspectives[0]).replace('Justification: ', '').replace('Clarification: ', '')
                if sft_model:
                    content += f"{selected_pers}\n\n"
                    content += "You may consider constructing your response with the following perspective. However, you shall still focus on the paper content and make concrete responses.\n"
                else:
                    content += "You may consider constructing your response with the following perspective. However, you shall still focus on the paper content and make concrete responses.\n"
                    content += f"Perspective: {selected_pers}\n\n"
            
            elif mode == "2.1":
                # Mode 2.1: Show selected perspective only if confidence meets threshold
                if conf_valid:
                    hint_label = "$$$$ [Perspective]:\n\n" if sft_model else "$$$$ [Hint]\n\n"
                    content += "\n" + hint_label
                    selected_pers = str(perspectives[0]).replace('Justification: ', '').replace('Clarification: ', '')
                    if sft_model:
                        content += f"{selected_pers}\n\n"
                        content += "You may consider constructing your response with the provided perspective. However, you shall still focus on the paper content and make concrete responses.\n"
                    else:
                        content += "You may consider constructing your response with the following perspective. However, you shall still focus on the paper content and make concrete responses.\n"
                        content += f"Perspective: {selected_pers}\n\n"
                        
    if tokenizer:
        tokens = tokenizer.encode(content, add_special_tokens=False)
        if len(tokens) > MAX_TOKENS:
            print(len(tokens))
        tokens = tokens[:MAX_TOKENS]
        content = tokenizer.decode(tokens, skip_special_tokens=True)
                           
    return content


def gen_rebuttal(paper_reviews_without_rebuttal, args):
    '''
    paper_reviews_without_rebuttal: List of dicts, each dict contains 'paper_id' and 'reviews' (list of dicts with 'review_content' and 'initial_score')
    args: command line arguments
    args.mode:
        - "0.0": 输入整篇论文，整段review，输出rebuttal (Baseline)
        - "0.1": 输入整篇论文，将review拆分成子问题，输出rebuttal
        - "1.0": 将review拆分成子问题，对每个子问题retrieve relevant paragraphs，之后输出rebuttal
        - "2.0": 将review拆分成子问题，对每个子问题retrieve relevant paragraphs并生成反驳论点
        - "2.1": 基于2.0，使用planner模型从生成的perspectives中选择最好的一个
        
    Structure:
        Step 1: Pre-processing
        Step 2: Perspective Generation
        Step 3: Rebuttal Generation
    '''
    
    # Load planner model if mode is 2.1
    planner_model = None
    if args.mode == "2.1":
        if not args.planner_model:
            raise ValueError("Mode 2.1 requires --planner_model argument")
        
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(args.planner_model, map_location=device)
        config = checkpoint.get('config', {})
        
        # Initialize planner model
        planner_model = SoftMatchingPlanner(
            pretrained_encoder=config.get('encoder', 'sentence-transformers/all-MiniLM-L6-v2'),
            mlp_hidden=config.get('mlp_hidden', [512]),
            freeze_encoder=config.get('freeze_encoder', False)
        ).to(device)
        
        # Load model weights
        planner_model.load_state_dict(checkpoint['model_state_dict'])
        planner_model.eval()
        print(f"Loaded planner model from {args.planner_model}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Test Accuracy: {checkpoint.get('test_accuracy', 'unknown')}")
    
    global tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except:
        pass
    
    # Initialize llm and tokenizer as None, will be initialized on first use
    def ensure_llm_initialized(model_path=None):
        """Ensure LLM is initialized. If model_path is provided and different from current, reinitialize."""
        global llm, tokenizer
        if model_path is None:
            model_path = args.model
        
        if llm is None or (model_path != args.model and model_path != getattr(llm, 'model', None)):
            if llm is not None:
                del llm
            llm = LLM(
                model=model_path,
                dtype='bfloat16',
                tensor_parallel_size=args.n_gpus,
                gpu_memory_utilization=0.85,
                disable_sliding_window=False,
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.model)
            except:
                pass
        return llm, tokenizer
    
    ''' Step 1'''
    if args.mode in ["0.1", "1.0", "2.0", "2.c", "2.j", "2.1"]:
        # Check if reviews already have decomposed_content
        first_review = paper_reviews_without_rebuttal[0]["reviews"][0]
        has_decomposed_content = "decomposed_content" in first_review and isinstance(first_review["decomposed_content"], list) and len(first_review["decomposed_content"]) > 0
        
        '''Substep 1.1: Extract points from reviews'''
        if has_decomposed_content:
            print("Skipping Decomposition Step: All reviews already have decomposed_content.")
            review_points = []
            for paper in paper_reviews_without_rebuttal:
                for review in paper["reviews"]:
                    review_points.append(review["decomposed_content"])
        else:
            all_decompose_messages = []
            original_reviews = []  # Store original review content for fallback
            for paper in paper_reviews_without_rebuttal:
                for review in paper["reviews"]:
                    all_decompose_messages.append([
                        {"role": "system", "content": REBUTTAL_DECOMPOSER_SYS_PROMPT},
                        {"role": "user", "content": review["review_content"]},
                    ])
                    original_reviews.append(review["review_content"])
            
            if args.is_api:
                decompose_responses = api_batch_inference(
                    all_decompose_messages, 
                    sampling_params={"temperature": args.temperature, "top_p": args.top_p, "max_tokens": 5000}, 
                    model=args.model, 
                    n_threads=12,
                    progress=True
                )
            else:
                llm, tokenizer = ensure_llm_initialized()
                all_decompose_messages = [tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) for messages in all_decompose_messages]
                params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=10000)
                decompose_responses = [output.outputs[0].text.strip() for output in llm.generate(all_decompose_messages, params)]
            
            # Process decomposed points
            review_points = []  # List of lists: each review has a list of points
            for id_review, res in enumerate(decompose_responses):
                points = process_json(res)
                # If process_json fails or returns non-list, use original review as single point
                if not isinstance(points, list) or len(points) == 0:
                    points = [original_reviews[id_review]]
                review_points.append(points)
        
        '''Substep 1.2: Retrieve relevant paragraphs for each decomposed point (preprocessing)'''
        # For retrieval-based modes, perform passage retrieval now and store
        if args.mode in ["1.0", "2.0", "2.c", "2.j", "2.1"]:
            # If any review already contains relevant_paragraphs, skip global retrieval
            first_review_check = paper_reviews_without_rebuttal[0]["reviews"][0]
            has_relevant_paragraphs_global = "relevant_paragraphs" in first_review_check and isinstance(first_review_check["relevant_paragraphs"], list)
            if has_relevant_paragraphs_global:
                print("Skipping global Retrieval Step: All reviews already have relevant_paragraphs.")
            else:
                if args.encoder_model == "":
                    # For mode 2.1 planner path, encoder_model is still required for retrieval unless planner already handled it
                    raise AssertionError("Please provide --encoder_model for retrieval (required in preprocessing).")
                from sentence_transformers import SentenceTransformer
                encoder_model = SentenceTransformer(args.encoder_model)

                id_review_for_retrieval = 0
                for paper in paper_reviews_without_rebuttal:
                    paper_content = paper_archive.get(paper["paper_id"], "No content found.")
                    paper_paragraphs = paper_paragraphs_archive.get(paper["paper_id"], sep_passage(paper_content))

                    # Encode passages for this paper
                    doc_embeddings = encoder_model.encode(paper_paragraphs, convert_to_tensor=True)

                    # Collect all queries (points) for this paper in order
                    queries = []
                    review_point_counts = []  # number of points for each review in this paper
                    for review in paper["reviews"]:
                        points = review_points[id_review_for_retrieval]
                        review_point_counts.append(len(points))
                        for point in points:
                            queries.append(point)
                        id_review_for_retrieval += 1

                    if len(queries) > 0:
                        retrieved_results = retrieve(queries, paper_paragraphs, encoder_model, doc_embeddings, top_k=args.top_k)
                        all_retrieved_indices = [sorted(indices) for indices, scores in retrieved_results]
                    else:
                        all_retrieved_indices = []

                    # Assign retrieved indices back to reviews (aligned by point counts)
                    idx_ptr = 0
                    for r_idx, review in enumerate(paper["reviews"]):
                        review["relevant_paragraphs"] = []
                        for _ in range(review_point_counts[r_idx]):
                            review["relevant_paragraphs"].append(all_retrieved_indices[idx_ptr])
                            idx_ptr += 1
        
    '''Step 2'''
    review_summaries_processed = []
    if args.mode in ["2.0", "2.c", "2.j", "2.1"]:
        
        '''Substep 2.1: Generate perspectives for each point'''
        # Determine cache file path
        import os
        model_name = args.perspective_model.split('/')[-1]
        data_split = "train" if "train" in args.input else "test"
        cache_dir = f"data/perspective/{model_name}"
        cache_file = f"{cache_dir}/{data_split}.json"
        
        # Check if cache exists
        if os.path.exists(cache_file):
            print(f"Loading cached perspectives from {cache_file}")
            perspective_responses = json.load(open(cache_file, "r"))
        else:
            # Collect all points for perspective generation
            all_perspective_messages = []
            point_indices = []  # Track which review each point belongs to
            
            if data_split == "train":
                # Calculate has_discussion for each review
                paper_reviews_with_gt_rebuttal = json.load(open(f"data/revised_score/{data_split}_real_real.json", "r"))["reviews"]
                has_discussion = [bool(review.get("discussion")) for paper in paper_reviews_with_gt_rebuttal for review in paper["reviews"]]

                for id_review, points in enumerate(review_points):
                    if has_discussion[id_review]:
                        for point in points:
                            all_perspective_messages.append([
                                {"role": "system", "content": REBUTTAL_PERSPECTIVE_SYS_PROMPT},
                                {"role": "user", "content": point},
                            ])
                            point_indices.append(id_review)
            else:
                for id_review, points in enumerate(review_points):
                    for point in points:
                        all_perspective_messages.append([
                            {"role": "system", "content": REBUTTAL_PERSPECTIVE_SYS_PROMPT},
                            {"role": "user", "content": point},
                        ])
                        point_indices.append(id_review)
                            
                
            # Generate perspectives
            if args.is_perspective_api:
                # print(all_perspective_messages[0][0])
                # print(all_perspective_messages[0][1])
                # exit(0)
                perspective_responses = api_batch_inference(
                    all_perspective_messages,
                    sampling_params={"temperature": args.temperature, "top_p": args.top_p, "max_tokens": 5000},
                    model=args.perspective_model,
                    n_threads=12,
                    progress=True
                )
            else:
                llm, tokenizer = ensure_llm_initialized(args.perspective_model)
                
                all_perspective_messages = [tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) for messages in all_perspective_messages]
                params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=5000)
                perspective_responses = [output.outputs[0].text.strip() for output in llm.generate(all_perspective_messages, params)]
                
                # Restore to main model if different
                if args.model != args.perspective_model:
                    llm, tokenizer = ensure_llm_initialized(args.model)
            
            # If training set, need to fill in empty responses for reviews without discussion
            if data_split == "train":
                full_perspective_responses = []
                perspective_idx = 0
                for id_review, points in enumerate(review_points):
                    if has_discussion[id_review]:
                        # Use generated perspectives
                        for _ in points:
                            full_perspective_responses.append(perspective_responses[perspective_idx])
                            perspective_idx += 1
                    else:
                        # Add empty list for each point
                        for _ in points:
                            full_perspective_responses.append("[]")
                perspective_responses = full_perspective_responses
            
            # Save perspectives to cache
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(perspective_responses, f, ensure_ascii=False, indent=2)
            print(f"Saved perspectives to {cache_file}")
            
        if args.preprocess_only:
            print("Preprocessing completed. Exiting as preprocess_only=True.")
            return

        # Combine points with perspectives
        perspective_idx = 0
        for id_review, points in enumerate(review_points):
            review_summary = []
            for point in points:
                perspectives = process_json(perspective_responses[perspective_idx])
                if not isinstance(perspectives, list):
                    perspectives = []
                element = {
                    "point": point,
                    "perspectives": perspectives
                }
                # For mode 2.0, mark all perspectives as selected (will be used in prompt)
                if args.mode == "2.0" and len(perspectives) > 0:
                    element["selected_perspectives"] = [str(p).replace('Justification: ', '').replace('Clarification: ', '') for p in perspectives]
                review_summary.append(element)
                perspective_idx += 1
            review_summaries_processed.append(review_summary)
        
        '''Substep 2.2: Filter perspectives based on mode (2.c, 2.j, 2.1)'''
        if args.mode in ["2.c", "2.j", "2.1"]:
            if args.mode == "2.c":
                print("Filtering perspectives: selecting Clarification type...")
                # Mode 2.c: Select clarification perspective
                for review_summary in review_summaries_processed:
                    for element in review_summary:
                        if len(element["perspectives"]) > 0:
                            # Find first Clarification perspective, or use first one as default
                            clarification_pers = element["perspectives"][0]
                            selected_idx = 0
                            for idx, perspective in enumerate(element["perspectives"]):
                                pers_str = str(perspective)
                                if pers_str.startswith("Clarification:"):
                                    clarification_pers = perspective
                                    selected_idx = idx
                                    break
                            # Store selected perspective info
                            element["selected_perspective"] = str(clarification_pers).replace('Justification: ', '').replace('Clarification: ', '')
                            element["selected_perspective_idx"] = selected_idx
            
            elif args.mode == "2.j":
                print("Filtering perspectives: selecting Justification type...")
                # Mode 2.j: Select justification perspective
                for review_summary in review_summaries_processed:
                    for element in review_summary:
                        if len(element["perspectives"]) > 0:
                            # Find first Justification perspective, or use first one as default
                            justification_pers = element["perspectives"][0]
                            selected_idx = 0
                            for idx, perspective in enumerate(element["perspectives"]):
                                pers_str = str(perspective)
                                if pers_str.startswith("Justification:"):
                                    justification_pers = perspective
                                    selected_idx = idx
                                    break
                            # Store selected perspective info
                            element["selected_perspective"] = str(justification_pers).replace('Justification: ', '').replace('Clarification: ', '')
                            element["selected_perspective_idx"] = selected_idx
            
            elif args.mode == "2.1":
                print("Filtering perspectives: using planner model to select best...")
                # Mode 2.1: Use planner model to select best perspective
                
                # Prepare retrieval results first to get passages
                id_review = 0
                for paper in paper_reviews_without_rebuttal:
                    paper_content = paper_archive.get(paper["paper_id"], "No content found.")
                    paper_paragraphs = paper_paragraphs_archive.get(paper["paper_id"], sep_passage(paper_content))
                    
                    # Check if reviews already have relevant_paragraphs
                    has_relevant_paragraphs = "relevant_paragraphs" in paper["reviews"][0] and isinstance(paper["reviews"][0]["relevant_paragraphs"], list)
                    
                    if not has_relevant_paragraphs:
                        # Need to do retrieval
                        if args.encoder_model == "":
                            raise ValueError("Mode 2.1 requires --encoder_model for retrieval")
                        encoder_model = SentenceTransformer(args.encoder_model)
                        doc_embeddings = encoder_model.encode(paper_paragraphs, convert_to_tensor=True)
                        
                        queries = [element["point"] for i, review in enumerate(paper["reviews"]) for element in review_summaries_processed[id_review + i]]
                        retrieved_results = retrieve(queries, paper_paragraphs, encoder_model, doc_embeddings, top_k=args.top_k)
                        all_retrieved_indices = [sorted(indices) for indices, scores in retrieved_results]
                        
                        # Store in reviews
                        id_element = 0
                        for review in paper["reviews"]:
                            review["relevant_paragraphs"] = []
                            for element_idx, element in enumerate(review_summaries_processed[id_review]):
                                review["relevant_paragraphs"].append(all_retrieved_indices[id_element])
                                id_element += 1
                            id_review += 1
                    else:
                        id_review += len(paper["reviews"])
                
                # Now use planner to select best perspective for each point
                # Collect all planner inputs and their positions
                planner_inputs = []
                input_positions = []  # Store (id_review, element_idx) for each input
                
                id_review = 0
                for paper in paper_reviews_without_rebuttal:
                    paper_paragraphs = paper_paragraphs_archive.get(paper["paper_id"], sep_passage(paper_archive.get(paper["paper_id"], "No content found.")))
                    
                    for review in paper["reviews"]:
                        for element_idx, element in enumerate(review_summaries_processed[id_review]):
                            if len(element["perspectives"]) > 0:
                                # Prepare planner input
                                retrieved_indices = review["relevant_paragraphs"][element_idx][:args.top_k]
                                planner_input = {
                                    "paper_id": paper["paper_id"],
                                    "passages": sorted(set(retrieved_indices)),
                                    "passages_content": [paper_paragraphs[idx] for idx in sorted(set(retrieved_indices))],
                                    "perspectives": [str(p).replace('Justification: ', '').replace('Clarification: ', '') for p in element["perspectives"]]
                                }
                                planner_inputs.append(planner_input)
                                input_positions.append((id_review, element_idx))
                        
                        id_review += 1
                
                # Process planner inputs in batches
                batch_size = 64
                all_best_results = []
                
                print(f"Processing {len(planner_inputs)} planner inputs in batches of {batch_size}...")
                from tqdm import tqdm
                for i in tqdm(range(0, len(planner_inputs), batch_size), desc="Planner batches"):
                    batch_inputs = planner_inputs[i:i + batch_size]
                    # Get both best indices and confidence scores
                    batch_results = planner_model.predict_best_view(batch_inputs, return_confidence=True)
                    all_best_results.extend(batch_results)
                
                # Update perspectives with selected indices and confidence
                for idx, (id_review, element_idx) in enumerate(input_positions):
                    best_idx, confidence = all_best_results[idx]
                    element = review_summaries_processed[id_review][element_idx]
                    # Store selected perspective info
                    element["selected_perspective"] = str(element["perspectives"][best_idx]).replace('Justification: ', '').replace('Clarification: ', '')
                    element["selected_perspective_idx"] = best_idx
                    element["confidence"] = confidence
            
            else:
                raise NotImplementedError("Unknown filtering mode.")
        
    else:
        # For mode 0.1 and 1.0, no perspectives needed
        for points in review_points:
            review_summary = [{"point": point, "perspectives": []} for point in points]
            review_summaries_processed.append(review_summary)    
        
        
    ''' Step 3'''     
    '''Step 3.1: Generate the rebuttal'''
    all_messages = []
    if args.mode == "0.0":
        for paper in paper_reviews_without_rebuttal:
            paper_content = paper_archive.get(paper["paper_id"], "No content found.")
            for review in paper["reviews"]:
                content = build_rebuttal_prompt_content(
                    mode=args.mode,
                    paper_content=paper_content,
                    review_point=review["review_content"],
                    initial_score=review["initial_score"],
                    sft_model = not args.is_api,
                )
                messages = [
                    {"role": "system", "content": REBUTTAL_SYS_PROMPT},
                    {"role": "user", "content": content},
                ]
                all_messages.append(messages)
    
    elif args.mode == "0.1":
        id_review = 0
        for paper in paper_reviews_without_rebuttal:
            paper_content = paper_archive.get(paper["paper_id"], "No content found.")
            for review in paper["reviews"]:
                for element in review_summaries_processed[id_review]:
                    content = build_rebuttal_prompt_content(
                        mode=args.mode,
                        paper_content=paper_content,
                        review_point=element["point"],
                        sft_model = not args.is_api,
                    )
                    messages = [
                        {"role": "system", "content": REBUTTAL_POINTWISE_SYS_PROMPT},
                        {"role": "user", "content": content},
                    ]
                    all_messages.append(messages)
                    
                id_review += 1
    
    elif args.mode in ["1.0", "2.0", "2.c", "2.j", "2.1"]:
        id_review = 0
        
        # Check if first review has relevant_paragraphs field
        first_review = paper_reviews_without_rebuttal[0]["reviews"][0]
        has_relevant_paragraphs = "relevant_paragraphs" in first_review and isinstance(first_review["relevant_paragraphs"], list)
        if has_relevant_paragraphs:
            print("Skipping Retrieval Step: All reviews already have relevant_paragraphs.")
        else:
            # Retrieval should have been performed during preprocessing (Substep 1.2).
            # If we reach here, relevant_paragraphs are missing from input and preprocessing was not run.
            raise AssertionError(
                "relevant_paragraphs not found for reviews. Please run preprocessing to perform retrieval (Substep 1.2) or provide relevant_paragraphs in the input."
            )
    
        for paper_idx, paper in enumerate(paper_reviews_without_rebuttal):
            paper_content = paper_archive.get(paper["paper_id"], "No content found.")
            paper_paragraphs = paper_paragraphs_archive.get(paper["paper_id"], sep_passage(paper_content))
            
            # Collect all relevant_paragraphs into a flat list for this paper
            all_retrieved_indices = []
            for review_idx, review in enumerate(paper["reviews"]):
                for element_idx in range(len(review["relevant_paragraphs"])):
                    relevant_order = review["relevant_paragraphs"][element_idx]
                    retrieved_indices = relevant_order[:min(args.top_k, len(relevant_order))]
                    all_retrieved_indices.append(sorted(retrieved_indices))
            
            # Generate rebuttal messages using retrieved_indices from preprocessing
            id_element = 0
            for review_idx, review in enumerate(paper["reviews"]):
                for element_idx, element in enumerate(review_summaries_processed[id_review]):
                    retrieved_indices = all_retrieved_indices[id_element]
                    retrieved_content = merge_passages([paper_paragraphs[idx] for idx in retrieved_indices])
                    
                    content = build_rebuttal_prompt_content(
                        mode=args.mode,
                        paper_content=retrieved_content,
                        review_point=element["point"],
                        perspectives=element.get("perspectives", []),
                        conf_valid=element.get("confidence", 0) > args.conf_threshold - 1e-8,
                        sft_model = not args.is_api,
                    )
 
                    messages = [
                        {"role": "system", "content": REBUTTAL_POINTWISE_SYS_PROMPT},
                        {"role": "user", "content": content},
                    ]
                    
                    all_messages.append(messages)
                    
                    id_element += 1
                    
                id_review += 1
    
    else:
        raise NotImplementedError("Mode not implemented.")

    if args.is_api:
        responses = api_batch_inference(
            all_messages, 
            sampling_params={"temperature": args.temperature, "top_p": args.top_p, "max_tokens": 5000}, 
            model=args.model, 
            n_threads=12, 
            progress=True
        )
    else:
        llm, tokenizer = ensure_llm_initialized()
        all_messages = [tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) for messages in all_messages]
        params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=10000)
        responses = [output.outputs[0].text.strip() for output in llm.generate(all_messages, params)]
    
    responses = [clean_model_output(response) for response in responses]
    
    '''Step 3.2: Post-Processing rebuttal content'''
    if args.mode == "0.0":
        id_review = 0
        for paper in paper_reviews_without_rebuttal:
            for review in paper["reviews"]:
                review["discussion"] = [{"role": "user", "content": responses[id_review]}]
                id_review += 1
    elif args.mode in ["0.1", "1.0", "2.0", "2.c", "2.j", "2.1"]:
        id_review, id_ans = 0, 0
        
        for paper_idx, paper in enumerate(paper_reviews_without_rebuttal):
            for review_idx, review in enumerate(paper["reviews"]):
                # Save the review summary
                review["summary"] = review_summaries_processed[id_review]
                
                content = "Dear reviewer:\nWe're very grateful for your constructive comments. Below are responses to your suggestions and concerns.\n\n"
                if len(review_summaries_processed[id_review]) == 1:
                    content += responses[id_ans] + "\n\n"
                    id_ans += 1
                else:
                    for element in review_summaries_processed[id_review]:
                        content += "Question: " + element["point"] + "\n"
                        content += "Response: " + responses[id_ans] + "\n\n"
                        id_ans += 1
                    
                content += "We hope these responses address your concerns satisfactorily."
                review["discussion"].append({"role": "user", "content": content})
                id_review += 1
    
    # Remove relevant_paragraphs field and temporary skip info before saving
    for paper in paper_reviews_without_rebuttal:
        for review in paper["reviews"]:
            if "relevant_paragraphs" in review:
                del review["relevant_paragraphs"]
                
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(paper_reviews_without_rebuttal, f, ensure_ascii=False, indent=2)
    



if __name__ == "__main__":
    # 设置命令行参数解析

    parser = argparse.ArgumentParser(description="Generate rebuttal scores for reviews")
    parser.add_argument("--input", "-i", type=str, default="data/test.json", 
                       help="Input JSON file containing reviews")
    parser.add_argument("--output", "-o", type=str, default="data/rebuttal/test_real.json",
                       help="Output JSON file to save results")
    parser.add_argument("--model", "-m", type=str, default="/data/models/Qwen2.5-7B-Instruct",
                       help="Model to use for generating responses")
    parser.add_argument("--is_api", action='store_true',)
    parser.add_argument("--perspective_model", type=str, default="",
                       help="Model to use for generating perspectives")
    parser.add_argument("--is_perspective_api", action='store_true',)
    parser.add_argument("--encoder_model", type=str, default="",
                       help="Model to use for retrieval")
    parser.add_argument("--top_k", type=int, default=15,
                       help="Number of top relevant paragraphs to retrieve (default: 15)")
    parser.add_argument("--limit", "-l", type=int, default=-1,
                       help="Limit number of papers to process")
    parser.add_argument("--n_gpus", type=int, default=1,
                       help="Number of GPUs to use for inference")
    parser.add_argument("--mode", type=str, default="0.0",)
    parser.add_argument("--preprocess_only", action='store_true',
                       help="Only run preprocessing (decomposition and perspective generation) and exit")
    parser.add_argument("--planner_model", type=str, default="",
                       help="Path to trained planner model checkpoint for mode 2.1")
    parser.add_argument("--conf_threshold", type=float, default=0.0,
                       help="The MINIMUM confidence value for planner to work")
    parser.add_argument("--gt_perspective_path", type=str, default="data/revised_score/test_real_real.json")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature for sampling (default: 0.0)")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p (nucleus sampling) for sampling (default: 1.0)")
    args = parser.parse_args()

    paper_reviews_without_rebuttal = json.load(open(args.input, "r"))
    if args.limit != -1:
        paper_reviews_without_rebuttal = paper_reviews_without_rebuttal[:args.limit]

    if args.perspective_model == "":
        args.perspective_model = args.model
        args.is_perspective_api = args.is_api
    
    if os.path.exists(args.output):
        print(f"Output file {args.output} already exists. Skipping generation.")
        exit(0)
    
    # # DEBUG
    # if args.n_gpus == 1:
    #     assert False
    
    gen_rebuttal(paper_reviews_without_rebuttal, args)
    