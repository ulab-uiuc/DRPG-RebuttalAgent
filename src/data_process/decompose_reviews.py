"""
分解评审脚本
将评审内容分解为独立的问题点，并在原文件中添加 decomposed_content 字段
可选：生成 relevant_paragraphs 字段（段落相关度排序）
"""

from common_tools import *
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import json
import torch
import numpy as np


# 加载论文档案
paper_archive = json.load(open("data/processed_papers.json", "r"))
paper_paragraphs_archive = json.load(open("data/processed_paper_paragraphs.json", "r"))


def decompose_reviews(reviews_data, args, output_path=None):
    """
    分解评审内容为独立的问题点
    
    Args:
        reviews_data: List of dicts, 每个dict包含 'paper_id' 和 'reviews'
        args: 命令行参数
            - args.model: 使用的模型
            - args.is_api: 是否使用API模式
            - args.n_gpus: GPU数量（非API模式）
            - args.add_retrieval: 是否添加 relevant_paragraphs 字段
            - args.encoder_model: 检索使用的编码器模型
            - args.doc_embeddings: 预计算的文档embeddings路径
        output_path: 输出文件路径（如果提供则保存结果）
    
    Returns:
        处理后的reviews_data，每个review增加了 'decomposed_content' 字段
        如果 args.add_retrieval=True，还会添加 'relevant_paragraphs' 字段
    """
    
    # 初始化LLM模型
    if not args.is_api:
        llm = LLM(
            model=args.model,
            dtype='bfloat16',
            tensor_parallel_size=args.n_gpus,
            gpu_memory_utilization=0.8,
            disable_sliding_window=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # 如果需要检索，初始化编码器模型
    if args.add_retrieval:
        print(f"初始化编码器模型: {args.encoder_model}")
        encoder_model = SentenceTransformer(args.encoder_model)
        
        # 加载预计算的文档embeddings
        if args.doc_embeddings:
            print(f"加载预计算的文档embeddings: {args.doc_embeddings}")
            all_doc_embeddings = torch.load(args.doc_embeddings)
        else:
            all_doc_embeddings = None
    
    # 收集所有需要分解的评审
    all_decompose_messages = []
    original_reviews = []
    review_indices = []  # 记录每个评审对应的 (paper_idx, review_idx)
    
    for paper_idx, paper in enumerate(reviews_data):
        for review_idx, review in enumerate(paper["reviews"]):
            all_decompose_messages.append([
                {"role": "system", "content": REBUTTAL_DECOMPOSER_SYS_PROMPT},
                {"role": "user", "content": review["review_content"]},
            ])
            original_reviews.append(review["review_content"])
            review_indices.append((paper_idx, review_idx))
    
    print(f"总共需要分解 {len(all_decompose_messages)} 条评审")
    
    # 批量推理
    if args.is_api:
        decompose_responses = api_batch_inference(
            all_decompose_messages, 
            sampling_params={"temperature": 0, "max_tokens": 10000}, 
            model=args.model, 
            n_threads=10,
            progress=True
        )
    else:
        all_decompose_messages = [
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) 
            for messages in all_decompose_messages
        ]
        params = SamplingParams(temperature=0, max_tokens=10000)
        decompose_responses = [
            output.outputs[0].text.strip() 
            for output in llm.generate(all_decompose_messages, params)
        ]
    
    # 处理分解结果
    for idx, (res, (paper_idx, review_idx)) in enumerate(zip(decompose_responses, review_indices)):
        points = process_json(res)
        
        # 如果解析失败或返回空列表，使用原始评审作为单个点
        if not isinstance(points, list) or len(points) == 0:
            points = [original_reviews[idx]]
            print(f"警告: 第 {idx+1} 条评审分解失败，使用原始评审作为单个点")
        
        # 确保所有点都是字符串
        points = [str(p) if not isinstance(p, str) else p for p in points]
        # 将分解结果添加到原数据中
        reviews_data[paper_idx]["reviews"][review_idx]["decomposed_content"] = points
        
        
    # 先保存一部分结果
    json.dump(reviews_data, open(output_path, "w"), ensure_ascii=False, indent=2)
        
        
    # 如果需要，生成 relevant_paragraphs 字段
    if args.add_retrieval:
        print("\n" + "=" * 60)
        print("开始生成 relevant_paragraphs 字段")
        print("=" * 60)
        
        for paper_idx, paper in enumerate(reviews_data):
            paper_id = paper["paper_id"]
            
            # 获取论文段落
            paper_content = paper_archive.get(paper_id, "")
            if not paper_content:
                print(f"警告: 论文 {paper_id} 内容未找到，跳过检索")
                continue
            
            paper_paragraphs = paper_paragraphs_archive.get(paper_id, sep_passage(paper_content))
            
            # 获取文档embeddings
            if all_doc_embeddings and paper_id in all_doc_embeddings:
                doc_embeddings = all_doc_embeddings[paper_id].to("cuda:0")
            else:
                # 现场编码
                print(f"  为论文 {paper_id} 现场编码段落...")
                doc_embeddings = encoder_model.encode(paper_paragraphs, convert_to_tensor=True)
            
            # 对每个评审的每个问题点进行检索
            for review_idx, review in enumerate(paper["reviews"]):
                decomposed_points = review.get("decomposed_content", [])
                relevant_paragraphs = []
                
                # 编码所有问题点
                if decomposed_points:
                    query_embeddings = encoder_model.encode(decomposed_points, convert_to_tensor=True)
                    
                    # 计算相似度并排序
                    for query_idx, query_embedding in enumerate(query_embeddings):
                        similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
                        # 获取按相似度排序的段落索引（降序）
                        sorted_indices = torch.argsort(similarities, descending=True).cpu().numpy().tolist()
                        relevant_paragraphs.append(sorted_indices)
                
                # 添加到评审中
                review["relevant_paragraphs"] = relevant_paragraphs
            
            if (paper_idx + 1) % 10 == 0 or paper_idx == 0:
                print(f"已完成 {paper_idx + 1}/{len(reviews_data)} 篇论文的检索")
    
    # 如果提供了输出路径，保存结果
    if output_path:
        print(f"\n正在保存结果到: {output_path}")
        
        # 先正常序列化为带缩进的JSON字符串
        json_str = json.dumps(reviews_data, ensure_ascii=False, indent=2)
        
        # 对 relevant_paragraphs 字段进行特殊处理：将其压缩为单行
        # 使用正则表达式查找所有 "relevant_paragraphs": [ ... ] 的内容并压缩
        import re
        
        def compress_relevant_paragraphs(match):
            """将 relevant_paragraphs 字段压缩为单行"""
            # 获取完整的数组内容
            full_match = match.group(0)
            # 移除所有换行和多余空格
            compressed = re.sub(r'\s+', ' ', full_match)
            # 确保数组内部格式正确（逗号后有一个空格）
            compressed = re.sub(r'\s*,\s*', ', ', compressed)
            compressed = re.sub(r'\[\s*', '[', compressed)
            compressed = re.sub(r'\s*\]', ']', compressed)
            return compressed
        
        # 匹配 "relevant_paragraphs": 后面的完整数组（包括嵌套）
        # 使用递归匹配来处理嵌套数组
        pattern = r'"relevant_paragraphs":\s*\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\]'
        json_str = re.sub(pattern, compress_relevant_paragraphs, json_str)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        
        print("=" * 60)
        print("✓ 文件保存完成！")
        
        # 统计信息
        total_reviews = sum(len(paper["reviews"]) for paper in reviews_data)
        total_points = sum(
            len(review.get("decomposed_content", []))
            for paper in reviews_data
            for review in paper["reviews"]
        )
        avg_points = total_points / total_reviews if total_reviews > 0 else 0
        
        print(f"\n统计信息:")
        print(f"  总评审数: {total_reviews}")
        print(f"  总问题点数: {total_points}")
        print(f"  平均每条评审分解为: {avg_points:.2f} 个问题点")
        
        if args.add_retrieval:
            total_rankings = sum(
                len(review.get("relevant_paragraphs", []))
                for paper in reviews_data
                for review in paper["reviews"]
            )
            print(f"  生成的段落排序数: {total_rankings}")
        
        # 显示示例
        if len(reviews_data) > 0 and len(reviews_data[0]["reviews"]) > 0:
            print(f"\n示例 - 第一篇论文的第一条评审:")
            first_review = reviews_data[0]["reviews"][0]
            print(f"  原始评审长度: {len(first_review['review_content'])} 字符")
            print(f"  分解后问题点数: {len(first_review.get('decomposed_content', []))}")
            print(f"  问题点列表:")
            for i, point in enumerate(first_review.get("decomposed_content", [])[:3], 1):
                preview = point[:80] + "..." if len(point) > 80 else point
                print(f"    {i}. {preview}")
            if len(first_review.get("decomposed_content", [])) > 3:
                print(f"    ... (共 {len(first_review['decomposed_content'])} 个点)")
            
            if args.add_retrieval and "relevant_paragraphs" in first_review:
                print(f"\n  相关段落排序:")
                for i, ranking in enumerate(first_review["relevant_paragraphs"][:3], 1):
                    top5 = ranking[:5] if len(ranking) > 5 else ranking
                    print(f"    问题点{i} 的Top-5段落: {top5}")
                if len(first_review.get("relevant_paragraphs", [])) > 3:
                    print(f"    ... (共 {len(first_review['relevant_paragraphs'])} 个问题点的排序)")
    
    return reviews_data


def main():
    parser = argparse.ArgumentParser(description="分解评审内容为独立的问题点，可选生成段落相关度排序")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="输入的评审JSON文件路径")
    parser.add_argument("--output", "-o", type=str, default="",
                       help="输出JSON文件路径（默认为原位修改输入文件）")
    parser.add_argument("--model", "-m", type=str, default="/data/models/Qwen2.5-7B-Instruct",
                       help="使用的模型路径或名称")
    parser.add_argument("--is_api", action='store_true',
                       help="是否使用API模式")
    parser.add_argument("--n_gpus", type=int, default=1,
                       help="使用的GPU数量（非API模式）")
    parser.add_argument("--limit", "-l", type=int, default=-1,
                       help="限制处理的论文数量（用于测试）")
    parser.add_argument("--add_retrieval", action='store_true',
                       help="是否添加 relevant_paragraphs 字段（段落相关度排序）")
    parser.add_argument("--encoder_model", type=str, default="",
                       help="检索使用的编码器模型路径（add_retrieval=True时需要）")
    parser.add_argument("--doc_embeddings", type=str, default="")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.add_retrieval and not args.encoder_model:
        parser.error("--add_retrieval 需要指定 --encoder_model")
    
    # 读取输入文件
    print(f"正在读取输入文件: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        reviews_data = json.load(f)
    
    print(f"读取到 {len(reviews_data)} 篇论文的评审数据")
    
    # 限制处理数量（用于测试）
    if args.limit > 0:
        reviews_data = reviews_data[:args.limit]
        print(f"限制处理前 {args.limit} 篇论文")
    
    # 统计总评审数
    total_reviews = sum(len(paper["reviews"]) for paper in reviews_data)
    print(f"总共 {total_reviews} 条评审需要分解")
    
    # 执行分解
    print(f"\n使用模型: {args.model}")
    print(f"模式: {'API' if args.is_api else f'Local (GPU: {args.n_gpus})'}")
    if args.add_retrieval:
        print(f"检索模式: 启用 (编码器: {args.encoder_model})")
    print("=" * 60)
    
    # 确定输出路径
    output_path = args.output if args.output else args.input
    
    # 执行分解并保存（保存功能已集成到函数内部）
    reviews_data = decompose_reviews(reviews_data, args, output_path=output_path)


if __name__ == "__main__":
    main()
