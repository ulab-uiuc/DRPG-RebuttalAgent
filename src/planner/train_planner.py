import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import os
import sys
import wandb
import json
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent))
from common_tools import process_json
from planner.process_planner_data import PlannerDataset


embeddings_archive = torch.load("data/processed_paper_embeddings.pt")

class SoftMatchingPlanner(nn.Module):
    """
    模型结构：
      - Shared encoder: same pretrained LM for paragraphs & perspectives
      - Matching MLP: f([h_i; g_j]) → scalar score
      - Soft top-k pooling: learnable τ
    similarity_only=True 时：
      - 强制 freeze_encoder=True
      - 不训练 MLP
      - 前向直接使用 passage-perspective embedding cosine 相似度
    """
    def __init__(self, pretrained_encoder, mlp_hidden=[512], k=3, freeze_encoder=False, similarity_only=False):
        super().__init__()
        self.similarity_only = similarity_only
        self.k = k

        # ====== 1. Encoder ======
        self.encoder = AutoModel.from_pretrained(pretrained_encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_encoder)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = self.encoder.config.hidden_size

        # 可选：冻结 encoder 参数（只训练 MLP）
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder or self.similarity_only:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ====== 2. Matching MLP ======
        if not self.similarity_only:
            if isinstance(mlp_hidden, int):
                mlp_hidden = [mlp_hidden]

            if self.k == 0:
                print("Hello")
                self.first_hidden_size = self.hidden_size
            else:
                self.first_hidden_size = self.hidden_size * 2

            layers = []
            input_dim = self.first_hidden_size
            for hidden_dim in mlp_hidden:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, 1))
            self.matching_mlp = nn.Sequential(*layers)
            print(self.matching_mlp)
        else:
            self.first_hidden_size = self.hidden_size
            self.matching_mlp = None

    # -------------------------------------------------------------
    # 辅助函数：编码一批文本 (token-level 平均池化)
    # -------------------------------------------------------------
    def encode_texts(self, texts, type='passage'):
        inputs = self.tokenizer(
            texts, padding=True, return_tensors="pt"
        ).to(next(self.parameters()).device)

        outputs = self.encoder(**inputs)
        attn_mask = inputs["attention_mask"].unsqueeze(-1)
        emb = (outputs.last_hidden_state * attn_mask).sum(1) / attn_mask.sum(1)
        return emb  # [batch, hidden]

    # -------------------------------------------------------------
    # 前向传播
    # samples: list[dict] 或 dict (单个sample)
    #   每个dict包含:
    #   - 'passages_content': list[str] - passage texts
    #   - 'perspectives': list[str] - perspective texts
    #   - 'paper_id': str (optional) - paper identifier
    # 返回: 对于批量输入，返回 list[tuple(S, scores)]
    #       对于单个输入，返回 S, scores
    # -------------------------------------------------------------
    def forward(self, samples, verbose: bool = True):
        device = next(self.parameters()).device
        is_single = isinstance(samples, dict)
        if is_single:
            samples = [samples]

        batch_size = len(samples)

        # 1️⃣ 收集所有passages和perspectives
        all_passages = []
        all_perspectives = []
        passage_boundaries = [0]
        perspective_boundaries = [0]

        for sample in samples:
            passages = sample['passages_content']
            perspectives = sample['perspectives']
            all_passages.extend(passages)
            all_perspectives.extend(perspectives)
            passage_boundaries.append(passage_boundaries[-1] + len(passages))
            perspective_boundaries.append(perspective_boundaries[-1] + len(perspectives))

        # 2️⃣ 批量编码所有文本
        if self.freeze_encoder:
            h_list = []
            for sample in samples:
                passage_ids = sample['passages']
                paper_id = sample.get('paper_id', None)
                paper_embs = embeddings_archive[paper_id]
                h_sample = torch.stack([paper_embs[pid] for pid in passage_ids])
                h_list.append(h_sample)
            h_all = torch.cat(h_list, dim=0).to(device)
        else:
            h_all = self.encode_texts(all_passages, "passage") if all_passages else torch.empty(0, self.hidden_size).to(device)

        g_all = self.encode_texts(all_perspectives, "perspective") if all_perspectives else torch.empty(0, self.hidden_size).to(device)

        results = []

        for i in range(batch_size):
            h = h_all[passage_boundaries[i]:passage_boundaries[i+1]]  # [N_i, H]
            g = g_all[perspective_boundaries[i]:perspective_boundaries[i+1]]  # [M_i, H]

            if self.similarity_only:
                h_norm = F.normalize(h, dim=-1)
                g_norm = F.normalize(g, dim=-1)
                scores = torch.matmul(h_norm, g_norm.T)  # [N, M]
                S = scores.mean(dim=0)  # [M]

            elif self.k == 0:
                S = self.matching_mlp(g).squeeze(-1)  # [M]
                scores = None

            else:
                N, H = h.size()
                M = g.size(0)
                h_expand = h.unsqueeze(1).expand(N, M, H)
                g_expand = g.unsqueeze(0).expand(N, M, H)
                pair = torch.cat([h_expand, g_expand], dim=-1)  # [N, M, 2H]
                scores = self.matching_mlp(pair).squeeze(-1)  # [N, M]
                S = scores.mean(dim=0)  # [M]

            if verbose:
                results.append((S, scores))
            else:
                results.append(S)

        if is_single:
            return results[0]
        else:
            return results



def custom_collate_fn(batch):
    return batch


def train_epoch(model, dataloader, optimizer, device, epoch, log_interval=10):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=True)
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        # 批量前向传播
        results = model(batch)
        
        # 计算每个sample的loss并累加
        batch_loss = 0
        for i, (S, scores) in enumerate(results):
            target_idx = torch.tensor(batch[i]['ground_truth_pos']).to(device)
            loss = F.cross_entropy(S.unsqueeze(0), target_idx.unsqueeze(0))
            batch_loss += loss
        
        # 平均batch中的loss
        batch_loss = batch_loss / len(batch)
        
        # 反向传播
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
        
        # 更新进度条
        avg_loss = total_loss / num_batches
        pbar.set_postfix({
            'loss': f'{batch_loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })
        
        # 记录到wandb
        global_step = epoch * len(dataloader) + batch_idx
        wandb.log({
            "train/loss": batch_loss.item(),
            "train/step": global_step
        }, step=global_step)
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device, save_path=None):
    """评估模型
    
    Args:
        model: 待评估的模型
        dataloader: 数据加载器
        device: 设备
        save_path: 保存评估结果的路径（可选）
        
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 保存详细评估结果
    eval_results = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=True)
        for batch_idx, batch in enumerate(pbar):
            # 批量前向传播
            results = model(batch)
            
            # 处理每个sample的结果
            for sample_idx, (S, scores) in enumerate(results):
                sample = batch[sample_idx]
                target_idx = torch.tensor(sample['ground_truth_pos']).to(device)
                
                # 提取数据用于记录
                passages = sample['passages_content']
                perspectives = sample['perspectives']
                
                # 计算loss
                loss = F.cross_entropy(S.unsqueeze(0), target_idx.unsqueeze(0))
                total_loss += loss.item()
                
                # 计算准确率
                pred_idx = torch.argmax(S)
                is_correct = (pred_idx == target_idx).item()
                correct += is_correct
                total += 1
                
                # 记录详细结果
                result_item = {
                    'batch_idx': batch_idx,
                    'sample_idx': sample_idx,
                    'paper_id': sample.get('paper_id', f'unknown_{batch_idx}_{sample_idx}'),
                    'review_id': sample.get('review_id', f'unknown_{batch_idx}_{sample_idx}'),
                    'num_passages': len(passages),
                    'num_perspectives': len(perspectives),
                    'passages': passages,
                    'perspectives': perspectives,
                    'ground_truth_idx': target_idx.item(),
                    'predicted_idx': pred_idx.item(),
                    'is_correct': is_correct,
                    'loss': loss.item(),
                    'scores': S.cpu().tolist(),
                    'matching_scores': scores.cpu().tolist() if scores is not None else None,  # [N, M] 匹配分数矩阵，k=0时为None
                }
                eval_results.append(result_item)
            
            # 更新进度条
            if total > 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/total:.4f}',
                    'acc': f'{correct/total:.4f}'
                })
    
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    # 保存评估结果
    if save_path is not None:
        import json
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        summary = {
            'total_samples': total,
            'correct': correct,
            'accuracy': accuracy,
            'avg_loss': avg_loss,
            'results': eval_results
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Evaluation results saved to {save_path}")
    
    return avg_loss, accuracy


def training_loop(model, train_loader, test_loader, optimizer, device, args):
    """完整的训练循环"""
    best_accuracy = 0.0
    print(f"\nStarting training for {args.num_epochs} epochs...\n")
    
    for epoch in range(args.num_epochs):
        # 创建当前epoch的输出文件夹
        epoch_output_dir = os.path.join(args.output, f"epoch_{epoch + 1}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args.log_interval)
        
        # 测试并保存评估结果
        eval_save_path = os.path.join(epoch_output_dir, "eval_results.json")
        test_loss, test_accuracy = evaluate(model, test_loader, device, save_path=eval_save_path)
        
        # 打印epoch总结
        print(f"Epoch {epoch + 1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
        
        # 记录epoch级别的指标到wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": train_loss,
            "test/loss": test_loss,
            "test/accuracy": test_accuracy,
        })
        
        # 保存checkpoint到epoch文件夹
        checkpoint_path = os.path.join(epoch_output_dir, "checkpoint.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'config': {
                'encoder': args.encoder,
                'mlp_hidden': args.mlp_hidden_size,
                'freeze_encoder': args.freeze_encoder,
            }
        }, checkpoint_path)
        
        # 保存最佳模型（放在根output文件夹）
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_path = os.path.join(args.output, "best_model.pt")
            best_epoch_marker = os.path.join(args.output, "best_epoch.txt")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'config': {
                    'encoder': args.encoder,
                    'mlp_hidden': args.mlp_hidden_size,
                    'freeze_encoder': args.freeze_encoder,
                }
            }, best_model_path)
            
            # 记录最佳epoch
            with open(best_epoch_marker, 'w') as f:
                f.write(f"Best epoch: {epoch + 1}\n")
                f.write(f"Best accuracy: {best_accuracy:.4f}\n")
            
            print(f"  → New best model saved (accuracy: {best_accuracy:.4f})")
    
    # 训练结束
    print(f"\n{'='*60}")
    print(f"Training completed! Best test accuracy: {best_accuracy:.4f}")
    print(f"{'='*60}\n")
    
    # 保存最终模型（放在根output文件夹）
    final_model_path = os.path.join(args.output, "final_model.pt")
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'encoder': args.encoder,
            'mlp_hidden': args.mlp_hidden_size,
            'freeze_encoder': args.freeze_encoder,
        }
    }, final_model_path)
    
    return best_accuracy




if __name__ == "__main__":
    import os
    
    parser = argparse.ArgumentParser(description="Train soft matching planner")
    parser.add_argument("--output", "-o", type=str, default="models/planner/test", 
                       help="Output Folder Name")
    parser.add_argument("--encoder", type=str, required=True,
                       help="Pretrained encoder model name or path (e.g., bert-base-uncased)")
    parser.add_argument("--freeze_encoder", action="store_true",
                       help="Freeze encoder parameters during training")
    parser.add_argument("--randomize", action="store_true",)
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--top_k", type=int, default=15,
                       help="Number of relevant paragraphs, -1 for all")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for optimizer")
    parser.add_argument("--mlp_hidden_size", type=int, nargs="+", default=[512],
                       help="MLP hidden layer sizes (space-separated for multiple layers, e.g., 512 256 128)")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="How many batches to wait before logging training status")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name")
    parser.add_argument("--similarity_only", action="store_true")
    
    args = parser.parse_args()
    
    # 使用解析后的列表
    mlp_hidden = args.mlp_hidden_size
    
    os.makedirs(args.output, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化模型
    model = SoftMatchingPlanner(
        pretrained_encoder=args.encoder,
        mlp_hidden=mlp_hidden,
        freeze_encoder=args.freeze_encoder
    ).to(device)
    
    print(f"Model initialized with encoder: {args.encoder}")
    print(f"Hidden size (from encoder config): {model.first_hidden_size}")
    print(f"MLP hidden layers: {mlp_hidden}")
    print(f"Encoder frozen: {args.freeze_encoder}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 加载数据集
    print(f"\nLoading training data...")
    train_dataset = PlannerDataset(rebuttal_file="data/revised_score/train_real_real.json", retrieval_file="data/train.json", persp_file="data/perspective/llama-3.3-70b-instruct/train.json", cutoff = 50000, randomize=args.randomize, top_k=args.top_k)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn  # 使用自定义collate函数支持batch_size > 1
    )
    
    print(f"Loading test data...")
    test_dataset = PlannerDataset(rebuttal_file="data/revised_score/test_real_real.json", retrieval_file="data/test.json", persp_file="data/perspective/llama-3.3-70b-instruct/test.json", randomize=args.randomize, top_k=args.top_k)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    if args.similarity_only:
        assert args.freeze_encoder
        print("Only evaluation for similarity only mode.")
        loss, acc = evaluate(model, test_loader, device)
        print(f"Similarity-only | loss={loss:.4f} acc={acc:.4f}")
        
    else:
        # 初始化wandb
        wandb.init(
            name=args.wandb_run_name,
            config=vars(args)
        )
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        # 执行训练循环
        best_accuracy = training_loop(model, train_loader, test_loader, optimizer, device, args)
        
        wandb.finish()
        print(f"\nTraining finished. Best accuracy: {best_accuracy:.4f}")