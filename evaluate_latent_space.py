"""
Evaluate VAE Latent Space with t-SNE Visualization

This script analyzes the trained VAE latent space by:
1. Extracting latent representations from the validation set
2. Computing t-SNE embeddings for visualization
3. Creating plots showing separation between answerable/unanswerable questions
4. Analyzing latent space statistics and clustering
"""

import os
# Set environment variables to fix warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import AutoTokenizer

from config import get_config
from data import create_dataloader
from models.latent_diffusion import LatentDiffusionQA
from models.scaler import LatentScaler


def set_plot_style():
    """Set consistent plot styling."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def extract_latent_representations(
    model, 
    dataloader, 
    device, 
    max_samples: int = 2000,
    use_mean: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], List[bool]]:
    """
    Extract latent representations from the VAE.
    
    Returns:
        latents: [N, latent_dim] - pooled latent vectors
        answers: List of answer strings
        questions: List of question strings  
        is_impossible: List of boolean flags
    """
    model.eval()
    all_latents = []
    all_answers = []
    all_questions = []
    all_is_impossible = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= max_samples:
                break
                
            answer_ids = batch["answer_input_ids"].to(device)
            answer_mask = batch["answer_attention_mask"].to(device)
            
            # Get latent representation from VAE
            if model.use_vae:
                # Get sequence-level latent [batch, seq_len, latent_dim]
                z, mean, logvar = model.vae.encode(answer_ids, answer_mask)
                
                # Use mean or sampled latent
                latent = mean if use_mean else z
                
                # Pool over sequence dimension (mean pooling over valid tokens)
                mask_expanded = answer_mask.unsqueeze(-1).float()
                pooled_latent = (latent * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                # For embedding bridge, get embeddings and pool
                latent = model.vae.encode(answer_ids)
                mask_expanded = answer_mask.unsqueeze(-1).float()
                pooled_latent = (latent * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            
            all_latents.append(pooled_latent.cpu().numpy())
            
            # Decode answers and questions for analysis
            batch_size = answer_ids.size(0)
            for i in range(batch_size):
                if sample_count >= max_samples:
                    break
                    
                # Decode answer
                answer_text = model.tokenizer.decode(
                    answer_ids[i], 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                all_answers.append(answer_text.strip())
                
                # Get question from batch if available
                if "question_input_ids" in batch:
                    question_text = model.tokenizer.decode(
                        batch["question_input_ids"][i],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    all_questions.append(question_text.strip())
                else:
                    all_questions.append("")
                
                all_is_impossible.append(batch["is_impossible"][i].item())
                sample_count += 1
    
    latents = np.vstack(all_latents)[:max_samples]
    answers = all_answers[:max_samples]
    questions = all_questions[:max_samples]
    is_impossible = all_is_impossible[:max_samples]
    
    return latents, answers, questions, is_impossible


def compute_tsne_embeddings(latents: np.ndarray, perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    """Compute t-SNE embeddings."""
    print(f"Computing t-SNE for {latents.shape[0]} samples...")
    
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, latents.shape[0] - 1),
        random_state=random_state,
        learning_rate=200,
        init='pca',
        max_iter=1000  # Use max_iter instead of n_iter for newer scikit-learn versions
    )
    
    tsne_results = tsne.fit_transform(latents)
    print(f"t-SNE completed! Shape: {tsne_results.shape}")
    
    return tsne_results


def analyze_latent_statistics(latents: np.ndarray, is_impossible: List[bool]) -> Dict:
    """Analyze latent space statistics."""
    stats = {}
    
    # Separate latents by answerability
    answerable_mask = np.array([not imp for imp in is_impossible])
    unanswerable_mask = np.array(is_impossible)
    
    answerable_latents = latents[answerable_mask]
    unanswerable_latents = latents[unanswerable_mask]
    
    # Basic statistics
    stats['total_samples'] = len(latents)
    stats['answerable_samples'] = len(answerable_latents)
    stats['unanswerable_samples'] = len(unanswerable_latents)
    
    # Dimension statistics
    stats['latent_dim'] = latents.shape[1]
    stats['mean_norm'] = np.mean(np.linalg.norm(latents, axis=1))
    stats['std_norm'] = np.std(np.linalg.norm(latents, axis=1))
    
    # Class-specific statistics
    if len(answerable_latents) > 0:
        stats['answerable_mean_norm'] = np.mean(np.linalg.norm(answerable_latents, axis=1))
        stats['answerable_std_norm'] = np.std(np.linalg.norm(answerable_latents, axis=1))
        stats['answerable_centroid'] = np.mean(answerable_latents, axis=0)
    
    if len(unanswerable_latents) > 0:
        stats['unanswerable_mean_norm'] = np.mean(np.linalg.norm(unanswerable_latents, axis=1))
        stats['unanswerable_std_norm'] = np.std(np.linalg.norm(unanswerable_latents, axis=1))
        stats['unanswerable_centroid'] = np.mean(unanswerable_latents, axis=0)
    
    # Centroid distance
    if len(answerable_latents) > 0 and len(unanswerable_latents) > 0:
        centroid_distance = np.linalg.norm(stats['answerable_centroid'] - stats['unanswerable_centroid'])
        stats['centroid_distance'] = centroid_distance
    
    # Clustering analysis
    if len(latents) > 10:
        # K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(latents)
        
        # Silhouette score
        sil_score = silhouette_score(latents, cluster_labels)
        stats['silhouette_score'] = sil_score
        
        # Cluster purity (how well clusters align with answerability)
        cluster_0_answerable = np.sum((cluster_labels == 0) & answerable_mask)
        cluster_0_unanswerable = np.sum((cluster_labels == 0) & unanswerable_mask)
        cluster_1_answerable = np.sum((cluster_labels == 1) & answerable_mask)
        cluster_1_unanswerable = np.sum((cluster_labels == 1) & unanswerable_mask)
        
        # Calculate purity
        cluster_0_purity = max(cluster_0_answerable, cluster_0_unanswerable) / max(1, cluster_0_answerable + cluster_0_unanswerable)
        cluster_1_purity = max(cluster_1_answerable, cluster_1_unanswerable) / max(1, cluster_1_answerable + cluster_1_unanswerable)
        stats['cluster_purity'] = (cluster_0_purity + cluster_1_purity) / 2
    
    return stats


def plot_tsne_visualizations(
    tsne_results: np.ndarray,
    is_impossible: List[bool],
    answers: List[str],
    questions: List[str],
    output_dir: str,
    stats: Dict
):
    """Create comprehensive t-SNE visualization plots."""
    set_plot_style()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy arrays
    is_impossible = np.array(is_impossible)
    answerable_mask = ~is_impossible
    
    # Color mapping
    colors = ['red' if imp else 'blue' for imp in is_impossible]
    labels = ['Unanswerable' if imp else 'Answerable' for imp in is_impossible]
    
    # 1. Main t-SNE plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        tsne_results[:, 0], 
        tsne_results[:, 1], 
        c=colors, 
        alpha=0.7, 
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    plt.title(f'VAE Latent Space t-SNE Visualization\n'
              f'Answerable: {np.sum(answerable_mask)} | Unanswerable: {np.sum(is_impossible)}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Add legend
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Answerable')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Unanswerable')
    plt.legend(handles=[blue_patch, red_patch], loc='best')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tsne_main.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. t-SNE with sample annotations
    plt.figure(figsize=(14, 10))
    
    # Plot all points
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, alpha=0.3, s=30)
    
    # Annotate interesting samples
    num_samples_to_annotate = min(20, len(tsne_results))
    indices_to_annotate = np.random.choice(len(tsne_results), num_samples_to_annotate, replace=False)
    
    for idx in indices_to_annotate:
        x, y = tsne_results[idx]
        label = "U" if is_impossible[idx] else "A"
        answer_short = answers[idx][:30] + "..." if len(answers[idx]) > 30 else answers[idx]
        
        plt.annotate(
            f"{label}: {answer_short}",
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    plt.title('t-SNE with Sample Annotations', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tsne_annotated.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # t-SNE component distributions
    axes[0, 0].hist(tsne_results[answerable_mask, 0], bins=30, alpha=0.7, color='blue', label='Answerable')
    axes[0, 0].hist(tsne_results[is_impossible, 0], bins=30, alpha=0.7, color='red', label='Unanswerable')
    axes[0, 0].set_title('t-SNE Component 1 Distribution')
    axes[0, 0].set_xlabel('Component 1 Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(tsne_results[answerable_mask, 1], bins=30, alpha=0.7, color='blue', label='Answerable')
    axes[0, 1].hist(tsne_results[is_impossible, 1], bins=30, alpha=0.7, color='red', label='Unanswerable')
    axes[0, 1].set_title('t-SNE Component 2 Distribution')
    axes[0, 1].set_xlabel('Component 2 Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Answer length distribution
    answer_lengths = [len(ans.split()) for ans in answers]
    answerable_lengths = [answer_lengths[i] for i in range(len(answer_lengths)) if answerable_mask[i]]
    unanswerable_lengths = [answer_lengths[i] for i in range(len(answer_lengths)) if is_impossible[i]]
    
    axes[1, 0].hist(answerable_lengths, bins=20, alpha=0.7, color='blue', label='Answerable')
    axes[1, 0].hist(unanswerable_lengths, bins=20, alpha=0.7, color='red', label='Unanswerable')
    axes[1, 0].set_title('Answer Length Distribution (words)')
    axes[1, 0].set_xlabel('Number of Words')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Latent norm distribution
    if 'answerable_mean_norm' in stats and 'unanswerable_mean_norm' in stats:
        axes[1, 1].bar(['Answerable', 'Unanswerable'], 
                       [stats['answerable_mean_norm'], stats['unanswerable_mean_norm']],
                       color=['blue', 'red'], alpha=0.7)
        axes[1, 1].set_title('Mean Latent Norm by Class')
        axes[1, 1].set_ylabel('Mean Norm')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def save_analysis_results(stats: Dict, output_dir: str):
    """Save analysis results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    json_stats = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            json_stats[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            json_stats[key] = float(value) if isinstance(value, np.floating) else int(value)
        else:
            json_stats[key] = value
    
    with open(f"{output_dir}/latent_analysis.json", 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("LATENT SPACE ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total samples analyzed: {stats['total_samples']}")
    print(f"Answerable samples: {stats['answerable_samples']}")
    print(f"Unanswerable samples: {stats['unanswerable_samples']}")
    print(f"Latent dimension: {stats['latent_dim']}")
    print(f"Mean latent norm: {stats['mean_norm']:.4f} ± {stats['std_norm']:.4f}")
    
    if 'answerable_mean_norm' in stats:
        print(f"Answerable norm: {stats['answerable_mean_norm']:.4f} ± {stats['answerable_std_norm']:.4f}")
        print(f"Unanswerable norm: {stats['unanswerable_mean_norm']:.4f} ± {stats['unanswerable_std_norm']:.4f}")
    
    if 'centroid_distance' in stats:
        print(f"Centroid distance: {stats['centroid_distance']:.4f}")
    
    if 'silhouette_score' in stats:
        print(f"Silhouette score: {stats['silhouette_score']:.4f}")
        print(f"Cluster purity: {stats['cluster_purity']:.4f}")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VAE Latent Space with t-SNE")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_file", type=str, default="data/dev-v2.0.json", help="Data file to analyze")
    parser.add_argument("--output_dir", type=str, default="./latent_analysis", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=2000, help="Maximum samples to analyze")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--use_mean", action='store_true', default=True, help="Use mean instead of sampled latent")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/cpu/mps)")
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            # Use CPU for evaluation due to MPS compatibility issues with transformers
            print("MPS detected but using CPU for transformer compatibility")
            device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load config
    config = get_config()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_encoder)
    
    # Create data loader
    print(f"Loading data from {args.data_file}...")
    dataloader, dataset = create_dataloader(
        args.data_file,
        tokenizer,
        args.batch_size,
        max_context_length=config.model.max_context_length,
        max_question_length=config.model.max_question_length,
        max_answer_length=config.model.max_answer_length,
        use_balanced_sampler=False,
        shuffle=False,
        num_workers=0,  # Use single worker to avoid fork warnings
    )
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    latent_scaler = LatentScaler()
    
    model = LatentDiffusionQA(
        tokenizer=tokenizer,
        latent_dim=config.model.vae_latent_dim,
        d_model=config.model.denoiser_dim,
        num_layers=config.model.denoiser_layers,
        num_heads=config.model.denoiser_heads,
        ff_dim=config.model.denoiser_ff_dim,
        dropout=config.model.dropout,
        max_answer_len=config.model.max_answer_length,
        num_train_timesteps=config.diffusion.num_train_timesteps,
        num_inference_timesteps=config.diffusion.num_inference_timesteps,
        schedule_type=config.diffusion.schedule_type,
        use_vae=True,
        base_encoder=config.model.base_encoder,
        false_negative_penalty_weight=config.training.false_negative_penalty_weight,
        scaler=latent_scaler,
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # VAE warmup checkpoints store state dict directly
        model.load_state_dict(checkpoint)
    
    # Load scaler stats if available
    if "scaler_mean" in checkpoint and checkpoint["scaler_mean"] is not None:
        print("Loading latent scaler stats from checkpoint...")
        model.scaler.mean = checkpoint["scaler_mean"].to(device)
        model.scaler.std = checkpoint["scaler_std"].to(device)
        model.scaler.to(device)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    
    # Extract latent representations
    print("Extracting latent representations...")
    latents, answers, questions, is_impossible = extract_latent_representations(
        model, dataloader, device, args.max_samples, args.use_mean
    )
    
    print(f"Extracted {len(latents)} latent representations")
    
    # Analyze latent space statistics
    print("Analyzing latent space statistics...")
    stats = analyze_latent_statistics(latents, is_impossible)
    
    # Compute t-SNE embeddings
    tsne_results = compute_tsne_embeddings(latents, args.perplexity)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_tsne_visualizations(tsne_results, is_impossible, answers, questions, args.output_dir, stats)
    
    # Save results
    save_analysis_results(stats, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
