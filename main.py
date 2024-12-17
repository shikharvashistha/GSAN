import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from sklearn.metrics import f1_score
import seaborn as sns
from dgl import AddSelfLoop
from sklearn.manifold import TSNE
from utils.constants import *
import matplotlib.pyplot as plt
import numpy as np

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb    :128'

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, target):
        # Ensure target is one-hot encoded if pred is logits
        if pred.dim() > 1 and target.dim() == 1:
            target = F.one_hot(target, num_classes=pred.size(1)).float()

        # Compute L1 loss
        return F.l1_loss(pred, target)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target):
        # Ensure target is one-hot encoded if pred is logits
        if pred.dim() > 1 and target.dim() == 1:
            target = F.one_hot(target, num_classes=pred.size(1)).float()
        return F.mse_loss(pred, target)

class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, pred, target):
        # Ensure target is one-hot encoded if pred is logits
        if pred.dim() > 1 and target.dim() == 1:
            target = F.one_hot(target, num_classes=pred.size(1)).float()
        return F.smooth_l1_loss(pred, target)


def evaluate(g, features, labels, mask, model, epoch=None):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]

        # If labels are not one-hot encoded, convert them
        if labels.dim() == 1:
            labels = F.one_hot(labels, num_classes=logits.size(1)).float()

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == torch.argmax(labels, dim=1))  # Ensure correct comparison

        # Compute micro F1 score
        pred_labels = indices.cpu().numpy()
        true_labels = torch.argmax(labels, dim=1).cpu().numpy()
        micro_f1 = f1_score(true_labels, pred_labels, average='micro')

        return correct.item() / labels.size(0), micro_f1


def plot_attention_weights(g, attention_weights_tuple, epoch, dataset):
    g = g.to('cpu')
    attention_weights = attention_weights_tuple[1].squeeze().cpu().detach().numpy()

    if attention_weights.ndim > 1 and attention_weights.shape[1] > 1:
        attention_weights = np.mean(attention_weights, axis=1)
        print("Converted attention weights to mean values across dimensions.")

    nx_g = g.to_networkx().to_undirected()
    num_nodes = len(nx_g)
    attn_matrix = np.zeros((num_nodes, num_nodes))

    for (src, dst), attn in zip(zip(g.edges()[0].numpy(), g.edges()[1].numpy()), attention_weights):
        attn_matrix[src, dst] = attn
        attn_matrix[dst, src] = attn

    # Use a percentile to normalize to avoid the influence of outliers/extreme values
    low, high = np.percentile(attention_weights, [5, 95])
    norm = plt.Normalize(vmin=low, vmax=high)
    cmap = plt.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(attn_matrix, interpolation='nearest', cmap=cmap, norm=norm)
    fig.colorbar(cax, label='Attention Weights')

    ax.set_xlabel('Nodes')
    ax.set_ylabel('Nodes')
    ax.set_title(f"Attention Weights Heatmap at Epoch {epoch} - {dataset}")
    plt.savefig(f"graphs/attention_weights_heatmap_epoch_{epoch}_{dataset}.pdf", format='pdf', dpi=1200)
    plt.show()

def plot_tsne(features, labels, epoch, dataset=None):
    tsne = TSNE(n_components=2, random_state=33)
    embeddings = tsne.fit_transform(features.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', alpha=0.6)
    colorbar = plt.colorbar(scatter)
    plt.axis('off')  # Hide x and y axes
    colorbar.ax.tick_params(labelsize=17)  # Set font size for color bar labels
    pdf_filename = f"graphs/tsne_epoch_{epoch}_{dataset}.pdf"
    plt.savefig(pdf_filename, format='pdf', dpi=1200)
    plt.show()

def plot_heatmap(data, title, epoch):
    sns.heatmap(data, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f'{title} at Epoch {epoch}')
    plt.xlabel('Parameter')
    plt.ylabel('Epoch')
    pdf_filename = f"graphs/{title}_epoch_{epoch}.pdf"
    plt.savefig(pdf_filename, format='pdf', dpi=1200)
    plt.show()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GSAN for node classification on citation networks')
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'],
                        help='Dataset to use')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')
    parser.add_argument('--checkpoint', type=str, default='./models/checkpoints/best_model.pth', help='Path to the model checkpoint')
    args = parser.parse_args()

    transform = AddSelfLoop()

    # Load data
    if args.dataset == 'cora':
        dataset = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        dataset = PubmedGraphDataset()
    else:
        raise ValueError("Dataset not supported. Please choose from 'cora', 'citeseer', or 'pubmed'.")

    g = dataset[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    masks = (train_mask, val_mask, test_mask)

    # Ensure all data is moved to GPU
    g = g.to(args.device)
    features = features.to(args.device)
    labels = labels.to(args.device)
    train_mask = train_mask.to(args.device)
    val_mask = val_mask.to(args.device)
    test_mask = test_mask.to(args.device)

    # Load model checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading model checkpoint from {args.checkpoint}")
        model = torch.load(args.checkpoint, map_location=args.device)  # Load the entire model
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # Evaluate on validation set
    val_acc, val_micro_f1 = evaluate(g, features, labels, val_mask, model)
    print(f"Validation Accuracy: {val_acc:.4f}, Validation Micro F1: {val_micro_f1:.4f}")

if __name__ == '__main__':
    main()
