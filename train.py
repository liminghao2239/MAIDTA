import json
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from models.maidta import MAIDTA
from data_utils.dataset import prepare_maidta_data
from utils import *


def collate_fn(data_list):
    drug_batch = {
        'smiles_tokens': torch.stack([d[0]['smiles_tokens'] for d in data_list]),
        'graph_x': torch.cat([d[0]['graph_x'] for d in data_list], dim=0),
        'graph_edge_index': torch.cat([
            d[0]['graph_edge_index'] + d[0]['graph_x'].size(0) * i
            for i, d in enumerate(data_list)
        ], dim=1),
        'graph_edge_attr': torch.cat([d[0]['graph_edge_attr'] for d in data_list], dim=0),
        'y': torch.stack([d[0]['y'] for d in data_list]),
        'batch': torch.cat([
            torch.full((d[0]['graph_x'].size(0),), i, dtype=torch.long)
            for i, d in enumerate(data_list)
        ], dim=0)
    }

    prot_batch = {
        'seq_tokens': torch.stack([d[1]['seq_tokens'] for d in data_list]),
        'graph_x': torch.cat([d[1]['graph_x'] for d in data_list], dim=0),
        'graph_edge_index': torch.cat([
            d[1]['graph_edge_index'] + d[1]['graph_x'].size(0) * i
            for i, d in enumerate(data_list)
        ], dim=1),
        'graph_edge_attr': torch.cat([d[1]['graph_edge_attr'] for d in data_list], dim=0),
        'batch': torch.cat([
            torch.full((d[1]['graph_x'].size(0),), i, dtype=torch.long)
            for i, d in enumerate(data_list)
        ], dim=0)
    }

    return drug_batch, prot_batch


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for drug_batch, prot_batch in loader:
        for key in drug_batch:
            if isinstance(drug_batch[key], torch.Tensor):
                drug_batch[key] = drug_batch[key].to(device)
        for key in prot_batch:
            if isinstance(prot_batch[key], torch.Tensor):
                prot_batch[key] = prot_batch[key].to(device)

        optimizer.zero_grad()
        pred, _, _ = model(drug_batch, prot_batch)
        label = drug_batch['y'].view(-1)

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(pred.detach().cpu().numpy())
        all_labels.extend(label.detach().cpu().numpy())

    return total_loss / len(loader), get_mse(np.array(all_labels), np.array(all_preds)), get_ci(np.array(all_labels), np.array(all_preds))


def eval_epoch(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for drug_batch, prot_batch in loader:
            for key in drug_batch:
                if isinstance(drug_batch[key], torch.Tensor):
                    drug_batch[key] = drug_batch[key].to(device)
            for key in prot_batch:
                if isinstance(prot_batch[key], torch.Tensor):
                    prot_batch[key] = prot_batch[key].to(device)

            pred, _, _ = model(drug_batch, prot_batch)
            label = drug_batch['y'].view(-1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(config.get('seed', 2024))
    np.random.seed(config.get('seed', 2024))

    data_path = config['data_path']
    data_file = os.path.join(data_path, 'data.csv')
    df = pd.read_csv(data_file)

    smiles_list = df['SMILES'].values
    seq_list = df['Sequence'].values
    labels = df['Affinity'].values

    train_idx, test_idx = train_test_split(range(len(df)), test_size=0.2, random_state=config.get('seed', 2024))
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=config.get('seed', 2024))

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    esm_dir = config.get('esm_dir', None)

    train_data = prepare_maidta_data(
        [smiles_list[i] for i in train_idx],
        [seq_list[i] for i in train_idx],
        [labels[i] for i in train_idx],
        esm_dir=esm_dir,
        device=device
    )
    val_data = prepare_maidta_data(
        [smiles_list[i] for i in val_idx],
        [seq_list[i] for i in val_idx],
        [labels[i] for i in val_idx],
        esm_dir=esm_dir,
        device=device
    )
    test_data = prepare_maidta_data(
        [smiles_list[i] for i in test_idx],
        [seq_list[i] for i in test_idx],
        [labels[i] for i in test_idx],
        esm_dir=esm_dir,
        device=device
    )

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    model = MAIDTA(config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.MSELoss()

    best_val_mse = float('inf')
    best_epoch = 0

    print("Starting training...")
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_mse, train_ci = train_epoch(model, train_loader, optimizer, criterion, device)
        val_labels, val_preds = eval_epoch(model, val_loader, device)
        val_mse = get_mse(val_labels, val_preds)
        val_ci = get_ci(val_labels, val_preds)

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train MSE: {train_mse:.4f}, Train CI: {train_ci:.4f}, Val MSE: {val_mse:.4f}, Val CI: {val_ci:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  -> New best model saved (Epoch {epoch}, Val MSE: {val_mse:.4f})")

    print(f"\nTraining complete. Best model at epoch {best_epoch} with Val MSE: {best_val_mse:.4f}")

    model.load_state_dict(torch.load('best_model.pt'))
    test_labels, test_preds = eval_epoch(model, test_loader, device)

    print("\n=== Test Results ===")
    print(f"MSE: {get_mse(test_labels, test_preds):.4f}")
    print(f"RMSE: {get_rmse(test_labels, test_preds):.4f}")
    print(f"CI: {get_ci(test_labels, test_preds):.4f}")
    print(f"Pearson: {get_pearson(test_labels, test_preds):.4f}")
    print(f"Spearman: {get_spearman(test_labels, test_preds):.4f}")
    print(f"R2: {get_r2(test_labels, test_preds):.4f}")
    print(f"Rm2: {get_rm2(test_labels, test_preds):.4f}")


if __name__ == "__main__":
    import os
    main()