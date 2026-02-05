import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_tabular_graph(
    csv_path,
    edge_path,
    id_col="user_id",
    label_col="label",
    sensitive_col=None,
    label_drop_value=None,
    label_threshold=None,
    sensitive_binarize=True,
    edge_sep="\t",
    standardize=True,
):
    nodes = pd.read_csv(csv_path)

    if label_col not in nodes.columns:
        raise ValueError(f"label_col '{label_col}' not found in {csv_path}")

    labels_raw = pd.to_numeric(nodes[label_col], errors="coerce")
    if label_drop_value is not None:
        nodes = nodes[labels_raw != label_drop_value].copy()
        labels_raw = pd.to_numeric(nodes[label_col], errors="coerce")

    if labels_raw.isna().any():
        nodes = nodes[labels_raw.notna()].copy()
        labels_raw = pd.to_numeric(nodes[label_col], errors="coerce")

    if label_threshold is not None:
        labels = (labels_raw > label_threshold).astype(int)
    else:
        labels = labels_raw.astype(int)

    sensitive = None
    if sensitive_col is not None:
        if sensitive_col not in nodes.columns:
            raise ValueError(f"sensitive_col '{sensitive_col}' not found in {csv_path}")
        sensitive_raw = pd.to_numeric(nodes[sensitive_col], errors="coerce").fillna(0)
        if sensitive_binarize:
            sensitive = (sensitive_raw > 0).astype(int)
        else:
            sensitive = sensitive_raw.astype(int)

    exclude_cols = {id_col, label_col}
    if sensitive_col:
        exclude_cols.add(sensitive_col)

    feature_cols = [
        c for c in nodes.columns
        if c not in exclude_cols
    ]

    if standardize:
        scaler = StandardScaler()
        features = scaler.fit_transform(nodes[feature_cols].values)
    else:
        features = nodes[feature_cols].values

    X = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels.values, dtype=torch.long)
    sensitive_tensor = None
    if sensitive is not None:
        sensitive_tensor = torch.tensor(sensitive.values, dtype=torch.long)

    edges = pd.read_csv(edge_path, sep=edge_sep, header=None)
    edges = edges.iloc[:, :2]
    edges.columns = ["src", "dst"]

    id_map = {
        uid: i for i, uid in enumerate(nodes[id_col].values)
    }

    src_list = []
    dst_list = []

    for src, dst in zip(edges["src"], edges["dst"]):
        if src in id_map and dst in id_map:
            src_list.append(id_map[src])
            dst_list.append(id_map[dst])

    edge_index = torch.tensor(
        [src_list, dst_list],
        dtype=torch.long
    )

    edge_index = torch.cat(
        [edge_index, edge_index.flip(0)],
        dim=1
    )

    data = Data(x=X, edge_index=edge_index, y=y)
    if sensitive_tensor is not None:
        data.sensitive = sensitive_tensor

    return data, feature_cols


def create_masks(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)

    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size + val_size]] = True
    test_mask[perm[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class SupervisedGCN(torch.nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        out = self.classifier(z)
        return out


@torch.no_grad()
def fairness_metrics(y_true, y_pred, sensitive):
    metrics = {}

    for s_val in [0, 1]:
        mask_s = (sensitive == s_val)

        if mask_s.sum() > 0:
            metrics[f"P_yhat1_s{s_val}"] = y_pred[mask_s].float().mean()
        else:
            metrics[f"P_yhat1_s{s_val}"] = torch.tensor(0.0)

        mask_y1_s = mask_s & (y_true == 1)
        if mask_y1_s.sum() > 0:
            metrics[f"TPR_s{s_val}"] = y_pred[mask_y1_s].float().mean()
        else:
            metrics[f"TPR_s{s_val}"] = torch.tensor(0.0)

    delta_sp = torch.abs(
        metrics["P_yhat1_s0"] - metrics["P_yhat1_s1"]
    )

    delta_eo = torch.abs(
        metrics["TPR_s0"] - metrics["TPR_s1"]
    )

    return delta_sp.item(), delta_eo.item()


def parse_args():
    parser = argparse.ArgumentParser(description="GCN baseline for tabular graphs")
    parser.add_argument("--csv", default="dataset/NBA/nba.csv")
    parser.add_argument("--edges", default="dataset/NBA/nba_relationship.txt")
    parser.add_argument("--id-col", default="user_id")
    parser.add_argument("--label-col", default="SALARY")
    parser.add_argument("--sensitive-col", default="country")
    parser.add_argument("--label-drop", default="-1")
    parser.add_argument("--label-threshold", default="0")
    parser.add_argument("--no-sensitive-binarize", action="store_true", default=False)
    parser.add_argument("--edge-sep", default="\\t")
    parser.add_argument("--no-standardize", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    label_threshold = None
    if str(args.label_threshold).lower() != "none":
        label_threshold = float(args.label_threshold)

    label_drop = None
    if str(args.label_drop).lower() != "none":
        label_drop = float(args.label_drop)

    sensitive_col = args.sensitive_col
    if str(sensitive_col).lower() == "none":
        sensitive_col = None

    data, feature_cols = load_tabular_graph(
        csv_path=args.csv,
        edge_path=args.edges,
        id_col=args.id_col,
        label_col=args.label_col,
        sensitive_col=sensitive_col,
        label_drop_value=label_drop,
        label_threshold=label_threshold,
        sensitive_binarize=not args.no_sensitive_binarize,
        edge_sep=args.edge_sep,
        standardize=not args.no_standardize,
    )

    train_mask, val_mask, test_mask = create_masks(
        num_nodes=data.num_nodes,
        seed=args.seed
    )

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    device = torch.device("cpu")

    encoder = GCNEncoder(
        in_channels=data.num_features,
        hidden_channels=args.hidden
    )

    model = SupervisedGCN(
        encoder=encoder,
        hidden_dim=args.hidden,
        num_classes=2
    ).to(device)

    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(mask):
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        acc = (pred[mask] == data.y[mask]).float().mean().item()

        delta_sp, delta_eo = fairness_metrics(
            y_true=data.y[mask],
            y_pred=pred[mask],
            sensitive=data.sensitive[mask]
        )

        return acc, delta_sp, delta_eo

    for epoch in range(1, args.epochs + 1):
        loss = train()

        if epoch % 20 == 0:
            train_acc, _, _ = evaluate(data.train_mask)
            val_acc, val_sp, val_eo = evaluate(data.val_mask)
            test_acc, test_sp, test_eo = evaluate(data.test_mask)

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f} | "
                f"Train {train_acc:.3f} | "
                f"Val {val_acc:.3f} | "
                f"Test {test_acc:.3f} | "
                f"ΔSP {test_sp:.3f} | "
                f"ΔEO {test_eo:.3f}"
            )

main()
