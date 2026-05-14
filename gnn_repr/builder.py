from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear
from torch_geometric.data import HeteroData

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE
from utils import initialize_weights


@dataclass(frozen=True)
class GNNRepresentationArtifact:
    dataset: str
    channels: int
    num_layers: int
    aggr: str
    train_steps: int
    graph_cache_profile: str
    downstream_graph_cache_profile: str
    mask_ratio: float
    edge_loss_weight: float
    recon_loss_weight: float
    encoder_state_dict: dict[str, Tensor]
    gnn_state_dict: dict[str, Tensor]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "channels": self.channels,
            "num_layers": self.num_layers,
            "aggr": self.aggr,
            "train_steps": self.train_steps,
            "graph_cache_profile": self.graph_cache_profile,
            "downstream_graph_cache_profile": self.downstream_graph_cache_profile,
            "mask_ratio": self.mask_ratio,
            "edge_loss_weight": self.edge_loss_weight,
            "recon_loss_weight": self.recon_loss_weight,
            "encoder_state_dict": self.encoder_state_dict,
            "gnn_state_dict": self.gnn_state_dict,
        }


class GNNRepresentationPretrainer(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict,
        *,
        channels: int,
        num_layers: int,
        aggr: str,
        dropout: float = 0.0,
        mask_ratio: float = 0.3,
        edge_loss_weight: float = 1.0,
        recon_loss_weight: float = 1.0,
        edge_samples_per_relation: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.mask_embedding = Embedding(1, channels)
        self.reconstruction_head = Linear(channels, channels)
        self.mask_ratio = float(mask_ratio)
        self.edge_loss_weight = float(edge_loss_weight)
        self.recon_loss_weight = float(recon_loss_weight)
        self.edge_samples_per_relation = int(edge_samples_per_relation)
        self.edge_type_to_idx = {
            edge_type: idx for idx, edge_type in enumerate(sorted(data.edge_types))
        }
        self.relation_embedding = Embedding(len(self.edge_type_to_idx), channels)
        self.dropout = float(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.gnn.reset_parameters()
        torch.nn.init.normal_(self.mask_embedding.weight, std=0.02)
        initialize_weights(self.reconstruction_head)
        torch.nn.init.normal_(self.relation_embedding.weight, std=0.02)

    def encode_masked_batch(
        self,
        batch: HeteroData,
        root_node_type: str,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        x_dict = self.encoder(batch.tf_dict)
        batch_size = int(batch[root_node_type].batch_size)
        if batch_size <= 0:
            raise RuntimeError(f"Empty root batch for node type {root_node_type}.")
        target_repr = x_dict[root_node_type][:batch_size].detach()
        num_mask = max(1, int(batch_size * self.mask_ratio))
        mask_indices = torch.randperm(batch_size, device=target_repr.device)[:num_mask]
        x_dict = {node_type: features.clone() for node_type, features in x_dict.items()}
        x_dict[root_node_type][mask_indices] = self.mask_embedding.weight[0]
        node_repr = self.gnn(x_dict, batch.edge_index_dict)
        return node_repr, target_repr, mask_indices

    def reconstruction_loss(
        self,
        node_repr: dict[str, Tensor],
        root_node_type: str,
        target_repr: Tensor,
        mask_indices: Tensor,
    ) -> Tensor:
        pred = self.reconstruction_head(node_repr[root_node_type][: target_repr.size(0)][mask_indices])
        target = target_repr[mask_indices]
        return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()

    def edge_prediction_loss(
        self,
        node_repr: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
    ) -> Tensor:
        losses: list[Tensor] = []
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.numel() == 0:
                continue
            src_type, _, dst_type = edge_type
            num_edges = edge_index.size(1)
            sample_count = min(num_edges, self.edge_samples_per_relation)
            perm = torch.randperm(num_edges, device=edge_index.device)[:sample_count]
            sampled_edge_index = edge_index[:, perm]
            src_idx, dst_idx = sampled_edge_index[0], sampled_edge_index[1]
            src_repr = node_repr[src_type][src_idx]
            dst_repr = node_repr[dst_type][dst_idx]
            rel_idx = self.edge_type_to_idx[edge_type]
            rel_repr = self.relation_embedding.weight[rel_idx].unsqueeze(0)
            pos_logits = (src_repr * rel_repr * dst_repr).sum(dim=-1)
            neg_dst_idx = torch.randint(
                low=0,
                high=node_repr[dst_type].size(0),
                size=(sample_count,),
                device=edge_index.device,
            )
            neg_repr = node_repr[dst_type][neg_dst_idx]
            neg_logits = (src_repr * rel_repr * neg_repr).sum(dim=-1)
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            losses.append(0.5 * (pos_loss + neg_loss))
        if not losses:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        return torch.stack(losses).mean()

    def forward(self, batch: HeteroData, root_node_type: str) -> dict[str, Tensor]:
        node_repr, target_repr, mask_indices = self.encode_masked_batch(batch, root_node_type)
        recon_loss = self.reconstruction_loss(node_repr, root_node_type, target_repr, mask_indices)
        edge_loss = self.edge_prediction_loss(node_repr, batch.edge_index_dict)
        total_loss = self.recon_loss_weight * recon_loss + self.edge_loss_weight * edge_loss
        return {
            "loss": total_loss,
            "recon_loss": recon_loss.detach(),
            "edge_loss": edge_loss.detach(),
        }

    def build_artifact(
        self,
        *,
        dataset: str,
        channels: int,
        num_layers: int,
        aggr: str,
        train_steps: int,
        graph_cache_profile: str,
        downstream_graph_cache_profile: str,
    ) -> GNNRepresentationArtifact:
        return GNNRepresentationArtifact(
            dataset=dataset,
            channels=channels,
            num_layers=num_layers,
            aggr=aggr,
            train_steps=int(train_steps),
            graph_cache_profile=graph_cache_profile,
            downstream_graph_cache_profile=downstream_graph_cache_profile,
            mask_ratio=self.mask_ratio,
            edge_loss_weight=self.edge_loss_weight,
            recon_loss_weight=self.recon_loss_weight,
            encoder_state_dict={k: v.detach().cpu() for k, v in self.encoder.state_dict().items()},
            gnn_state_dict={k: v.detach().cpu() for k, v in self.gnn.state_dict().items()},
        )

    @staticmethod
    def save_artifact(
        artifact: GNNRepresentationArtifact,
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact.to_dict(), path)
        return path
