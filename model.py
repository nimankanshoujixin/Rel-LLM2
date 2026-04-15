from typing import Any, Dict, List
import contextlib
import random
import copy
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleDict, Sigmoid, Sequential, Linear, Dropout, LayerNorm

import torch_frame.data
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from transformers import AutoModelForCausalLM, AutoTokenizer

from relbench.base import TaskType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch_frame.utils.infer_stype import infer_series_stype
from torch_frame import stype
from utils import get_label_texts, get_task_description, get_task_question, infer_class_labels, initialize_weights

# llama model type: https://huggingface.co/meta-llama
# encode special tokens for Llama 3.2: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
BOS = '<|begin_of_text|>'
EOS_USER = '<|eot_id|>'  # end of the message in a turn
EOS = '<|end_of_text|>'
IGNORE_INDEX = -100  # default = -100 in Pytorch CrossEntropyLoss, https://github.com/huggingface/transformers/issues/29819
accept_stypes = [stype.numerical, stype.categorical, stype.text_tokenized, stype.multicategorical, stype.text_embedded]   # no timestamp


class Model(torch.nn.Module):

    def __init__(self, data: HeteroData, col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]], num_layers: int, channels: int, out_channels: int, aggr: str,
                 norm: str = "batch_norm", dropout=0.0, shallow_list: List[NodeType] = [],  # List of node types to add shallow embeddings to input
                 id_awareness: bool = False, model_type: str = "meta-llama/Llama-3.2-1B", max_new_tokens=1, llm_frozen=False, output_mlp=False, output_probs=True, num_demo=4,
                 dataset=None, task=None, gamma=2.0, alpha=[1.0, 1.0], mask_ratio=0.5, pretrain_random_table=False, pretrain_mask_cell=True,
                 device: torch.device | None = None, basis_artifact: Dict[str, Any] | None = None, basis_tau: float = 0.07, basis_tau_res: float = 0.07,
                 basis_topk: int = 8, basis_residual_alpha: float = 0.1, basis_lambda_bce: float = 1.0, basis_lambda_ctr: float = 0.1,
                 basis_lambda_mgn: float = 0.1, basis_margin: float = 0.2):
        super().__init__()
        self.encoder = HeteroEncoder(channels=channels, node_to_col_names_dict={node_type: data[node_type].tf.col_names_dict for node_type in data.node_types},
                                     node_to_col_stats=col_stats_dict, )
        self.temporal_encoder = HeteroTemporalEncoder(node_types=[node_type for node_type in data.node_types if "time" in data[node_type]], channels=channels, )
        self.gnn = HeteroGraphSAGE(node_types=data.node_types, edge_types=data.edge_types, channels=channels, aggr=aggr, num_layers=num_layers)
        self.head = MLP(channels, out_channels=out_channels, norm=norm, num_layers=1, dropout=dropout)
        self.embedding_dict = ModuleDict({node: Embedding(data.num_nodes_dict[node], channels) for node in shallow_list})
        self.id_awareness_emb = Embedding(1, channels) if id_awareness else None
        self.output_mlp = output_mlp
        self.output_probs = output_probs
        self.gamma = gamma
        self.alpha = alpha
        self.uses_label_scorer = False
        self.uses_regression_head = False
        self.uses_direct_supervision = False
        self.uses_generation_supervision = False
        self.multiclass_label_text: list[str] = []
        self.multiclass_candidate_token_ids: list[list[int]] = []
        self.label_texts: list[str] = []
        self.label_token_ids: list[list[int]] = []
        self.sample_head = None
        self.label_head = None
        self.regression_head = None
        self.basis_enabled = False
        self.basis_topk = basis_topk
        self.basis_tau = basis_tau
        self.basis_tau_res = basis_tau_res
        self.basis_residual_alpha = basis_residual_alpha
        self.basis_lambda_bce = basis_lambda_bce
        self.basis_lambda_ctr = basis_lambda_ctr
        self.basis_lambda_mgn = basis_lambda_mgn
        self.basis_margin = basis_margin
        self.latest_align_loss = 0.0
        self.latest_align_components = {"bce": 0.0, "center": 0.0, "margin": 0.0}
        self.basis_ids: list[str] = []
        self.basis_types: list[str] = []
        self.basis_descs: list[str] = []
        self.basis_indices_by_table: dict[str, list[int]] = {}
        self.basis_indices_by_table_pair: dict[tuple[str, str], list[int]] = {}
        self.basis_indices_by_join_pair: dict[tuple[str, str], list[int]] = {}
        self.basis_indices_by_join_triplet: dict[tuple[str, str, str], list[int]] = {}

        # pretrain setup
        self.pretrain_mask_cell = pretrain_mask_cell
        self.pretrain_random_table = pretrain_random_table
        self.mask_ratio = mask_ratio
        self.mask_embed = Embedding(1, channels)
        self.column_keep = {}

        # https://huggingface.co/meta-llama/Llama-3.2-1B
        if model_type == 'gnn':
            self.model = None
            if not dist.is_initialized() or dist.get_rank() == 0:
                print('Using default GNNs without LLMs')
        else:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print('Loading LLAMA')
            self.num_demo = num_demo
            self.dataset = dataset
            self.task = task
            self.max_new_tokens = max_new_tokens  # only 1 number for classification but can be multiple for regression  # TODO: how many is the optimal?
            self.tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=False,
                                                           padding_side="left")  # https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
            self.tokenizer.pad_token = self.tokenizer.eos_token  # for padding, https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/36
            self.tokenizer.add_special_tokens({'mask_token': '<MASK>'})  # add masked token
            model = AutoModelForCausalLM.from_pretrained(model_type, torch_dtype=torch.float16, low_cpu_mem_usage=True)  # 16 instead of 32 with less memory!
            if device is not None:
                model = model.to(device)
            model.resize_token_embeddings(len(self.tokenizer))  # expand vocab due to '<MASK>', https://huggingface.co/docs/transformers/en/main_classes/tokenizer
            if llm_frozen:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("Freezing LLAMA!")
                for name, param in model.named_parameters():
                    param.requires_grad = False
            else:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("Training LLAMA with LORA!")  # TODO: use_dora=True
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)
                lora_r: int = 8
                lora_alpha: int = 16
                lora_dropout: float = 0.05
                lora_target_modules = ["q_proj", "v_proj", ]
                config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
                model = get_peft_model(model, config)

            self.model = model
            self.model_device = next(self.model.parameters()).device
            self.word_embedding = self.model.model.get_input_embeddings()
            out_dim = self.word_embedding.embedding_dim
            self.projector = Sequential(Linear(channels, 1024), Sigmoid(), Dropout(dropout), Linear(1024, out_dim), Dropout(dropout)).to(self.model_device)
            self.lm_head = MLP(out_dim, out_channels=out_channels, norm=norm, num_layers=1, dropout=dropout) if self.output_mlp else None
            if basis_artifact is not None:
                self._init_basis_alignment(basis_artifact, out_dim)
            self.uses_label_scorer = self.task.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTICLASS_CLASSIFICATION,
            ]
            self.uses_regression_head = self.task.task_type == TaskType.REGRESSION
            self.uses_direct_supervision = self.output_mlp or self.uses_label_scorer or self.uses_regression_head
            self.uses_generation_supervision = not self.uses_direct_supervision
            if self.uses_label_scorer:
                self.label_texts = get_label_texts(self.task)
                self.label_token_ids = [
                    self.tokenizer(label, add_special_tokens=False).input_ids
                    for label in self.label_texts
                ]
                self.sample_head = MLP(out_dim, out_channels=out_dim, norm=norm, num_layers=1, dropout=dropout).to(self.model_device)
                self.label_head = MLP(out_dim, out_channels=out_dim, norm=norm, num_layers=1, dropout=dropout).to(self.model_device)
            else:
                self.sample_head = None
                self.label_head = None
            self.regression_head = (
                MLP(out_dim, out_channels=1, norm=norm, num_layers=1, dropout=dropout).to(self.model_device)
                if self.uses_regression_head else None
            )

            # cached token embeddings
            self.bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model_device))
            self.pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model_device)).unsqueeze(0)
            if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                class_labels = infer_class_labels(self.task)
                if len(class_labels) != out_channels:
                    class_labels = list(range(out_channels))
                self.multiclass_label_text = [str(label) for label in class_labels]
                self.multiclass_candidate_token_ids = [
                    self.tokenizer(label, add_special_tokens=False).input_ids + self.eos_id_list
                    for label in self.multiclass_label_text
                ]

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()
        if self.model is not None:
            self.projector.apply(initialize_weights)
            if self.lm_head is not None:
                self.lm_head.reset_parameters()
            if self.sample_head is not None:
                self.sample_head.reset_parameters()
            if self.label_head is not None:
                self.label_head.reset_parameters()
            if self.regression_head is not None:
                self.regression_head.reset_parameters()
            if hasattr(self, "basis_query_norm"):
                self.basis_query_norm.reset_parameters()

    def _init_basis_alignment(self, basis_artifact: Dict[str, Any], out_dim: int) -> None:
        basis_vectors = basis_artifact["A_db"].to(dtype=torch.float32)
        if basis_vectors.dim() != 2 or basis_vectors.size(1) != out_dim:
            raise ValueError(
                f"Basis artifact dimension mismatch: expected [K, {out_dim}], got {tuple(basis_vectors.shape)}."
            )
        self.register_buffer("basis_vectors", basis_vectors, persistent=False)
        self.register_buffer("basis_norm", F.normalize(basis_vectors, dim=-1), persistent=False)
        self.basis_ids = list(basis_artifact["basis_ids"])
        self.basis_types = list(basis_artifact["basis_types"])
        self.basis_descs = list(basis_artifact["basis_descs"])
        (
            self.basis_indices_by_table,
            self.basis_indices_by_table_prefix,
            self.basis_indices_by_table_pair,
            self.basis_indices_by_join_pair,
            self.basis_indices_by_join_triplet,
        ) = self._build_basis_index_maps(self.basis_ids)
        self.basis_query_norm = LayerNorm(out_dim).to(self.model_device)
        self.basis_enabled = True

    @staticmethod
    def _canonical_pair(table_a: str, table_b: str) -> tuple[str, str]:
        return tuple(sorted((table_a, table_b)))

    @staticmethod
    def _canonical_triplet(table_a: str, table_b: str, table_c: str) -> tuple[str, str, str]:
        forward = (table_a, table_b, table_c)
        backward = (table_c, table_b, table_a)
        return min(forward, backward)

    def _build_basis_index_maps(
        self, basis_ids: list[str]
    ) -> tuple[
        dict[str, list[int]],
        dict[str, dict[str, list[int]]],
        dict[tuple[str, str], list[int]],
        dict[tuple[str, str], list[int]],
        dict[tuple[str, str, str], list[int]],
    ]:
        table_map: dict[str, list[int]] = defaultdict(list)
        table_prefix_map: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        table_pair_map: dict[tuple[str, str], list[int]] = defaultdict(list)
        join_pair_map: dict[tuple[str, str], list[int]] = defaultdict(list)
        join_triplet_map: dict[tuple[str, str, str], list[int]] = defaultdict(list)

        for idx, basis_id in enumerate(basis_ids):
            parts = basis_id.split("::")
            prefix = parts[0]
            if prefix == "table":
                table_map[parts[1]].append(idx)
                table_prefix_map[parts[1]][prefix].append(idx)
            elif prefix == "column":
                table_map[parts[1]].append(idx)
                table_prefix_map[parts[1]][prefix].append(idx)
            elif prefix == "pk":
                table_map[parts[1]].append(idx)
                table_prefix_map[parts[1]][prefix].append(idx)
            elif prefix == "stat":
                table_map[parts[1]].append(idx)
                table_prefix_map[parts[1]][prefix].append(idx)
            elif prefix == "fk":
                src_table, dst_table = parts[1], parts[3]
                table_pair_map[self._canonical_pair(src_table, dst_table)].append(idx)
            elif prefix == "join":
                if len(parts) == 3:
                    join_pair_map[self._canonical_pair(parts[1], parts[2])].append(idx)
                elif len(parts) == 4:
                    join_triplet_map[self._canonical_triplet(parts[1], parts[2], parts[3])].append(idx)

        return (
            dict(table_map),
            {table: {prefix: list(indices) for prefix, indices in prefix_map.items()} for table, prefix_map in table_prefix_map.items()},
            dict(table_pair_map),
            dict(join_pair_map),
            dict(join_triplet_map),
        )

    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def eos_user_id_list(self):
        return self.tokenizer(EOS_USER, add_special_tokens=False).input_ids

    @property
    def eos_id_list(self):
        return self.tokenizer(EOS, add_special_tokens=False).input_ids  # LLAMA tokenizer does not add an eos_token_id at the end of inputs

    @property
    def false_id(self):
        return self.tokenizer('No', add_special_tokens=False).input_ids[0]

    @property
    def true_id(self):
        return self.tokenizer('Yes', add_special_tokens=False).input_ids[0]

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast; if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    def encode(self, batch, entity_table):
        seed_time = batch[entity_table].seed_time  # seed time indicates at which time the target is to be predicted, filtering future data.
        batch_size = len(seed_time)
        x_dict = self.encoder(batch.tf_dict)  # encode interactions within each table (tensor_frame)

        # batch_dict -> index of each node in seed time (different from batch.batch!)
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)  # add time embedding to time-dependent node features
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        for node_type, embedding in self.embedding_dict.items():  # id embedding
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
        return x_dict, batch_size

    def column_filter(self, df, df_name):
        if df_name not in self.column_keep:
            self.column_keep[df_name] = [col for col in df.columns if infer_series_stype(df[col]) in accept_stypes]
        return self.column_keep[df_name]

    def pretrain(self, batch, entity_table):
        select_table = entity_table
        batch_size = len(batch[entity_table].seed_time)
        num_tokens_to_mask = int(batch_size * self.mask_ratio)  # Number of tokens to mask
        mask_indices = torch.randperm(batch_size)[:num_tokens_to_mask].to(self.device)
        if self.pretrain_mask_cell:
            select_column = random.choice([k for k, v in batch[entity_table].tf._col_to_stype_idx.items() if v[0] != stype.timestamp])  # exclude timestamp
            select_stype, select_idx = batch[entity_table].tf._col_to_stype_idx[select_column]
            select_feat = batch[entity_table].tf.feat_dict[select_stype]
            if isinstance(select_feat, torch_frame.data.MultiEmbeddingTensor):    # MultiEmbeddingTensor not support value setting...
                mask_values = select_feat.values.clone()
                offset = select_feat.offset
                mask_values[mask_indices, offset[select_idx]: offset[select_idx + 1]] = torch.zeros_like(mask_values[mask_indices, offset[select_idx]: offset[select_idx + 1]])
                batch[entity_table].tf.feat_dict[select_stype].values = mask_values
            elif isinstance(select_feat, torch.Tensor):
                batch[entity_table].tf.feat_dict[select_stype][mask_indices] = torch.zeros_like(select_feat[mask_indices])  # timestamp cannot be masked with 0 (min_year)
            x_dict, _ = self.encode(batch, entity_table)
        else:
            x_dict, _ = self.encode(batch, entity_table)
            if self.pretrain_random_table:
                select_table = random.choice([i for i in x_dict.keys() if x_dict[i].numel() > 0])  # random select a node type
            x_dict[select_table][mask_indices] = self.mask_embed.weight   # mask token embeddings

        x_dict = self.gnn(x_dict, batch.edge_index_dict)  # interactions among different tables
        node_embed = x_dict[select_table][:batch_size]
        node_embed = self.projector(node_embed)

        # Seed entity information
        seed_df_indices = batch[select_table].n_id[mask_indices].cpu().numpy()  # input_id -> the ID of the training table
        seed_df = batch[select_table].df.iloc[seed_df_indices]
        filtered_df = seed_df[self.column_filter(seed_df, select_table)]
        if self.pretrain_mask_cell: filtered_df = filtered_df[[select_column]]
        # print(filtered_df)
        # for col in seed_df.columns:   # TODO: check stype for other datasets
        #     print(infer_series_stype(seed_df[col]), ' : ', seed_df[col].iloc[0])

        batch_input_ids, batch_label_input_ids = [], []
        for index, row in filtered_df.iterrows():  # iterate each sample in the batch
            input_ids, label_input_ids = [], []
            row_dict = list(row.to_dict().items())
            random.shuffle(row_dict)
            for col_name, col_value in row_dict:  # todo: multiple columns?
                if col_value in ['\\N'] and not self.pretrain_mask_cell: continue  # filter no meaningful words
                other_values = [val for val in filtered_df[col_name].dropna().unique() if val != col_value and val != '\\N']
                if random.random() > 0.5 or len(other_values) < 1:
                    prompt = f'{col_name} is {col_value}.'
                    label_tokens = self.true_id
                else:
                    new_value = random.choice(other_values)
                    prompt = f'{col_name} is {new_value}.'
                    label_tokens = self.false_id

                input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids + self.eos_user_id_list
                label_input_ids += len(input_ids) * [IGNORE_INDEX] + [label_tokens] + self.eos_id_list
                break

            batch_input_ids.append(input_ids)
            batch_label_input_ids.append(label_input_ids)

        # tokenizer happens on CPU
        question = ' Question: Is the statement correct? Give Yes or No as answer.'
        question_embeds = self.word_embedding(self.tokenizer(question, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(num_tokens_to_mask):
            # Add bos & eos token: https://github.com/XiaoxinHe/G-Retriever/issues/17
            # print(self.tokenizer.decode(batch_input_ids[i]))
            inputs_embeds = self.word_embedding(torch.tensor(batch_input_ids[i]).to(self.device))
            inputs_embeds = torch.cat([self.bos_embeds, node_embed[mask_indices[i]].unsqueeze(0), question_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(num_tokens_to_mask):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([self.pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * (max_length - len(batch_label_input_ids[i])) + batch_label_input_ids[i]  # `inputs_embeds` contain `labels`

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.device)

        with self.maybe_autocast():
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True, labels=label_input_ids)
        return outputs.loss

    def label_tokenize(self, batch, entity_table):
        if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
            label = ['Yes' if i else 'No' for i in batch[entity_table].y.bool().tolist()]  # convert 0/1 to true/false
        elif self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            label = [str(i) for i in batch[entity_table].y.long().tolist()]
        elif self.task.task_type == TaskType.REGRESSION:
            label = [str(i) for i in batch[entity_table].y.float().tolist()]
        else:
            label = [str(i) for i in batch[entity_table].y.tolist()]
        labels = self.tokenizer(label, add_special_tokens=False)
        return labels

    def get_demo_info(self, demo_batch, entity_table):
        x_dict, demo_batch_size = self.encode(demo_batch, entity_table)
        assert self.num_demo <= demo_batch_size, 'Too large demo numbers!'
        x_dict = self.gnn(x_dict, demo_batch.edge_index_dict)
        demo_node_embeds = self.projector(x_dict[entity_table][:demo_batch_size])
        demo_label_ids = self.label_tokenize(demo_batch, entity_table).input_ids
        if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
            demo_labels = torch.tensor(demo_label_ids, device=self.device)
        else:
            demo_labels = [torch.tensor(label_ids, device=self.device) for label_ids in demo_label_ids]
        return demo_node_embeds, demo_labels

    def get_label_representations(self) -> Tensor:
        label_embeds = []
        for token_ids in self.label_token_ids:
            token_tensor = torch.tensor(token_ids, device=self.device, dtype=torch.long)
            token_embeds = self.word_embedding(token_tensor)
            label_embeds.append(token_embeds.mean(dim=0))
        label_embeds = torch.stack(label_embeds, dim=0)
        label_embeds = label_embeds.to(
            device=next(self.label_head.parameters()).device,
            dtype=next(self.label_head.parameters()).dtype,
        )
        label_embeds = self.label_head(label_embeds)
        return F.normalize(label_embeds, dim=-1)

    def encode_prompt_representation(self, inputs_embeds: Tensor, attention_mask: Tensor) -> Tensor:
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[-1]
        last_indices = attention_mask.sum(dim=1).long().clamp_min(1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        prompt_repr = hidden[batch_indices, last_indices]
        return prompt_repr

    def score_multiclass_candidates(self, inputs_embeds: Tensor, attention_mask: Tensor) -> Tensor:
        batch_size = inputs_embeds.size(0)
        with self.maybe_autocast():
            prompt_outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=True,
            )

        prompt_log_probs = torch.nn.functional.log_softmax(
            prompt_outputs.logits[:, -1, :].float(),
            dim=-1,
        )
        prompt_past_key_values = prompt_outputs.past_key_values
        scores = []

        for candidate_token_ids in self.multiclass_candidate_token_ids:
            candidate_ids = torch.tensor(
                candidate_token_ids,
                device=self.device,
                dtype=torch.long,
            )
            score = prompt_log_probs[:, candidate_ids[0]]
            token_count = 1

            if candidate_ids.numel() > 1:
                past_key_values = prompt_past_key_values
                current_attention_mask = attention_mask
                for token_idx in range(candidate_ids.numel() - 1):
                    current_token = candidate_ids[token_idx].view(1, 1).expand(batch_size, 1)
                    current_attention_mask = torch.cat(
                        [
                            current_attention_mask,
                            torch.ones(
                                (batch_size, 1),
                                device=self.device,
                                dtype=attention_mask.dtype,
                            ),
                        ],
                        dim=1,
                    )
                    with self.maybe_autocast():
                        step_outputs = self.model(
                            input_ids=current_token,
                            attention_mask=current_attention_mask,
                            past_key_values=past_key_values,
                            return_dict=True,
                            use_cache=True,
                        )
                    step_log_probs = torch.nn.functional.log_softmax(
                        step_outputs.logits[:, -1, :].float(),
                        dim=-1,
                    )
                    score = score + step_log_probs[:, candidate_ids[token_idx + 1]]
                    past_key_values = step_outputs.past_key_values
                    token_count += 1

            scores.append(score / token_count)

        return torch.stack(scores, dim=1)

    def recursive_sample(self, batch_data: HeteroData, node_type: str, target_nodes: torch.Tensor, num_hops: int = 2):
        """
        Recursively samples neighbors from a batch heterogeneous graph while ensuring previously sampled node types are excluded.
        Args:
            batch_data (HeteroData): A batched heterogeneous graph from PyG's NeighborLoader.
            target_nodes (torch.Tensor): The indices of the target nodes in the `node_type`.
            node_type (str): The node type of the target nodes (e.g., "entity_table").
            num_hops (int): Number of recursive hops to sample.
        """
        sampled_nodes = [node_type]  # Track sampled node types to avoid re-sampling
        neighbor_dict = {node_type: {node: {} for node in target_nodes.tolist()}}  # Initialize nested dictionary

        def sample_neighbors(current_nodes, current_node_type, depth, tmp_dict):
            """Recursively sample neighbors up to num_hops while avoiding duplicate node types."""
            if depth == num_hops: return
            next_nodes = {}
            for edge_type in batch_data.edge_types:  # Iterate through edge types to find valid neighbors
                src_type, _, dst_type = edge_type
                if src_type == current_node_type and dst_type not in sampled_nodes:
                    src_nodes = batch_data[edge_type].edge_index[0].tolist()
                    dst_nodes = batch_data[edge_type].edge_index[1].tolist()
                    # print(edge_type, current_node_type, len(src_nodes), len(dst_nodes))
                    for src, dst in zip(src_nodes, dst_nodes):
                        if src in current_nodes:  # Ensure it's a valid node from the current set
                            if dst_type not in tmp_dict[src]:
                                tmp_dict[src][dst_type] = {}
                            tmp_dict[src][dst_type][dst] = {}

                            if dst_type not in next_nodes:
                                next_nodes[dst_type] = set()
                            next_nodes[dst_type].add(dst)
            for node in tmp_dict.keys():
                for next_node_type, nodes in next_nodes.items():   # Recursive call for the next hop
                    if next_node_type in tmp_dict[node].keys():
                        child_nodes = set(tmp_dict[node][next_node_type].keys())
                        if child_nodes:
                            sample_neighbors(
                                child_nodes,
                                next_node_type,
                                depth + 1,
                                tmp_dict[node][next_node_type],
                            )

        sample_neighbors(set(target_nodes.tolist()), node_type, depth=0, tmp_dict=neighbor_dict[node_type])   # Start recursive sampling from target nodes
        return neighbor_dict

    def get_neighbor_embedding(self, neighbor_dict, embed_dict):

        def recursive_collect(node_type, node_id, sub_neighbors):
            """Recursively collect embeddings depth-first for a single node."""
            node_embedding = embed_dict[node_type][node_id].unsqueeze(0)  # Shape: (1, D)
            # Collect embeddings from deeper neighbors recursively
            neighbor_embeds = []
            for sub_type, sub_dict in sub_neighbors.items():
                for sub_id, sub_sub_neighbors in sub_dict.items():
                    neighbor_embeds.append(recursive_collect(sub_type, sub_id, sub_sub_neighbors))
            if neighbor_embeds:
                neighbor_embeds = torch.cat(neighbor_embeds)  # Concatenate along feature dimension
                node_embedding = torch.cat([node_embedding, neighbor_embeds])
            return node_embedding

        all_embeddings = []
        for target_type, targets in neighbor_dict.items():
            for target_id, neighbors in targets.items():
                all_embeddings.append(recursive_collect(target_type, target_id, neighbors))
        return torch.cat(all_embeddings) if all_embeddings else None

    def _add_basis_weight(
        self,
        target: dict[int, float],
        indices: list[int],
        weight: float,
    ) -> None:
        if weight <= 0.0:
            return
        for idx in indices:
            target[idx] = max(target.get(idx, 0.0), weight)

    def _add_table_basis_weights(
        self,
        target: dict[int, float],
        table_name: str,
        *,
        table_weight: float,
        column_weight: float,
        pk_weight: float,
        stat_weight: float,
    ) -> None:
        prefix_map = self.basis_indices_by_table_prefix.get(table_name, {})
        self._add_basis_weight(target, prefix_map.get("table", []), table_weight)
        self._add_basis_weight(target, prefix_map.get("column", []), column_weight)
        self._add_basis_weight(target, prefix_map.get("pk", []), pk_weight)
        self._add_basis_weight(target, prefix_map.get("stat", []), stat_weight)

    def _collect_basis_targets(
        self,
        entity_table: str,
        sample_neighbors: dict[str, dict],
    ) -> dict[int, float]:
        target_weights: dict[int, float] = {}
        self._add_table_basis_weights(
            target_weights,
            entity_table,
            table_weight=1.0,
            column_weight=0.30,
            pk_weight=0.35,
            stat_weight=0.10,
        )

        def traverse(current_table: str, subtree: dict[str, dict], path: list[str]) -> None:
            for next_table, node_dict in subtree.items():
                hop = len(path)
                if hop == 1:
                    self._add_table_basis_weights(
                        target_weights,
                        next_table,
                        table_weight=0.75,
                        column_weight=0.15,
                        pk_weight=0.20,
                        stat_weight=0.03,
                    )
                    pair_weight = 0.90
                    join_pair_weight = 0.80
                    triplet_weight = 0.0
                else:
                    self._add_table_basis_weights(
                        target_weights,
                        next_table,
                        table_weight=0.45,
                        column_weight=0.05,
                        pk_weight=0.08,
                        stat_weight=0.0,
                    )
                    pair_weight = 0.60
                    join_pair_weight = 0.45
                    triplet_weight = 0.35

                self._add_basis_weight(
                    target_weights,
                    self.basis_indices_by_table_pair.get(self._canonical_pair(current_table, next_table), []),
                    pair_weight,
                )
                self._add_basis_weight(
                    target_weights,
                    self.basis_indices_by_join_pair.get(self._canonical_pair(current_table, next_table), []),
                    join_pair_weight,
                )
                new_path = path + [next_table]
                if len(new_path) >= 3:
                    self._add_basis_weight(
                        target_weights,
                        self.basis_indices_by_join_triplet.get(
                            self._canonical_triplet(new_path[-3], new_path[-2], new_path[-1]),
                            [],
                        ),
                        triplet_weight,
                    )
                for child_subtree in node_dict.values():
                    traverse(next_table, child_subtree, new_path)

        traverse(entity_table, sample_neighbors, [entity_table])
        return target_weights

    def align_graph_prompts(
        self,
        graph_prompts: list[Tensor],
        entity_table: str,
        basis_neighbors: dict[int, dict[str, dict]],
    ) -> tuple[list[Tensor], Tensor]:
        if not self.basis_enabled or not graph_prompts:
            zero = self.basis_vectors.new_zeros(())
            self.latest_align_components = {"bce": 0.0, "center": 0.0, "margin": 0.0}
            return graph_prompts, zero

        pooled = torch.stack([prompt.mean(dim=0) for prompt in graph_prompts], dim=0)
        q = self.basis_query_norm(
            pooled.to(
                device=next(self.basis_query_norm.parameters()).device,
                dtype=next(self.basis_query_norm.parameters()).dtype,
            )
        ).float()
        q_norm = F.normalize(q, dim=-1)
        logits = torch.matmul(q_norm, self.basis_norm.t()) / self.basis_tau

        batch_size = len(graph_prompts)
        basis_targets = torch.zeros(
            (batch_size, self.basis_vectors.size(0)),
            device=logits.device,
            dtype=torch.float32,
        )
        pos_masks = torch.zeros_like(basis_targets, dtype=torch.bool)
        for sample_idx in range(batch_size):
            pos_targets = self._collect_basis_targets(entity_table, basis_neighbors[sample_idx])
            if not pos_targets:
                pos_targets = {
                    idx: 1.0 for idx in self.basis_indices_by_table_prefix.get(entity_table, {}).get("table", [])
                }
            if not pos_targets:
                pos_targets = {idx: 1.0 for idx in self.basis_indices_by_table.get(entity_table, [])}
            pos_indices = list(pos_targets.keys())
            basis_targets[sample_idx, pos_indices] = torch.tensor(
                [pos_targets[idx] for idx in pos_indices],
                device=logits.device,
                dtype=torch.float32,
            )
            pos_masks[sample_idx, pos_indices] = True

        loss_bce = F.binary_cross_entropy_with_logits(logits, basis_targets)

        pos_weights = basis_targets.clamp_min(0.0)
        pos_weight_sum = pos_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        c_pos = torch.matmul(pos_weights, self.basis_norm) / pos_weight_sum
        c_pos = F.normalize(c_pos, dim=-1)
        cos_pos = F.cosine_similarity(q_norm, c_pos, dim=-1)
        loss_center = (1.0 - cos_pos).mean()

        neg_logits = logits.masked_fill(pos_masks, float("-inf"))
        neg_idx = neg_logits.argmax(dim=1)
        a_neg = self.basis_norm[neg_idx]
        cos_neg = F.cosine_similarity(q_norm, a_neg, dim=-1)
        loss_margin = F.relu(self.basis_margin - cos_pos + cos_neg).mean()

        align_loss = (
            self.basis_lambda_bce * loss_bce
            + self.basis_lambda_ctr * loss_center
            + self.basis_lambda_mgn * loss_margin
        )
        self.latest_align_components = {
            "bce": float(loss_bce.detach().cpu()),
            "center": float(loss_center.detach().cpu()),
            "margin": float(loss_margin.detach().cpu()),
        }

        topk = min(self.basis_topk, self.basis_vectors.size(0))
        topk_vals, topk_idx = torch.topk(logits, k=topk, dim=-1)
        topk_weights = torch.softmax(topk_vals / self.basis_tau_res, dim=-1)
        residual = (
            self.basis_vectors[topk_idx].to(device=pooled.device, dtype=pooled.dtype)
            * topk_weights.unsqueeze(-1).to(dtype=pooled.dtype)
        ).sum(dim=1)

        aligned_prompts = [
            prompt + self.basis_residual_alpha * residual[i].unsqueeze(0).to(prompt.dtype)
            for i, prompt in enumerate(graph_prompts)
        ]
        return aligned_prompts, align_loss

    def forward(self, batch: HeteroData, entity_table: NodeType, context=True, demo_info=None, inference: bool = False, pretrain_mode: bool = False) -> Tensor:
        if self.model is not None:
            self.latest_align_loss = self.word_embedding.weight.new_zeros(())
            self.latest_align_components = {"bce": 0.0, "center": 0.0, "margin": 0.0}
        if pretrain_mode:
            return self.pretrain(batch, entity_table)
        x_dict, batch_size = self.encode(batch, entity_table)

        # num_sampled_nodes_dict ->  the number of sampled nodes for each node type at each layer (hop)
        """ {'user_friends': [0, 67636, 0], 'users': [512, 0, 2812], 'event_attendees': [0, 4751, 149], 'events': [0, 85, 4943], 'event_interest': [0, 224, 1]} """
        x_dict = self.gnn(x_dict, batch.edge_index_dict)  # interactions among different tables
        node_embed = x_dict[entity_table][:batch_size]
        if self.model is None: return self.head(node_embed)  # output prediction
        node_embed = self.projector(node_embed)

        # encode description, questions and labels   # TODO: pad at last/in the middle? pad id to like 0006086 of the same length?
        task_desc = get_task_description(self.dataset, self.task)
        question = ' Question: ' + get_task_question(self.dataset, self.task)
        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION and self.multiclass_label_text:
            if len(self.multiclass_label_text) <= 20:
                question += ' Valid class ids: ' + ', '.join(self.multiclass_label_text) + '.'
            else:
                question += (
                    f" Valid class ids range from {self.multiclass_label_text[0]} "
                    f"to {self.multiclass_label_text[-1]}."
                )
        question += ' Answer: '  # https://huggingface.co/docs/transformers/tasks/prompting
        task_descs = self.tokenizer(task_desc, add_special_tokens=False)
        questions = self.tokenizer(question, add_special_tokens=False)
        if not inference and self.uses_generation_supervision:
            labels = self.label_tokenize(batch, entity_table)
        neighbors = None
        basis_neighbors = None
        if context:
            neighbors = self.recursive_sample(batch, entity_table, torch.arange(batch_size), num_hops=1)
        if self.basis_enabled:
            basis_neighbors = self.recursive_sample(batch, entity_table, torch.arange(batch_size), num_hops=2)[entity_table]
        if self.num_demo > 0 and demo_info is not None:
            # construct in-context demos
            demo_node_embeds, demo_labels = demo_info
            if self.task.task_type == TaskType.BINARY_CLASSIFICATION:  # balanced sampling
                mask = demo_labels == demo_labels[0].item()
                indices_A = torch.where(mask)[0]  # Indices for class 0
                indices_B = torch.where(~mask)[0]  # Indices for class 1
                count_A = indices_A.size(0)
                count_B = indices_B.size(0)
                num_demo_half = self.num_demo // 2
                extra = self.num_demo % 2
                assert count_A >= num_demo_half + extra and count_B >= num_demo_half + extra, "Not enough samples in one class"
                sampled_A = indices_A[torch.randint(0, count_A, (batch_size, num_demo_half), device=self.device)]  # (B, K_half)
                sampled_B = indices_B[torch.randint(0, count_B, (batch_size, num_demo_half), device=self.device)]  # (B, K_half)
                if extra:  # if M is odd, randomly choose which class to take the extra from for each B'
                    extra_class = torch.randint(0, 2, (batch_size,), device=self.device)
                    extra_A = indices_A[torch.randint(0, count_A, (batch_size,), device=self.device)]  # (B,)
                    extra_B = indices_B[torch.randint(0, count_B, (batch_size,), device=self.device)]  # (B,)
                    extra_samples = torch.where(extra_class, extra_B, extra_A).unsqueeze(1)  # (B, 1)
                    sampled_indices = torch.cat([sampled_A, sampled_B, extra_samples], dim=1)  # (B, K_half*2 + 1)
                else:
                    sampled_indices = torch.cat([sampled_A, sampled_B], dim=1)  # (B, K)
                shuffle_idx = torch.rand(batch_size, self.num_demo, device=self.device).argsort(dim=1)  # shuffle the indices
                sampled_indices = sampled_indices.gather(1, shuffle_idx)
            else:
                random_matrix = torch.rand(batch_size, len(demo_node_embeds), device=self.device)  # (B, B')
                sampled_indices = random_matrix.argsort(dim=1)[:, :self.num_demo]  # (B, K)
            demo_node_embeds = demo_node_embeds[sampled_indices]  # (B, K, D)
            if isinstance(demo_labels, torch.Tensor):
                demo_labels = demo_labels[sampled_indices]
            else:
                demo_labels = [
                    [demo_labels[idx.item()] for idx in row]
                    for row in sampled_indices
                ]

        graph_prompts = []
        for i in range(batch_size):
            graph_prompt = node_embed[i].unsqueeze(0)
            if context and neighbors is not None:
                neighbor_embed = self.get_neighbor_embedding(neighbors[entity_table][i], x_dict)
                if neighbor_embed is not None:
                    neighbor_embed = self.projector(neighbor_embed)
                    graph_prompt = torch.cat([graph_prompt, neighbor_embed])
            graph_prompts.append(graph_prompt)

        if self.basis_enabled and basis_neighbors is not None:
            graph_prompts, self.latest_align_loss = self.align_graph_prompts(
                graph_prompts,
                entity_table,
                basis_neighbors,
            )

        # tokenizer happens on CPU
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):  # TODO: do not need iteration (simplified)
            # Add bos & eos token
            input_ids = task_descs.input_ids + questions.input_ids + self.eos_user_id_list
            if not inference and self.uses_generation_supervision:
                label_input_ids = labels.input_ids[i] + self.eos_id_list  # EOS ceases generation
                input_ids += label_input_ids

            # prioritize the entity details (which vary) over the static question.
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.device))
            if self.num_demo > 0 and demo_info is not None:
                demo_embeds = []
                for k in range(self.num_demo):
                    demo_label_ids = demo_labels[i][k]
                    demo_embeds += [demo_node_embeds[i][k].unsqueeze(0), self.word_embedding(demo_label_ids)]
                demo_embeds.append(node_embed[i].unsqueeze(0))  # append the seed entity at last
                inputs_embeds = torch.cat([inputs_embeds[:-1], torch.cat(demo_embeds), inputs_embeds[-1:]])
            inputs_embeds = torch.cat([self.bos_embeds, graph_prompts[i], inputs_embeds], dim=0)  # node embed after BOS

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            if not inference and self.uses_generation_supervision:
                label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
                batch_label_input_ids.append(label_input_ids)  # auto-regressive + teacher forcing, https://github.com/XiaoxinHe/G-Retriever/issues/17

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([self.pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            if not inference and self.uses_generation_supervision:
                batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]  # `inputs_embeds` contain `labels`

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)

        if self.output_mlp or self.uses_label_scorer or self.uses_regression_head:
            hidden = self.encode_prompt_representation(inputs_embeds, attention_mask)
            if self.uses_label_scorer:
                hidden = hidden.to(
                    device=next(self.sample_head.parameters()).device,
                    dtype=next(self.sample_head.parameters()).dtype,
                )
                sample_repr = self.sample_head(hidden)
                sample_repr = F.normalize(sample_repr, dim=-1)
                label_repr = self.get_label_representations()
                pred = torch.matmul(sample_repr, label_repr.t())
            elif self.uses_regression_head:
                hidden = hidden.to(
                    device=next(self.regression_head.parameters()).device,
                    dtype=next(self.regression_head.parameters()).dtype,
                )
                pred = self.regression_head(hidden)
            else:
                hidden = hidden.to(
                    device=next(self.lm_head.parameters()).device,
                    dtype=next(self.lm_head.parameters()).dtype,
                )
                pred = self.lm_head(hidden)
            if pred.dim() > 1 and pred.size(1) == 1:
                pred = pred.view(-1)
            return pred

        if not inference:
            #########################
            # Training
            #########################
            label_input_ids = torch.tensor(batch_label_input_ids).to(self.device)

            if self.task.task_type != TaskType.BINARY_CLASSIFICATION:
                with self.maybe_autocast():
                    outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True, labels=label_input_ids)
                return outputs.loss
            else:  # prevent over-fitting due to binary class imbalance
                with self.maybe_autocast():
                    outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
                # Shift so that tokens < n predict n, https://github.com/huggingface/transformers/issues/10480
                # https://discuss.huggingface.co/t/where-to-look-for-a-loss-definition-for-a-pretrained-model/26073/2
                logits = outputs.logits[..., :-1, :].contiguous()  # (B, L-1，C)
                labels = label_input_ids[..., 1:].contiguous()  # (B, L-1)
                valid_mask = (labels != IGNORE_INDEX)
                labels = labels[valid_mask]  # (2 * B), including the binary class + EOS
                logits = logits[valid_mask]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs = probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                focal_weight = (1 - probs).pow(self.gamma)
                loss = -focal_weight * probs.log()  # focal loss
                if self.alpha is not None:
                    class_weights = torch.ones(self.model.vocab_size).to(self.device)
                    class_weights[self.false_id] = self.alpha[0]
                    class_weights[self.true_id] = self.alpha[1]  # class weights
                    alpha_t = class_weights.gather(dim=0, index=labels)
                    loss = alpha_t * loss
                return loss.mean()

        #########################
        # Inference
        #########################
        if self.task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = self.score_multiclass_candidates(inputs_embeds, attention_mask)
        else:
            with self.maybe_autocast():
                outputs = self.model.generate(inputs_embeds=inputs_embeds, max_new_tokens=self.max_new_tokens, attention_mask=attention_mask, return_dict_in_generate=True,
                                              output_scores=True, use_cache=True,  # https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958
                                              pad_token_id=self.tokenizer.pad_token_id)  # suppress hf warning

        if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
            if self.output_probs:  # yes/no
                pred = outputs.scores[0][..., [self.false_id, self.true_id]]  # https://huggingface.co/docs/transformers/en/internal/generation_utils
                # print('before softmax:', pred)
                pred = torch.softmax(pred, dim=-1)[..., 1]  # output probs instead of 0/1, https://github.com/huggingface/transformers/issues/14498
                pred = torch.nan_to_num(pred, nan=0.5)
            else:
                seq = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                # print(seq)
                pred = torch.tensor([0.0 if i == 'No' else 1.0 for i in seq])
        elif self.task.task_type == TaskType.REGRESSION:
            seq = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            pred = []
            for i in seq:
                try:
                    pred.append(float(i))
                except ValueError:
                    pred.append(0.0)  # Skip invalid entries
            # print('Sequence: ', seq, 'Scores: ', pred)
            pred = torch.tensor(pred)
        return pred

    @staticmethod
    def focal_loss(logits, labels, gamma=2.0, alpha_weights=None):
        probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits to probabilities
        targets_one_hot = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()  # One-hot encoding
        ce_loss = -targets_one_hot * torch.log(probs)  # Cross-entropy loss
        loss = (1 - probs) ** gamma * ce_loss  # Apply focal scaling
        if alpha_weights is not None:
            loss *= alpha_weights  # Weight per class
        return loss.mean()

    def forward_dst_readout(self, batch: HeteroData, entity_table: NodeType, dst_table: NodeType, ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError("id_awareness must be set True to use forward_dst_readout")
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        return self.head(x_dict[dst_table])

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param
