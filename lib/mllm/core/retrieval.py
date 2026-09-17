import torch
import numpy as np
import json
import re
from .prompts import SQL_PROMPT, SQL_ANSWER_PROMPT, PRED_PROMPT, SQL_ANSWER_COUNT_PROMPT, REASONING_PROMPT
import logging

logger = logging.getLogger(__name__)

def compute_text_similarity(query_list, key_list, embedding_model, tokenizer, return_all=False):
    encoded_input = tokenizer(query_list + key_list, padding=True, truncation=True, return_tensors='pt')
    # Move tokenized inputs to the same device as the embedding model
    device = next(embedding_model.parameters()).device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = embedding_model(**encoded_input)
        embeddings = model_output[0][:, 0]
    query_emb = torch.nn.functional.normalize(embeddings[:len(query_list)], p=2, dim=1)
    key_emb = torch.nn.functional.normalize(embeddings[len(query_list):], p=2, dim=1)
    sims = query_emb @ key_emb.T
    if return_all:
        return sims
    else:
        return torch.mean(sims)

def node2indices(node_list, question_type, video_inputs, args):
    n_refine = 8 if question_type == "order" else args.n_refine
    sorted_node_list = sorted(map(int, node_list[:n_refine]))
    indices = []
    for idx in sorted_node_list:
        start_time = idx * args.chunk_size
        end_time = min((idx + 1) * args.chunk_size, len(video_inputs[0]))
        indices.extend(range(start_time, end_time))
    indices = set(indices)
    indices = sorted(indices)
    return indices, sorted_node_list

def allocate_node(args, video_graph, entity_graph, query_list, embedding_model, tokenizer, threshold=0.5):
    node_list = []
    for key in list(entity_graph.keys()):
        if compute_text_similarity(query_list, [key], embedding_model, tokenizer) > threshold:
            node_list.extend(entity_graph[key])
    for (node, data) in video_graph.nodes(data=True):
        if node in node_list:
            continue
        if data.get('captions') is None:
            key_list = data.get('entities', []) + data.get('actions', []) + data.get('scenes', []) + data.get('scenes', [])
        else:
            key_list = data.get('entities', []) + data.get('actions', []) + data.get('scenes', []) + data.get('scenes', []) + data.get('captions', [])
        if compute_text_similarity(query_list, key_list, embedding_model, tokenizer) > threshold:
            node_list.append(node)

    node_list = list(set(node_list))
    return node_list


# ---------------------------------------------------------------------------
# Batched / cached embedding helpers  (P1 optimisation)
# ---------------------------------------------------------------------------

def precompute_embeddings(texts, embedding_model, tokenizer):
    """Run a single forward pass and return a normalised embedding matrix.

    Parameters
    ----------
    texts : list[str]
        Strings to embed.
    embedding_model : transformers model
        BGE (or compatible) encoder.
    tokenizer : transformers tokenizer

    Returns
    -------
    torch.Tensor  – shape ``(len(texts), D)`` with L2-normalised rows.
        Returns ``None`` when *texts* is empty.
    """
    if not texts:
        return None
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    # Move tokenized inputs to the same device as the embedding model
    device = next(embedding_model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        out = embedding_model(**encoded)
        emb = out[0][:, 0]
    return torch.nn.functional.normalize(emb, p=2, dim=1)


def allocate_node_batched(
    args,
    video_graph,
    entity_graph,
    query_list,
    embedding_model,
    tokenizer,
    entity_key_embeddings=None,
    entity_keys=None,
    node_content_embeddings=None,
    node_ids=None,
    threshold=0.5,
):
    """Like ``allocate_node`` but uses pre-computed embeddings.

    If *entity_key_embeddings* / *node_content_embeddings* are ``None``
    they are computed on-the-fly (but still batched, so faster than the
    original per-key loop).
    """
    # -- query embeddings (computed once) ----------------------------------
    query_emb = precompute_embeddings(query_list, embedding_model, tokenizer)
    if query_emb is None:
        return []

    node_list = []

    # -- entity-graph matching --------------------------------------------
    if entity_keys is None:
        entity_keys = list(entity_graph.keys())
    if entity_key_embeddings is None and entity_keys:
        entity_key_embeddings = precompute_embeddings(
            entity_keys, embedding_model, tokenizer,
        )
    if entity_key_embeddings is not None and len(entity_keys) > 0:
        # (Q, D) @ (D, K) → (Q, K)  then mean over queries → (K,)
        sims = torch.mean(query_emb @ entity_key_embeddings.T, dim=0)
        for idx, key in enumerate(entity_keys):
            if sims[idx] > threshold:
                node_list.extend(entity_graph[key])

    # -- per-node content matching ----------------------------------------
    if node_ids is None:
        node_ids = list(video_graph.nodes)
    existing = set(node_list)
    candidate_ids = [n for n in node_ids if n not in existing]

    if node_content_embeddings is None and candidate_ids:
        # build content strings on-the-fly
        content_texts = []
        for nid in candidate_ids:
            data = video_graph.nodes[nid]
            parts = (
                data.get("entities", [])
                + data.get("actions", [])
                + data.get("scenes", [])
                + data.get("scenes", [])
            )
            if data.get("captions") is not None:
                parts += data.get("captions", [])
            content_texts.append("; ".join(parts) if parts else "")
        node_content_embeddings = precompute_embeddings(
            content_texts, embedding_model, tokenizer,
        )
        # filter the matching candidate_ids
        if node_content_embeddings is not None:
            sims = torch.mean(query_emb @ node_content_embeddings.T, dim=0)
            for i, nid in enumerate(candidate_ids):
                if sims[i] > threshold:
                    node_list.append(nid)
    elif node_content_embeddings is not None:
        # Use the pre-computed node embeddings (indexed same as node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        cand_indices = [id_to_idx[n] for n in candidate_ids if n in id_to_idx]
        if cand_indices:
            sub_emb = node_content_embeddings[cand_indices]
            sims = torch.mean(query_emb @ sub_emb.T, dim=0)
            for j, ci in enumerate(cand_indices):
                if sims[j] > threshold:
                    node_list.append(node_ids[ci])

    return list(set(node_list))

def extract_choices(question, candidates):
    if "(1)" in question or "(a)" in question:
        pattern = r'\(([a-zA-Z0-9]+)\)\s*(.+?)(?=\s*\([a-zA-Z0-9]+\)|$)'
        matches = re.findall(pattern, question, flags=re.DOTALL)
        query_list = [c[1].strip() for c in matches]
    elif re.search(r"\d+\.", question):
        pattern = r"\d+\.\s+([^\n]+)"
        matches = re.findall(pattern, question)
        query_list = [match.strip() for match in matches]
    elif "-->" in candidates[0]:
        choices = candidates[0].split("-->")
        query_list = [choice.strip() for choice in choices]
    elif len(candidates[0].split(",")) > 2:
        query_list = []
        for candidate in candidates:
            choices = re.sub(r'^[A-Za-z0-9]+\.\s*', '', candidate)
            choices = choices.rstrip('.')
            query_list.extend([item.strip().lower() for item in choices.split(',') if item.strip()])
        query_list = list(set(query_list))
    elif len(candidates[0].split(",")) > 1 and 'and' in candidates[0]:
        query_list = []
        for candidate in candidates:
            choices = re.sub(r'^[A-Za-z0-9]+\.\s*', '', candidate)
            choices = choices.rstrip('.')
            choices = choices.replace(' and ', ',')
            query_list.extend([item.strip().lower() for item in choices.split(',') if item.strip()])
        query_list = list(set(query_list))
    else:
        query_list = candidates
    return query_list

def count_and_sort_filtered(data):
    count_dict = {}
    for index, answers in data.items():
        if answers is not None and isinstance(answers, dict):
            for key, value in answers.items():
                if value != 'no' and value != '0' and value != 0:
                    count_dict[index] = count_dict.get(index, 0) + 1

    filtered_dict = {k: v for k, v in count_dict.items() if v > 0}

    sorted_indices = sorted(filtered_dict.keys(), key=lambda k: filtered_dict[k], reverse=True)

    return filtered_dict, sorted_indices