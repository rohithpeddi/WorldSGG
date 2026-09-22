import json
import os
import re
import ast
import numpy as np
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

# --- Limit CPU threads to avoid contention when multiple processes
#     share the same node (e.g. 8 vLLM workers on 8 GPUs). ----------
_CPU_CORES = os.cpu_count() or 1
_FAIR_THREADS = max(1, _CPU_CORES // max(int(os.environ.get("WORLD_SIZE", "8")), 1))
torch.set_num_threads(_FAIR_THREADS)

from .prompts import *
from .retrieval import (
    compute_text_similarity, extract_choices, allocate_node, node2indices,
    count_and_sort_filtered, precompute_embeddings, allocate_node_batched,
)
from .config_loader import (
    load_config, resolve_model_path, import_model_class,
    get_model_registry, resolve_embedding_model_path,
)
import logging

logger = logging.getLogger(__name__)

class Vgent():
    def __init__(self, args):
        self.args = args
        cfg = load_config()
        registry = get_model_registry(cfg)
        model_key = self.args.model_name

        if model_key not in registry:
            raise ValueError(
                f"Model '{model_key}' not in config.yaml. "
                f"Available: {list(registry.keys())}"
            )

        # Resolve model path: local weights dir → HuggingFace ID
        # CLI --model_weights_dir overrides config.yaml paths.model_weights
        model_weights_dir = getattr(self.args, 'model_weights_dir', None)
        if model_weights_dir:
            import os
            local_path = os.path.join(model_weights_dir, model_key)
            if os.path.isdir(local_path):
                logger.info(f"Using local weights (CLI): {local_path}")
                model_path = local_path
            else:
                logger.warning(
                    f"Local weights not found at {local_path}, "
                    f"falling back to config resolution"
                )
                model_path = resolve_model_path(cfg, model_key)
        else:
            model_path = resolve_model_path(cfg, model_key)

        # Dynamically import the model class
        model_class = import_model_class(cfg, model_key)

        use_vllm = getattr(self.args, 'use_vllm', True)
        logger.info(f"Loading model '{model_key}' ({'vLLM' if use_vllm else 'direct HuggingFace'} mode)")
        self.model = model_class(model_path, args)

        # Expose load_video for external calls (like in vgent_graph.py)
        # We bind the method to the instance so self is handled correctly
        self.load_video = self.model.load_video
        self.image_processor = self.model.image_processor # For preprocessing compatibility

        # Resolve embedding model path (local weights → HF ID fallback)
        embedding_model_path = resolve_embedding_model_path(cfg, "bge_large")
        logger.info(f"Loading embedding model from: {embedding_model_path}")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)

        # Move the BGE embedding model to GPU if available.
        # BGE-large is only ~670 MB — fits easily alongside vLLM even at
        # 0.95 gpu_memory_utilization on 80 GB cards.  Running on GPU
        # avoids CPU-thread contention when 8 processes share a node.
        if torch.cuda.is_available():
            self._emb_device = torch.device("cuda")
        else:
            self._emb_device = torch.device("cpu")
        self.embedding_model = AutoModel.from_pretrained(
            embedding_model_path,
        ).to(self._emb_device).eval()
        logger.info(
            f"Embedding model on {self._emb_device} "
            f"(CPU threads limited to {_FAIR_THREADS})"
        )
    
    def generate_entities(self, prompt, video_input, max_new_tokens=512):
        attempts = 0
        while attempts < 5:
            try:
                response = self.model.mllm_response(prompt, [video_input], max_new_tokens=max_new_tokens)
                response = strip_thinking_tags(response)
                info = json.loads(response.replace("```json", "").replace("```","").strip())
                
                entities = [f"{entity['entity name']}, {entity['description']}" 
                            for entity in info.get("entities", []) 
                            if "entity name" in entity and "description" in entity]
                
                actions = [f"{entity['entity name']}, {entity['action description']}" 
                        for entity in info.get("actions", []) 
                        if "entity name" in entity and "action description" in entity]
                
                scenes = [scene["location"] for scene in info.get("scenes", []) if "location" in scene]
                
                return entities, actions, scenes
            
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                attempts += 1
        
        return [], [], []
    
    def construct_graph(self, video_inputs, captions):
        # video_inputs is a list of chunks, but in vgent_graph it seems passed as [video]
        # construct_graph expects 'video_inputs' to be the full video tensor to be split?
        # In original code: split_video_inputs = torch.split(video_inputs[0], self.args.chunk_size, dim=0)
        # So video_inputs is [full_video_tensor]
        
        if not video_inputs or len(video_inputs) == 0 or video_inputs[0] is None:
             logger.warning("Empty video_inputs provided to construct_graph")
             return nx.DiGraph(), defaultdict(set)

        split_video_inputs = torch.split(video_inputs[0], self.args.chunk_size, dim=0)
        video_graph = nx.DiGraph()
        entity_graph = defaultdict(set)
        for idx, video_input in enumerate(tqdm(split_video_inputs, desc="Processing chunks", leave=False)):
            if video_input.shape[0] > 30:
                indices = torch.linspace(0, video_input.shape[0] - 1, 30).long()
                video_input = video_input[indices]
                sampled_indices = indices.tolist()
            else:
                sampled_indices = list(range(video_input.shape[0]))

            # Include caption context to ground entity extraction
            current_captions = self._get_current_captions(captions, idx)
            if current_captions:
                caption_text = " ".join(current_captions)
                grounded_prompt = (
                    f"Context: The following caption describes this video clip: "
                    f"\"{caption_text}\"\n\n{GRAPH_PROMPT}"
                )
            else:
                grounded_prompt = GRAPH_PROMPT

            entities, actions, scenes = self.generate_entities(grounded_prompt, video_input, max_new_tokens=512)
            
            video_graph.add_node(idx, actions=actions, scenes=scenes, entities=entities, captions=current_captions, sampled_indices=sampled_indices)
            self._update_entity_graph(entity_graph, video_graph, idx, entities, actions, scenes)
            
        return video_graph, entity_graph

    def _get_current_captions(self, captions, idx):
        if captions is not None:
            start_time = idx * self.args.chunk_size // self.args.fps
            end_time = (idx + 1) * self.args.chunk_size // self.args.fps
            return [text for time, text in captions if time >= start_time and time < end_time]
        return None

    def _update_entity_graph(self, entity_graph, video_graph, idx, entities, actions, scenes):
        """Update entity_graph by matching new entities to existing keys.

        **Optimised**: all new entities are embedded in a single batched
        forward pass and compared against the existing key embeddings via
        matrix multiplication, replacing the previous per-entity loop that
        ran a separate forward pass for every entity.
        """
        all_items = entities + actions + scenes
        if not all_items:
            return

        entity_names = [item.split(',')[0].lower() for item in all_items]

        # --- Batch-embed new items (query + key-name domains) ---------------
        new_emb = precompute_embeddings(
            all_items, self.embedding_model, self.embedding_tokenizer,
        )
        name_emb = precompute_embeddings(
            entity_names, self.embedding_model, self.embedding_tokenizer,
        )

        if new_emb is None or name_emb is None:
            # Fallback: just add them all as new keys
            for name in entity_names:
                entity_graph[name].add(idx)
            return

        # --- Embed existing keys (may be empty on the first chunk) ----------
        existing_keys = list(entity_graph.keys())
        existing_keys_set = set(existing_keys)

        if existing_keys:
            key_emb = precompute_embeddings(
                existing_keys, self.embedding_model, self.embedding_tokenizer,
            )
            if key_emb is None:
                for name in entity_names:
                    entity_graph[name].add(idx)
                return
            # (N_new, D) @ (D, N_existing) → (N_new, N_existing)
            sims = new_emb @ key_emb.T
        else:
            # First chunk: start with empty tensors so the loop below
            # handles intra-chunk matching naturally (P2 fix).
            emb_dim = new_emb.shape[1]
            key_emb = new_emb.new_empty(0, emb_dim)   # (0, D)
            sims = new_emb.new_empty(len(all_items), 0)  # (N_new, 0)

        # --- Sequential scan (preserves original ordering semantics) --------
        for i, entity in enumerate(all_items):
            entity_name = entity_names[i]

            matched = False
            if sims.shape[1] > 0:
                max_sim, max_sim_idx = sims[i].max(dim=0)
                if max_sim.item() > 0.7:
                    most_similar_entity = existing_keys[max_sim_idx.item()]
                    entity_graph[most_similar_entity].add(idx)
                    video_graph.add_edges_from(
                        (idx, j, {"label": most_similar_entity})
                        for j in entity_graph[most_similar_entity]
                    )
                    matched = True

            if not matched:
                entity_graph[entity_name].add(idx)
                # Only expand the candidate tensor when this key name is
                # truly novel — avoids duplicate columns (P3 fix).
                if entity_name not in existing_keys_set:
                    existing_keys.append(entity_name)
                    existing_keys_set.add(entity_name)
                    key_emb = torch.cat([key_emb, name_emb[i:i+1]], dim=0)
                    sims = torch.cat(
                        [sims, (new_emb @ name_emb[i:i+1].T)], dim=1,
                    )

    def extract_keywords(self, question, candidates, video_inputs):
        reason_prompt = REASONING_PROMPT.format(query=question, candidates=candidates)
        flag = True
        count = 0
        llm_info = None
        while flag and count < 5:
            try:
                response = self.model.mllm_response(reason_prompt, None, max_new_tokens=256)
                llm_info = json.loads(response.replace("```json", "").replace("```","").strip())
                flag = False
            except:
                count += 1
                continue
        
        query_list = llm_info["keywords"] if llm_info is not None and "keywords" in llm_info else []
        query_list = query_list + candidates
        query_list = list(set(query_list))

        return query_list, llm_info
    
    def retrieve_nodes(self, question, query_list, video_inputs, candidates, video_graph, entity_graph, captions, llm_info):
        indices = None
        # Logic for caption specific retrieval
        if "caption" in question.lower() and captions is not None and re.findall(r"'((?:[^']|(?<=\w)'(?=\w))*)'", question):
            query_caption = re.findall(r"'((?:[^']|(?<=\w)'(?=\w))*)'", question)
            indices = []
            for time, text in captions:
                if text in query_caption:
                    indices.append(time)
            node_list = []
        elif 'beginning' in question.lower() or 'at the start of' in question.lower():
            node_list = [i for i in range(3)]
        elif 'at the end of the video' in question.lower():
            # Calculate total chunks
            if not video_inputs or len(video_inputs) == 0 or video_inputs[0] is None:
                 return {"nodes": [], "indices": None}
                 
            total_chunks = round(np.ceil(len(video_inputs[0]) / self.args.chunk_size))
            node_list = [i for i in range(max(total_chunks - 3, 0), total_chunks)]
        elif video_graph is None:
            if not video_inputs or len(video_inputs) == 0 or video_inputs[0] is None:
                 return {"nodes": [], "indices": None}
                 
            total_chunks = round(np.ceil(len(video_inputs[0]) / self.args.chunk_size))
            node_list = list(range(total_chunks)) if (llm_info is not None and "tool" in llm_info and llm_info["tool"] in ["action counting", "order"] and self.args.task == 'mlvu') or len(video_inputs[0]) <= 128 else []
        else:
            if "order" in question.lower():
                query_list = extract_choices(question, candidates)
            query_list.append(question)
            node_list = allocate_node(self.args, video_graph, entity_graph, query_list, self.embedding_model, self.embedding_tokenizer)
            
            # Re-rank nodes based on text similarity
            key_list = []
            for node_id in node_list:
                node_data = video_graph.nodes[node_id]
                content = node_data.get('entities', []) + node_data.get('actions', []) + node_data.get('scenes', [])
                if node_data.get('captions') is not None:
                     content += node_data.get('captions', [])
                key_list.append("; ".join(content))
            
            sims = compute_text_similarity(query_list, key_list, self.embedding_model, self.embedding_tokenizer, return_all=True)
            sorted_indices = torch.argsort(torch.mean(sims, dim=0), descending=True)
            node_list = [node_list[i] for i in sorted_indices]
            
        return {"nodes": node_list[:self.args.n_retrieval], "indices": indices}
    
    def refine_nodes(self, retrieved_node_list, question, llm_info, candidates, video_inputs, captions, size_list=None):
        if len(retrieved_node_list["nodes"]) == 0:
            return retrieved_node_list, None, None
            
        info, obj_count = self._generate_refine_questions(question, llm_info, candidates)
        if info is None:
             return retrieved_node_list, None, None

        check_result = self._check_nodes_with_mllm(retrieved_node_list["nodes"], info, obj_count, candidates, video_inputs, captions, size_list)
        
        count_dict, sorted_nodes = count_and_sort_filtered(check_result)
        retrieved_node_list["nodes"] = sorted_nodes
        return retrieved_node_list, info, check_result

    def _generate_refine_questions(self, question, llm_info, candidates):
        input_candidates = " ".join(candidates)
        question_type = llm_info["tool"] if llm_info is not None and "tool" in llm_info else None
        prompt = SQL_PROMPT.format(query=question, candidates=input_candidates)
        info = None
        obj_count = False
        
        if question_type == "order" or "order" in question.lower():
            choices = extract_choices(question, candidates)
            info = {f"Q{choices.index(choice) + 1}": f"Is '{choice.lower()}' shown in video?" for choice in choices}
        elif question_type == "action counting" and 'action' in question.lower():
            try:
                match = re.search(r"'(.*?)'", question)
                extracted_text = match.group(1)
                info = {"Q1": f"Is there a scene featuring the '{extracted_text}' action in the video?"}
            except:
                info = None
        elif question_type == "object counting" or 'how many' in question.lower():
            obj_count = True
            info = {"Q1": question}
        
        if info is None:
             # Fallback to LLM generation
             attempts = 0
             while attempts < 5:
                try:
                    response = self.model.mllm_response(prompt, None, 512)
                    info = {}
                    info.update(json.loads(response.replace("```json", "").replace("```","").strip()))
                    break
                except:
                    attempts += 1
        return info, obj_count

    def _check_nodes_with_mllm(self, nodes, info, obj_count, candidates, video_inputs, captions, size_list):
        if not video_inputs or len(video_inputs) == 0 or video_inputs[0] is None:
             return {}
        
        split_video_inputs = torch.split(video_inputs[0], self.args.chunk_size, dim=0)
        split_size_list = torch.split(size_list, self.args.chunk_size, dim=0) if size_list is not None else None
        check_result = {}
        
        for node in nodes:
            if node >= len(split_video_inputs):
                continue
            video_input = split_video_inputs[node]
            size_list_input = split_size_list[node] if split_size_list is not None else None
            
            caption_prompt = ""
            if captions is not None:
                start_time = node * self.args.chunk_size // self.args.fps
                end_time = (node + 1) * self.args.chunk_size // self.args.fps
                select_captions = [text for time, text in captions if time >= start_time and time < end_time]
                caption_prompt = " This video's captions are listed below:\n" + " ".join(select_captions) + "\n"
            
            instruct = SQL_ANSWER_COUNT_PROMPT.format(questions=info) + caption_prompt if obj_count else SQL_ANSWER_PROMPT.format(questions=info) + caption_prompt
            
            try:
                output_text = self.model.mllm_response(instruct, [video_input], max_new_tokens=256, size_list=size_list_input)
                pred = json.loads(output_text.replace("```json", "").replace("```","").strip())
            except:
                pred = None
            check_result[node] = pred
        return check_result

    def aggregate_nodes(self, refined_node_list, llm_info, video_inputs, raw_video, size_list, captions, prompt, query, video_graph, sql_check, check_result, fps):
        question_type = llm_info["tool"] if llm_info is not None and "tool" in llm_info else None
        
        video_segments, select_captions = self._prepare_aggregation_inputs(refined_node_list, question_type, video_inputs, size_list, captions, video_graph)
        
        if select_captions:
             prompt = "This video's captions are listed below:\n" + " ".join(select_captions) + "\n" + prompt

        if self._is_simple_action_counting(question_type, refined_node_list, query):
             return self._handle_simple_action_counting(query, refined_node_list)

        agg_info = self._get_aggregation_info(sql_check, check_result, refined_node_list, llm_info, question_type, query)
        prompt += PRED_PROMPT + (agg_info if agg_info else "")
        
        # Handle dynamic resolution for qwenvl (or similar models) if needed
        # In original code this was specific to 'qwenvl' string check. 
        # Here we can assume specific handling might be needed or generic resize
        if "qwenvl" in self.args.model_name and refined_node_list["nodes"]:
             # This logic was specific to how QwenVL acts with 'raw_video' vs 'video_inputs'
             # For now, let's stick to using what we prepared as video_segments
             pass

        output_text = self.model.mllm_response(prompt, [video_segments], max_new_tokens=10, fps=fps)
        pred_answer = output_text.strip("()").strip()
        
        if pred_answer in query['letters']:
            return query['letters'][query['letters'].index(pred_answer)]
        else:
            return query['letters'][2] # Default fallback?

    def _prepare_aggregation_inputs(self, refined_node_list, question_type, video_inputs, size_list, captions, video_graph):
        node_list = refined_node_list["nodes"]
        indices = None
        select_captions = None
        video_segments = None

        if node_list and len(node_list) > 0:
            indices, sorted_node_list = node2indices(node_list, question_type, video_inputs, self.args)
            video_segments = video_inputs[0][indices]
            if captions:
                 if video_graph is None:
                     select_captions = []
                     for node_id in sorted_node_list:
                         select_captions.extend([text for time, text in captions if time >= node_id * self.args.chunk_size and time < (node_id + 1) * self.args.chunk_size])
                 else:
                     select_captions = [text for time, text in captions]
        
        elif refined_node_list["indices"] is not None:
             indices = refined_node_list["indices"]
             if captions:
                 select_captions = []
                 extend_indices = []
                 for index in indices:
                     select_captions.extend([text for time, text in captions if time == index])
                     extend_indices.extend(list(range(max(0, index - 10), min(len(video_inputs[0]) - 1, index + 10))))
                 indices = sorted(set(extend_indices))
             video_segments = video_inputs[0][indices]

        else:
             indices = np.linspace(0, len(video_inputs[0]) - 1, min(self.args.uniform_frame, len(video_inputs[0])), dtype=int)
             video_segments = video_inputs[0][indices]
             if captions:
                 select_captions = [text for time, text in captions]

        return video_segments, select_captions

    def _is_simple_action_counting(self, question_type, refined_node_list, query):
        return (question_type == "action counting" and 
                refined_node_list["nodes"] is not None and 
                (None not in [re.search(r'\d+', c) for c in query['candidates']]))

    def _handle_simple_action_counting(self, query, refined_node_list):
        numbers = [int(re.search(r'\d+', c).group()) for c in query['candidates']]
        pred_idx = min(range(len(numbers)), key=lambda i: abs(numbers[i] - len(refined_node_list["nodes"])))
        return query['letters'][pred_idx]

    def _get_aggregation_info(self, sql_check, check_result, refined_node_list, llm_info, question_type, query):
        multiple = llm_info["multiple"] if llm_info is not None and "multiple" in llm_info else 'no'
        if (sql_check is not None and check_result is not None and 
            refined_node_list["nodes"] is not None and len(refined_node_list["nodes"]) > 1 and 
            multiple == "yes" and question_type != "object counting"):
            
            input_text = ""
            for key, value in check_result.items():
                input_text += f"video [{key}]:\n"
                for question_id, question in sql_check.items():
                    if key in check_result and check_result[key] is not None and question_id in check_result[key] and check_result[key][question_id] != 'no':
                        input_text += f"{question}: {check_result[key][question_id]}\n"
            
            input_candidates = " ".join(query['candidates'])
            input_text = AGGREGATE_PROMPT.format(query=query['question'], candidates=input_candidates, input=input_text)
            try:
                msg = self.model.mllm_response(input_text, None, 128)
                return msg
            except:
                return None
        return None

    # ===================================================================
    # Fast-path helpers (used by --fast mode in process_ag_rag.py)
    # ===================================================================

    def precompute_graph_embeddings(self, video_graph, entity_graph):
        """Pre-compute embeddings for all entity keys and graph-node
        content strings.  Returns a cache dict that can be passed to
        ``retrieve_nodes_with_cache`` repeatedly."""
        entity_keys = list(entity_graph.keys())
        entity_key_emb = precompute_embeddings(
            entity_keys, self.embedding_model, self.embedding_tokenizer,
        )

        node_ids = list(video_graph.nodes)
        content_texts = []
        for nid in node_ids:
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
        node_content_emb = precompute_embeddings(
            content_texts, self.embedding_model, self.embedding_tokenizer,
        )

        return {
            "entity_keys": entity_keys,
            "entity_key_embeddings": entity_key_emb,
            "node_ids": node_ids,
            "node_content_embeddings": node_content_emb,
        }

    def batch_extract_keywords(self, prompts, candidates_list=None):
        """Extract keywords for *all* prompts in one batched LLM call.

        Parameters
        ----------
        prompts : list[str]
        candidates_list : list[list[str]] | None
            Per-prompt candidate lists (may be empty lists).

        Returns
        -------
        list[(query_list, llm_info)]
        """
        if candidates_list is None:
            candidates_list = [[] for _ in prompts]

        reason_prompts = [
            {"text": REASONING_PROMPT.format(query=p, candidates=c), "max_new_tokens": 256}
            for p, c in zip(prompts, candidates_list)
        ]
        responses = self.model.mllm_batch_response(reason_prompts)

        results = []
        for resp, cands in zip(responses, candidates_list):
            llm_info = None
            try:
                llm_info = json.loads(
                    resp.replace("```json", "").replace("```", "").strip()
                )
            except (json.JSONDecodeError, TypeError):
                pass
            qlist = (
                llm_info["keywords"]
                if llm_info is not None and "keywords" in llm_info
                else []
            )
            qlist = list(set(qlist + cands))
            results.append((qlist, llm_info))
        return results

    def retrieve_nodes_with_cache(
        self,
        question,
        query_list,
        video_inputs,
        candidates,
        video_graph,
        entity_graph,
        captions,
        llm_info,
        embedding_cache=None,
    ):
        """Same logic as ``retrieve_nodes`` but uses the pre-computed
        *embedding_cache* (from ``precompute_graph_embeddings``) to avoid
        redundant forward passes through the embedding model."""
        indices = None

        if (
            "caption" in question.lower()
            and captions is not None
            and re.findall(r"'((?:[^']|(?<=\w)'(?=\w))*)'", question)
        ):
            query_caption = re.findall(
                r"'((?:[^']|(?<=\w)'(?=\w))*)'", question
            )
            indices = [time for time, text in captions if text in query_caption]
            node_list = []
        elif "beginning" in question.lower() or "at the start of" in question.lower():
            node_list = [i for i in range(3)]
        elif "at the end of the video" in question.lower():
            if not video_inputs or len(video_inputs) == 0 or video_inputs[0] is None:
                return {"nodes": [], "indices": None}
            total_chunks = round(np.ceil(len(video_inputs[0]) / self.args.chunk_size))
            node_list = list(range(max(total_chunks - 3, 0), total_chunks))
        elif video_graph is None:
            if not video_inputs or len(video_inputs) == 0 or video_inputs[0] is None:
                return {"nodes": [], "indices": None}
            total_chunks = round(np.ceil(len(video_inputs[0]) / self.args.chunk_size))
            node_list = (
                list(range(total_chunks))
                if (
                    llm_info is not None
                    and "tool" in llm_info
                    and llm_info["tool"] in ["action counting", "order"]
                    and getattr(self.args, "task", None) == "mlvu"
                )
                or len(video_inputs[0]) <= 128
                else []
            )
        else:
            if "order" in question.lower():
                query_list = extract_choices(question, candidates)
            query_list.append(question)

            # --- Use cached embeddings via allocate_node_batched ----
            ec = embedding_cache or {}
            node_list = allocate_node_batched(
                self.args,
                video_graph,
                entity_graph,
                query_list,
                self.embedding_model,
                self.embedding_tokenizer,
                entity_key_embeddings=ec.get("entity_key_embeddings"),
                entity_keys=ec.get("entity_keys"),
                node_content_embeddings=ec.get("node_content_embeddings"),
                node_ids=ec.get("node_ids"),
            )

            # Re-rank nodes using cached embeddings (no model call)
            if node_list and ec.get("node_content_embeddings") is not None:
                node_ids_all = ec.get("node_ids", list(video_graph.nodes))
                id_to_idx = {nid: i for i, nid in enumerate(node_ids_all)}
                valid = [(i, id_to_idx[nid]) for i, nid in enumerate(node_list) if nid in id_to_idx]
                if valid:
                    local_idxs, cache_idxs = zip(*valid)
                    sub_emb = ec["node_content_embeddings"][list(cache_idxs)]
                    query_emb = precompute_embeddings(
                        query_list, self.embedding_model, self.embedding_tokenizer,
                    )
                    if query_emb is not None:
                        sims = torch.mean(query_emb @ sub_emb.T, dim=0)
                        ranked = torch.argsort(sims, descending=True)
                        node_list = [node_list[local_idxs[r]] for r in ranked]
            elif node_list:
                # Fallback: no cache, use compute_text_similarity
                key_list = []
                for node_id in node_list:
                    nd = video_graph.nodes[node_id]
                    content = (
                        nd.get("entities", [])
                        + nd.get("actions", [])
                        + nd.get("scenes", [])
                    )
                    if nd.get("captions") is not None:
                        content += nd.get("captions", [])
                    key_list.append("; ".join(content))
                sims = compute_text_similarity(
                    query_list, key_list,
                    self.embedding_model, self.embedding_tokenizer,
                    return_all=True,
                )
                sorted_indices = torch.argsort(
                    torch.mean(sims, dim=0), descending=True
                )
                node_list = [node_list[i] for i in sorted_indices]

        return {"nodes": node_list[: self.args.n_retrieval], "indices": indices}

    def batch_check_nodes_with_mllm(
        self, nodes, info, obj_count, candidates,
        video_inputs, captions, size_list=None,
    ):
        """Like ``_check_nodes_with_mllm`` but sends all node-check
        prompts to the LLM in a single batch call."""
        if not video_inputs or len(video_inputs) == 0 or video_inputs[0] is None:
            return {}

        split_video_inputs = torch.split(
            video_inputs[0], self.args.chunk_size, dim=0
        )
        split_size_list = (
            torch.split(size_list, self.args.chunk_size, dim=0)
            if size_list is not None
            else None
        )

        # Build all prompts
        batch_prompts = []
        valid_nodes = []
        for node in nodes:
            if node >= len(split_video_inputs):
                continue
            caption_prompt = ""
            if captions is not None:
                start_time = node * self.args.chunk_size // self.args.fps
                end_time = (node + 1) * self.args.chunk_size // self.args.fps
                sel = [t for time, t in captions if start_time <= time < end_time]
                caption_prompt = (
                    " This video's captions are listed below:\n"
                    + " ".join(sel)
                    + "\n"
                )
            instruct = (
                SQL_ANSWER_COUNT_PROMPT.format(questions=info) + caption_prompt
                if obj_count
                else SQL_ANSWER_PROMPT.format(questions=info) + caption_prompt
            )
            batch_prompts.append({
                "text": instruct,
                "video_inputs": [split_video_inputs[node]],
                "max_new_tokens": 256,
            })
            valid_nodes.append(node)

        if not batch_prompts:
            return {}

        responses = self.model.mllm_batch_response(batch_prompts)

        check_result = {}
        for node, resp in zip(valid_nodes, responses):
            try:
                pred = json.loads(
                    resp.replace("```json", "").replace("```", "").strip()
                )
            except (json.JSONDecodeError, TypeError):
                pred = None
            check_result[node] = pred
        return check_result