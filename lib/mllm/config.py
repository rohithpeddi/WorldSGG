import argparse
import torch
import logging

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local-rank', default=0)
    parser.add_argument('--tensor_parallel_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--model_name', default="qwenvl25_7b")
    parser.add_argument('--output_path', default="./eval")
    parser.add_argument('--chunk_size', type=int, default=64)
    parser.add_argument('--task', default="mlvu")
    parser.add_argument('--data_path', default="./data") 
    parser.add_argument('--use_vllm', action=argparse.BooleanOptionalAction, default=True, help='Use vLLM for inference (default: True). Pass --no-use_vllm to load model directly via HuggingFace Transformers.')

    args = parser.parse_args()
    args.duration = args.duration.split(",")

    return args