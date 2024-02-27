from config import *
from data import *
from utils import *
import torch
import argparse
import numpy as np
import os
import torch
from baukit import TraceDict
from os.path import join, basename, dirname
from torch.nn import Module
from tqdm import tqdm
import torch.nn as nn

def get_activations_bau(
        model: Module,
        prompt: torch.Tensor,
        model_name: str='model',
        layer_name: str='self_attn',
        device: str='cuda') -> np.ndarray:
    model.eval()
    #开源工具 baukit
    HEADS = [f"{model_name}.layers.{i}.{layer_name}.head_out" for i in range(model.config.num_hidden_layers)]
    with torch.no_grad():
        with TraceDict(model, HEADS) as ret:
            prompt = prompt.to(device)
            model(prompt, output_hidden_states=True)
        head_wise_activations = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_activations = torch.stack(head_wise_activations, dim=0).squeeze().numpy()
    # 所谓的一个特征 npy 

    return head_wise_activations

def recursive_combine_npy(folder_path: str):
    print(f"Recursively combining files in '{folder_path}'...")
    target_file_name = basename(folder_path)
    target_folder_path = dirname(folder_path)
    npys = []
    num_blocks = len(os.listdir(folder_path))
    for file_name_prefix in range(num_blocks):
        file_path = join(folder_path, f"{file_name_prefix}.npy")
        file = np.load(file_path)
        npys.append(file)
    npy = np.concatenate(npys, axis=0)
    np.save(join(target_folder_path, f"{target_file_name}.npy"), npy)

def main(args: argparse.Namespace):
    assert args.model_name in MODEL_MAPPING, f"Model name `{args.model_name}` does not exist."
    present_args(args)
    device = torch.device("cuda:0")
    model_name = args.model_name
    model_family = MODEL_MAPPING[model_name]
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)
    model.to(device)
    if args.dataset_name == 'tqa':
        data = load_dataset_fast(args.dataset_name, data_type=args.data_type)
        data = data[['question', 'answer', 'true']]
        data = data.rename(columns={'true': 'label'})
        dataset = LlmDataset(data, model_name, tokenizer)
    else:
        raise NotImplementedError
    if model_family == 'open-llama' or model_family == 'llama-2' or model_family == 'vicuna':
        model_name_ = 'model'
        layer_name = 'self_attn'
    elif model_family == 'dolly_v2' or model_family == 'stablelm' or model_family == 'redpajama-incite':
        model_name_ = 'gpt_neox'
        layer_name = 'attention'
    else:
        raise NotImplementedError
    prefix = f"{args.model_name}_{args.dataset_name}_{args.data_type}"
    prompt_activations = []   # hidden states of the last token of the prompt
    block_counter = args.block_size
    index = 0
    mkdir(join(FEATURES, f'{prefix}_prompt_activations'))
    for sample in tqdm(dataset, desc=f"{model_name}: ", bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
        prompt_ids = sample['prompt_ids']
        activations = get_activations_bau(
            model=model,
            prompt=prompt_ids,
            model_name=model_name_,
            layer_name=layer_name,
            device=device) # [num_layer, seq_len, hidden_size]
        # 最后一个token 包含了前面的信息 所有层的隐藏层的输出embedding
        prompt_activations.append(activations[:, -1, :])
        block_counter -= 1
        if block_counter == 0:
            np.save(join(FEATURES, f'{prefix}_prompt_activations', f'{index}.npy'), prompt_activations)
            prompt_activations = []
            block_counter = args.block_size
            index += 1
    print(f"Saving...")
    np.save(join(FEATURES, f'{prefix}_prompt_activations', f'{index}.npy'), prompt_activations)
    recursive_combine_npy(join(FEATURES, f'{prefix}_prompt_activations'))
    deldir(join(FEATURES, f'{prefix}_prompt_activations'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='open_llama_3b')
    parser.add_argument('--dataset_name', type=str, default='tqa', choices=['tqa', 'nq'])
    parser.add_argument('--data_type', type=str, default='finetune', choices=['finetune', 'original'])
    parser.add_argument('--block_size', type=int, default=500, help="Maximum size of samples loaded in memory.")
    args = parser.parse_args()
    main(args)
