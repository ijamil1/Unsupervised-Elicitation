import asyncio
import json
import math
import os
import random
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import argparse
import matplotlib.pyplot as plt
import time
from core.llm_api.llm import ModelAPI
from core.utils import setup_environment

from src.model_querying.prompt_creation import (
    get_judge_prompt_fewshot,
    get_judge_prompt_zeroshot
)
from src.model_querying.solution_extraction import (
get_yes_no_diff_logprobs
)
from src.pipeline.pipeline import Pipeline, PipelineConfig
from src.tools.dataloaders import (
    load_assignments)
from src.tools.path_utils import get_root_directory


def calculate_accuracy(train_data):
    train_probs = []
    for i in train_data.values():
        if i["label"] is None:
            continue
        if i["label"] == 1:
            train_probs.append(i["score"])
        else:
            train_probs.append(-i["score"])
    if len(train_probs) == 0:
        train_prob = 0
    else:
        train_prob = np.mean(train_probs)

    return {
        "train_accuracy": 0
        if len(train_data) == 0
        else np.mean([i["label"] == i["vanilla_label"] for i in train_data.values()]),
        "train_label_distribution": Counter(
            [i["vanilla_label"] for i in train_data.values()]
        ),
        "train_predict_distribution": Counter(
            [i["label"] for i in train_data.values()]
        ),
        "train_prob": train_prob,
        "train_size": len(train_data)
    }


def update_assign(data):
    for key, value in data.items():
        if value["score"] > 0:
            value["label"] = 1
        else:
            value["label"] = 0
    return data


async def compute_logprobs_batched(model_api, model_id, examples_dict):
    """
    Compute log probabilities for all examples using batched vLLM inference.

    This is the KEY optimization: instead of making len(examples_dict) separate API calls,
    we make a single batched call to vLLM which processes all prompts together.

    Args:
        model_api: The ModelAPI instance (should route to vLLM)
        model_id: Model identifier (e.g., "meta-llama/Meta-Llama-3.1-405B")
        examples_dict: Dict of {example_id: example_data}
                      Each example must have "demonstration" field prepared

    Returns:
        Dict of {example_id: score}
    """
    example_ids = list(examples_dict.keys())
    examples = list(examples_dict.values())

    # Prepare all prompts at once
    all_prompts = []
    for example in examples:
        prompt = get_judge_prompt_fewshot(example, pipeline=False)
        all_prompts.append(prompt)

    # CRITICAL: Make a single batched request to vLLM
    # vLLM will process all prompts in parallel using batch inference
    # Batch size is automatically determined by len(all_prompts) = len(cur_pool)
    responses = await model_api(
        model_id,
        all_prompts,  # List of prompts - vLLM handles batching internally
        logprobs=True,
        top_logprobs=20,
        max_tokens=1,
        temperature=0.0,
    )
    assert len(responses) == len(examples)
    
    # Extract scores from batched responses
    scores = {}
    for example_id, response in zip(example_ids, responses):
        try:
            logprobs = response["response"]["logprobs"]
            score = get_yes_no_diff_logprobs(logprobs)
            scores[example_id] = score
        except Exception as e:
            print(response)
            print(f"Error in compute_logprobs_batched extracting score for example {example_id}: {e}; there are {len(responses)} responses in total")
            scores[example_id] = 0

    return scores


def get_pipeline_batched(
    model,
    name=None,
    use_cache=True,
    num_problems=None,
    decision_id=None,
    iter=None,
    assignment=None,
):
    """
    BATCHED VERSION of get_pipeline that uses vLLM's batch inference.

    Instead of creating N separate API calls (one per example), this creates a single
    batched call that processes all examples together. This is the KEY optimization.

    Args:
        model: Model identifier
        name: Pipeline name
        use_cache: Whether to cache results
        num_problems: Number of problems (unused)
        decision_id: Decision ID (unused)
        iter: Iteration number
        assignment: Dict of examples with labels

    Returns:
        Pipeline instance configured for batched inference
    """
    pipeline_name = f"iterative-truth-assign-iter-{iter}-batched"
    if decision_id is not None:
        pipeline_name += f"-{decision_id}"
    if name is not None:
        pipeline_name += "-" + name

    ROOT_DIR = get_root_directory()

    pipeline_config = PipelineConfig(
        pipeline_name,
        openai_fraction_rate_limit=0.99,
        num_problems=num_problems,
        use_cache=use_cache,
    )
    pipeline = Pipeline(pipeline_config, model_api=model_api)  # Pass shared model_api

    assert assignment is not None
    initial_assign = pipeline.add_load_data_step(
        "get_assign", load_assignments, assignment
    )

    def add_train_demonstrations(train_data):
        """Same as before - prepare leave-one-out demonstrations"""
        copy_data = deepcopy(train_data)
        copy_data = {k: v for k, v in copy_data.items() if v["label"] is not None}
        keys = list(copy_data.keys())
        values = list(copy_data.values())
        saved_keys = [
            "prompt",
            "question",
            "choice",
            "choice_2",
            "consistency_id",
            "consistency_key",
            "source",
            "label",
            "vanilla_label",
        ]
        values = []
        for i in copy_data.values():
            values.append({saved_key: i[saved_key] for saved_key in saved_keys if saved_key in i})

        for idx, key in enumerate(keys):
            tmp_keys, tmp_values = [], []
            for j, (prev_key, prev_value) in enumerate(zip(keys, values)):
                if j != idx:
                    tmp_keys.append(prev_key)
                    tmp_values.append(prev_value)

            demos = {
                prev_key: prev_value
                for j, (prev_key, prev_value) in enumerate(zip(tmp_keys, tmp_values))
            }

            sorted_demos = {}
            for k, v in demos.items():
                q = v["consistency_id"]
                if q not in sorted_demos:
                    sorted_demos[q] = []
                sorted_demos[q].append((k, v))

            out_sorted_demos = {}
            for group in sorted_demos.values():
                for k, v in group:
                    out_sorted_demos[k] = v

            copy_data[key]["demonstration"] = out_sorted_demos

        return copy_data

    merged_train_data = pipeline.add_transformation_step(
        "add_train_demonstration",
        add_train_demonstrations,
        dependencies=[initial_assign],
    )

    # NEW: Batched inference step replaces add_query_step
    async def batched_inference_step(train_data):
        """
        Custom transformation step that performs batched inference.

        This replaces the query_model step which would create N separate API calls.
        Instead, we call compute_logprobs_batched which makes a single batched call.
        """
        # Compute scores using batched vLLM inference
        scores = await compute_logprobs_batched(
            pipeline.model_api,
            model,
            train_data,
        )

        # Add scores to examples
        result = {}
        for example_id, example in train_data.items():
            example_copy = example.copy()
            example_copy["score"] = scores.get(example_id, 0)
            result[example_id] = example_copy

        return result

    get_train_preds = pipeline.add_transformation_step(
        "get_train_preds_batched",
        batched_inference_step,
        dependencies=[merged_train_data],
    )

    eval_preds = pipeline.add_eval_step(
        "evaluate",
        calculate_accuracy,
        dependencies=[get_train_preds],
    )
    return pipeline


async def predict_assignment(model, example, demonstrations):
    """
    Predict label for a single example using vLLM direct inference.

    Modified to follow the same pattern as compute_logprobs_batched:
    - Makes direct API call to vLLM
    - Extracts logprobs and computes score directly (no parse_fn)
    - Uses get_yes_no_diff_logprobs for consistency

    Args:
        model: Model identifier
        example: The example to label
        demonstrations: Dict of labeled examples (leave-one-out)

    Returns:
        Predicted label (0 or 1)
    """
    # Prepare demonstrations (leave out current example)
    demos = [
        v
        for k, v in demonstrations.items()
        if k != example["uid"] and v["label"] is not None
    ]

    # Create prompt
    prompt = get_judge_prompt_fewshot(
        example,
        demos,
        pipeline=False,
    )

    # Make direct API call to vLLM (single prompt, not batched)
    response = await model_api(
        model,
        prompt,  # Single prompt string
        logprobs=True,
        top_logprobs=20,
        temperature=0.0,
        max_tokens=1,
    )

    
    # Extract logprobs and compute score directly (same as compute_logprobs_batched)
    try:
        logprobs = response[0]["response"]["logprobs"]
        score = get_yes_no_diff_logprobs(logprobs)
    except Exception as e:
        print('response: ', response)
        print(f"Error in predict_assignment extracting score for example {example.get('uid', 'unknown')}: {e}")
        score = 0

    # Convert score to binary label
    new_label = score > 0
    return int(new_label)

async def predict_assignment_zero_shot(model, example):
    """
    Predict label for a single example using zero-shot inference.

    Modified to follow the same pattern as predict_assignment.
    """
    # Create zero-shot prompt
    prompt = get_judge_prompt_zeroshot(example, pipeline=False)

    response = await model_api(
        model,
        prompt,
        logprobs=5,
        temperature=0.0,
        max_tokens=1,
    )

    # Extract logprobs and compute score directly
    try:
        if 'Instruct' in model:
            logprobs = response[0]["response"]["logprobs"][0]
        else:
            logprobs = response[0]["response"]["logprobs"]
        score = get_yes_no_diff_logprobs(logprobs)
    except Exception as e:
        print(f"Error in predict_assignment_zero_shot extracting score for example {example.get('uid', 'unknown')}: {e}")
        score = 0

    # Convert score to binary label
    new_label = score > 0
    return int(new_label)

def get_temperature(
    iteration, initial_temp, final_temp, decay_rate, schedule="exp"):
    """
    Calculate the temperature for simulated annealing.

    Parameters:
    - iteration: Current iteration number.
    - initial_temp: Initial temperature.
    - decay_rate: Rate at which the temperature decreases.

    Returns:
    - Current temperature.
    """
    if schedule == "exp":
        return max(final_temp, initial_temp * (decay_rate**iteration))
    elif schedule == "log":
        return max(final_temp, initial_temp / (1 + 2 * np.log(1 + iteration)))
    else:
        assert False


def get_energy(metric):
    return metric["train_prob"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=27565976)
    parser.add_argument("--testbed", type=str, default="truthfulQA")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_seed", type=int, default=8)
    parser.add_argument("--K", type=int, default=1500)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--initial_T", type=float, default=10)
    parser.add_argument("--final_T", type=float, default=0.01)
    parser.add_argument("--scheduler", type=str, default="log")

    # vLLM configuration

    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                        help="Fraction of GPU memory to use (0.0-1.0)")
    parser.add_argument("--max_model_len", type=int, default=10000,
                        help="Maximum sequence length (None for model default)")
    parser.add_argument("--max_num_batched_tokens", type=int, default=262144,
                        help="Maximum number of batched tokens per iteration")


    args = parser.parse_args()
    return args

def load_train_data(args):

    with open(get_root_directory() / "praxis_labs_ue1/data/truthfulqa_train.json") as f:
        train = json.load(f)
    
    template = """Question: {question}
        Claim: {answer}
        I think this claim is """

    for i in train:
        i['source'] = 'truthfulQA'
        i['consistency_key'] = 'A' if i['label'] else 'B'
        i['prompt'] = template.format(question=i['question'], answer=i['choice'])
    args.GROUP_SIZE = 4

    train_map = {}
    for i in train:
        if i['consistency_id'] not in train_map:
            train_map[i['consistency_id']] = []
        train_map[i['consistency_id']].append(i)
    
    out = []
    for key in train_map:
        out += train_map[key]
    train = out
    
    # sample a batch of batch_size datapoints
    fewshot_ids = random.sample(
        list(range(len(train)// args.GROUP_SIZE)), args.batch_size // args.GROUP_SIZE
    )
    fewshot_ids = [
        i * args.GROUP_SIZE + j for i in fewshot_ids for j in range(args.GROUP_SIZE)
    ]

    return train, fewshot_ids

def load_test_data(args):
    with open(get_root_directory() / "praxis_labs_ue1/data/truthfulqa_test.json") as f:
        test = json.load(f)
    
    template = """Question: {question}
        Claim: {answer}
        I think this claim is """

    for i in test:
        i['source'] = 'truthfulQA'
        i['consistency_key'] = 'A' if i['label'] else 'B'
        i['prompt'] = template.format(question=i['question'], answer=i['choice'])
    args.GROUP_SIZE = 4

    test_map = {}
    for i in test:
        if i['consistency_id'] not in test_map:
            test_map[i['consistency_id']] = []
        test_map[i['consistency_id']].append(i)
    
    out = []
    for key in test_map:
        out += test_map[key]
    test = out
    
    return test

def initialize(train, fewshot_ids, args):
    demonstrations = {}
    unlabeled_ids = []
    whole_ids = []
    seed_ids = []

    random_init_labels = [1] * (args.num_seed // 2) + [0] * (args.num_seed // 2)
    random.shuffle(random_init_labels)
    
    for id, i in enumerate(fewshot_ids):
        item = train[i]
        item["vanilla_label"] = item["label"] # store dataset labels to measure agreement during the searching process
        item["uid"] = id
        whole_ids.append(item["uid"])
        if id >= args.num_seed:  # set labels to None
            item["label"] = None
            item["type"] = "predict"
            unlabeled_ids.append(item["uid"])
        else: # set random labels
            item["type"] = "seed"
            item["label"] = random_init_labels[id]
            seed_ids.append(item["uid"])
        demonstrations[id] = item
        
    return demonstrations, unlabeled_ids, whole_ids, seed_ids

async def icm_main(args):
    train, fewshot_ids = load_train_data(args)

    demonstrations, unlabeled_ids, whole_ids, seed_ids = initialize(train, fewshot_ids, args)
    
    cur_metric = {
        "train_prob": -1e6,
        "train_accuracy": 1.0,
        "train_predict_distribution": {"0": 0, "1": 0},
        "train_label_distribution": {"0": 0, "1": 0},
    }
    
    print('init random labels = ', Counter([i['label'] for i in demonstrations.values() if i['type'] == 'seed']), 'init label acc = ', np.mean([i['label'] == i['vanilla_label'] for i in demonstrations.values() if i['type'] == 'seed']))
    name = f"{args.testbed}-llama70b-K{args.K}-bc{args.batch_size}_seed{args.seed}-initialsize{args.num_seed}-decay{args.decay}-initialT{args.initial_T}-finalT{args.final_T}-scheduler{args.scheduler}"

    iter = 0
    flip_cnt = 0
    example_id = 0

    for _ in tqdm(range(args.K), desc="searching"):
        cur_pool = {
            k: v for k, v in demonstrations.items() if v["label"] is not None
        }
        if iter == 0:
            pipeline = get_pipeline_batched(  # CHANGED: Use batched version
                args.model,
                name=name,
                num_problems=None,
                iter=iter,
                assignment=cur_pool,
            )
            results = await pipeline.run()
            cur_metric = results["evaluate"]
        cur_pool = {
            k: v for k, v in demonstrations.items() if v["label"] is not None
        }
        while True: # weighted sampling
            candidates_ids = whole_ids
            weights = [1 for _ in range(len(candidates_ids))]
            for idx, (demo_id, demo) in enumerate(demonstrations.items()):
                if demo["label"] is None:
                    weights[idx] = 100
            example_id = random.choices(candidates_ids, k=1, weights=weights)[0]
            break

        new_label = await predict_assignment(
                args.model,#base pre-trained LLM
                demonstrations[example_id],
                cur_pool,
            )
        

        if demonstrations[example_id]["label"] != new_label:
            tmp_demonstrations = deepcopy(demonstrations)
            tmp_demonstrations[example_id]["label"] = new_label
            
            tmp_pool = {
                k: v
                for k, v in tmp_demonstrations.items()
                if v["label"] is not None
            }
            pipeline = get_pipeline_batched(  # CHANGED: Use batched version
                model=args.model,
                name=name,
                num_problems=None,
                iter=iter,
                assignment=tmp_pool,
            )
            results = await pipeline.run()
            metric = results["evaluate"]
            T = get_temperature(
                flip_cnt, args.initial_T, args.final_T, args.decay, schedule=args.scheduler
            )

            if iter % 10 == 0:
                print(f"iter = {iter}, pool size = {len(cur_pool)}, cur acc = {cur_metric['train_accuracy']}, new acc = {metric['train_accuracy']}, cur score = {get_energy(cur_metric)}, new score = {get_energy(metric)}")

            accept_prob = math.exp((get_energy(metric) - get_energy(cur_metric)) / T)
            if random.random() < accept_prob:
                demonstrations = tmp_demonstrations
                flip_cnt += 1
                cur_metric = metric
            else:
                continue
        
        print("=" * 100)
        iter += 1
    
    max_uid = max(demonstrations.keys())
    
    #read in test data
    test = load_test_data(args)
    correct_cnt = 0
    for idx, i in enumerate(test):
        i['uid'] = max_uid + 1 + idx
        new_label = await predict_assignment(
                args.model,
                i,
                demonstrations,
            )
        
        i['new_label'] = new_label
        if i['label'] == i['new_label']:
            correct_cnt += 1
    return correct_cnt / len(test)

async def golden_supervision_main(args):
    train, fewshot_ids = load_train_data(args)

    demonstrations = {}

    for id, i in enumerate(fewshot_ids):
        item = train[i]
        item["uid"] = id    
        demonstrations[id] = item

    max_uid = max(demonstrations.keys())
    
    #read in test data
    test = load_test_data(args)
    correct_cnt = 0
    for idx, i in enumerate(test):
        i['uid'] = max_uid + 1 + idx
        new_label = await predict_assignment(
                args.model,
                i,
                demonstrations,
            )
        
        i['new_label'] = new_label
        if i['label'] == i['new_label']:
            correct_cnt += 1
    return correct_cnt / len(test)

async def zero_shot_chat_main(args):
    #read in test data
    test = load_test_data(args)
    correct_cnt = 0

    # Determine instruct model based on args.model size
    model_size = args.model.split('-')[-1]
    model_size_to_instruct = {
        '8B': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        '70B': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        '405B': 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
    }
    instruct_model = model_size_to_instruct.get(model_size, 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo')

    for idx, i in enumerate(test):
        new_label = await predict_assignment_zero_shot(
                instruct_model,
                i
            )

        i['new_label'] = new_label
        if i['label'] == i['new_label']:
            correct_cnt += 1
        time.sleep(0.5)
    return correct_cnt / len(test)

async def zero_shot_pretrained_main(args):
    #read in test data
    test = load_test_data(args)
    correct_cnt = 0
    for idx, i in enumerate(test):
        new_label = await predict_assignment_zero_shot(
                args.model,
                i
            )
        
        i['new_label'] = new_label
        if i['label'] == i['new_label']:
            correct_cnt += 1
    return correct_cnt / len(test)

def plot_test_accuracies(icm_test_accuracy, golden_supervision_test_accuracy, zero_shot_chat_test_accuracy, zero_shot_pretrained_test_accuracy):
    accuracies = [
        icm_test_accuracy * 100,
        golden_supervision_test_accuracy * 100,
        zero_shot_chat_test_accuracy * 100,
        zero_shot_pretrained_test_accuracy * 100,
    ]
    labels = [
        "Unsupervised (Mine)",
        "Golden Supervision",
        "Zero-shot (Chat)",
        "Zero-shot",
    ]

    bar_colors = [
        "#58b6c0",             # light blue / teal for unsupervised
        "#FFD700",             # golden for golden supervision
        "#B366CC",             # purple for zero-shot (chat)
        "#9658ca",             # purple for zero-shot (pretrained)
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot bars
    bars = ax.bar(
        x, 
        accuracies, 
        color=[bar_colors[0], bar_colors[1], bar_colors[3], bar_colors[2]],  # regular bars
        tick_label=labels,
        edgecolor="k",
        zorder=2
    )

    # Add hatching to "Zero-shot (Chat)" bar (index 2)
    bars[2].set_hatch('...')  # dotted hatch
    bars[2].set_color(bar_colors[2])  # Make sure it's the purple with dots

    # Set y-axis
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Test Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")

    # Add value labels atop bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.grid(axis='y', zorder=1, alpha=0.3)
    plt.savefig("figure_1.png", dpi=500)
    plt.close()

async def async_main():
    setup_environment(logger_level="error")
    args = get_args()

    valid_models = [
        'meta-llama/Llama-3.1-405B',
        'meta-llama/Llama-3.1-70B',
        'meta-llama/Llama-3.1-8B',
    ]
    assert args.model in valid_models, f"args.model must be one of {valid_models}, got {args.model}"

    print("task: ", args.testbed)
    random.seed(args.seed)

    print('entering icm_main')
    icm_test_accuracy = await icm_main(args)

    print('entering golden_supervision benchmarking method')
    golden_supervision_test_accuracy = await golden_supervision_main(args)

    print('entering zero-shot chat benchmarking method')
    zero_shot_chat_test_accuracy = await zero_shot_chat_main(args)

    print('entering zero-shot pretrained benchmarking method')
    zero_shot_pretrained_test_accuracy = await zero_shot_pretrained_main(args)

    print(f"ICM test accuracy = {icm_test_accuracy}")
    print(f"Golden supervision test accuracy = {golden_supervision_test_accuracy}")
    print(f"Zero shot chat test accuracy = {zero_shot_chat_test_accuracy}")
    print(f"Zero shot pretrained test accuracy = {zero_shot_pretrained_test_accuracy}")

    # Call the plotting function
    plot_test_accuracies(
        icm_test_accuracy,
        golden_supervision_test_accuracy,
        zero_shot_chat_test_accuracy,
        zero_shot_pretrained_test_accuracy
    )

if __name__ == "__main__":
    args = get_args()

    # Initialize ModelAPI with vLLM configuration from command line args
    model_api = ModelAPI(
        openai_fraction_rate_limit=0.99,
        use_vllm=True,
        vllm_model_name=args.model,  # Use --model arg for vLLM model name
        vllm_tensor_parallel_size=args.tensor_parallel_size,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_max_model_len=args.max_model_len,
        vllm_max_num_batched_tokens=args.max_num_batched_tokens,
        vllm_enable_prefix_caching=True
    )

    try:
        asyncio.run(async_main())
    finally:
        # Gracefully shutdown vLLM engine
        model_api.shutdown()