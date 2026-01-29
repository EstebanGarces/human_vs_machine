#!/usr/bin/env python3
"""
Truncation Hypothesis Experiments

Measures human token exclusion rates under top-k and top-p truncation
across multiple language models and text corpora.
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import gc

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from scipy import stats
from scipy.stats import bootstrap

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Models
MODELS_FULL = {
    'GPT2-XL': 'gpt2-xl',
    'Qwen2-7B': 'Qwen/Qwen2-7B',
    'Mistral-7B': 'mistralai/Mistral-7B-v0.3',
    'LLaMA3-8B': 'meta-llama/Meta-Llama-3-8B',
    'Falcon2-11B': 'tiiuae/falcon-11B',
}

MODELS_SMALL = {
    'GPT2-XL': 'gpt2-xl',
    'Qwen2-7B': 'Qwen/Qwen2-7B',
    'Mistral-7B': 'mistralai/Mistral-7B-v0.3',
}

# Parameters
TOP_K_VALUES = [1, 5, 10, 20, 50, 100]
TOP_P_VALUES = [0.6, 0.8, 0.9, 0.95, 0.99]
RANK_BINS = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
RANK_BIN_LABELS = ['[1,5]', '[6,10]', '[11,20]', '[21,50]', '[51,100]', '>100']

MAX_SAMPLES_PER_DATASET = 500
MAX_SEQUENCE_LENGTH = 256
MIN_SEQUENCE_LENGTH = 50
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_N_RESAMPLES = 10000
SEED = 42


def setup_google_drive(folder_name: str = 'truncation_experiments') -> str:
    """Mount Google Drive and create output directory."""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        output_dir = f'/content/drive/MyDrive/{folder_name}'
    except ImportError:
        output_dir = f'./{folder_name}'
    except Exception:
        output_dir = f'./{folder_name}'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/tables', exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    os.makedirs(f'{output_dir}/raw_data', exist_ok=True)
    return output_dir


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment or Colab secrets."""
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        return hf_token
    try:
        from google.colab import userdata
        return userdata.get('HF_TOKEN')
    except:
        return None


def login_huggingface(hf_token: Optional[str]) -> bool:
    if not hf_token:
        return False
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        return True
    except:
        return False


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_to_drive(data: Any, filepath: str, format: str = 'json'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
            else:
                pd.DataFrame(data).to_csv(filepath, index=False)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif format == 'text':
            with open(filepath, 'w') as f:
                f.write(str(data))
        return True
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        return False


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    if len(data) < 2:
        mean = np.mean(data) if data else 0.0
        return mean, mean, mean

    data_array = np.array(data)
    mean = np.mean(data_array)

    try:
        result = bootstrap(
            (data_array,), np.mean,
            n_resamples=min(BOOTSTRAP_N_RESAMPLES, len(data) * 100),
            confidence_level=confidence, method='percentile'
        )
        return mean, result.confidence_interval.low, result.confidence_interval.high
    except:
        sem = stats.sem(data_array)
        ci = sem * stats.t.ppf((1 + confidence) / 2, len(data_array) - 1)
        return mean, mean - ci, mean + ci


def compute_statistical_tests(group1: List[float], group2: List[float]) -> Dict:
    results = {}
    if len(group1) < 2 or len(group2) < 2:
        return {'error': 'Insufficient data'}

    try:
        stat, pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        results['mann_whitney_u'] = {'statistic': float(stat), 'p_value': float(pval)}
    except Exception as e:
        results['mann_whitney_u'] = {'error': str(e)}

    try:
        stat, pval = stats.ttest_ind(group1, group2, equal_var=False)
        results['welch_t_test'] = {'statistic': float(stat), 'p_value': float(pval)}
    except Exception as e:
        results['welch_t_test'] = {'error': str(e)}

    try:
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        results['cohens_d'] = float((np.mean(group1) - np.mean(group2)) / pooled_std) if pooled_std > 0 else 0
    except:
        pass

    return results


def load_and_prepare_datasets(max_samples: int = 500, min_length: int = 50, max_length: int = 256) -> Dict[str, List[str]]:
    datasets_dict = {}
    min_chars = min_length * 4

    # BookCorpus alternatives
    book_texts = []
    book_sources = [
        ('pg19', 'pg19', 'train', 'text'),
        ('bookcorpus', 'bookcorpus', 'train', 'text'),
        ('openwebtext', 'openwebtext', 'train', 'text'),
    ]

    for name, dataset_name, split, text_field in book_sources:
        if book_texts:
            break
        try:
            data = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
            for item in tqdm(data, desc=f"Loading {name}", total=max_samples * 5):
                text = item.get(text_field, '')
                if isinstance(text, str) and len(text) >= min_chars:
                    if len(text) > 10000:
                        start = len(text) // 4
                        text = text[start:start + 5000]
                    book_texts.append(text.strip())
                if len(book_texts) >= max_samples:
                    break
        except:
            continue

    datasets_dict['BookCorpus'] = book_texts[:max_samples]

    # WikiText
    try:
        wiki_data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
        wiki_texts = []
        for item in tqdm(wiki_data, desc="Loading WikiText"):
            text = item.get('text', '').strip()
            if len(text) >= min_chars and not text.startswith('='):
                wiki_texts.append(text)
            if len(wiki_texts) >= max_samples:
                break
        datasets_dict['WikiText'] = wiki_texts[:max_samples]
    except:
        datasets_dict['WikiText'] = []

    # News (CNN/DailyMail)
    try:
        news_data = load_dataset('cnn_dailymail', '3.0.0', split='train')
        news_texts = []
        for item in tqdm(news_data, desc="Loading News"):
            text = item.get('article', '').strip()
            if len(text) >= min_chars:
                news_texts.append(text)
            if len(news_texts) >= max_samples:
                break
        datasets_dict['WikiNews'] = news_texts[:max_samples]
    except:
        datasets_dict['WikiNews'] = []

    # Fallback for empty BookCorpus
    if len(datasets_dict['BookCorpus']) == 0 and len(datasets_dict['WikiText']) > 0:
        try:
            wiki_data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
            additional = []
            for item in wiki_data:
                text = item.get('text', '').strip()
                if len(text) >= min_chars and not text.startswith('='):
                    additional.append(text)
                if len(additional) >= max_samples:
                    break
            datasets_dict['BookCorpus'] = additional[:max_samples]
        except:
            pass

    return datasets_dict


def load_model_and_tokenizer(model_name: str, model_path: str, use_4bit: bool = True, 
                             hf_token: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit and 'gpt2' not in model_path.lower():
        try:
            from transformers import BitsAndBytesConfig
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=qconfig, device_map="auto",
                trust_remote_code=True, token=hf_token
            )
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto",
                trust_remote_code=True, token=hf_token
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto", trust_remote_code=True,
            token=hf_token if 'gpt2' not in model_path.lower() else None
        )

    model.eval()
    return model, tokenizer


def compute_token_ranks_with_full_info(text: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                                       max_length: int = 256, store_top_probs: int = 200) -> Dict:
    encoding = tokenizer(text, return_tensors='pt', truncation=True, 
                        max_length=max_length, return_attention_mask=True)
    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)
    seq_length = input_ids.shape[1]

    if seq_length < 2:
        return {'ranks': [], 'probabilities': [], 'sorted_probs': [], 
                'top_token_texts': [], 'actual_token_texts': [], 'num_tokens': 0}

    ranks, probabilities, sorted_probs_list = [], [], []
    top_token_texts, actual_token_texts = [], []

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        for t in range(1, seq_length):
            probs = F.softmax(logits[0, t-1, :], dim=-1)
            actual_token_id = input_ids[0, t].item()
            actual_token_prob = probs[actual_token_id].item()
            rank = (probs >= actual_token_prob).sum().item()

            top_probs, top_indices = torch.topk(probs, min(store_top_probs, len(probs)))
            top_tokens_decoded = [tokenizer.decode([idx.item()]) for idx in top_indices[:100]]

            ranks.append(rank)
            probabilities.append(actual_token_prob)
            sorted_probs_list.append(top_probs.cpu().tolist())
            top_token_texts.append(top_tokens_decoded)
            actual_token_texts.append(tokenizer.decode([actual_token_id]))

    return {
        'ranks': ranks, 'probabilities': probabilities, 'sorted_probs': sorted_probs_list,
        'top_token_texts': top_token_texts, 'actual_token_texts': actual_token_texts,
        'num_tokens': len(ranks)
    }


def compute_exclusion_rate_topk(ranks: List[int], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank > k) / len(ranks)


def compute_text_level_overlap(top_tokens_1: List[List[str]], top_tokens_2: List[List[str]], k: int) -> float:
    min_length = min(len(top_tokens_1), len(top_tokens_2))
    if min_length == 0:
        return 0.0

    overlaps = []
    for t in range(min_length):
        set_1 = set(tok.lower().strip() for tok in top_tokens_1[t][:k])
        set_2 = set(tok.lower().strip() for tok in top_tokens_2[t][:k])
        set_1.discard('')
        set_2.discard('')

        if len(set_1) == 0 or len(set_2) == 0:
            continue

        intersection = len(set_1 & set_2)
        min_size = min(len(set_1), len(set_2))
        if min_size > 0:
            overlaps.append(intersection / min_size)

    return np.mean(overlaps) if overlaps else 0.0


def run_exclusion_rate_experiment(datasets: Dict[str, List[str]], models_config: Dict[str, str],
                                  top_k_values: List[int], top_p_values: List[float],
                                  max_length: int = 256, output_dir: str = './output',
                                  hf_token: Optional[str] = None) -> Dict:
    results = {
        'exclusion_rates_topk': defaultdict(lambda: defaultdict(lambda: defaultdict(float))),
        'exclusion_rates_topp': defaultdict(lambda: defaultdict(lambda: defaultdict(float))),
        'rank_distributions': defaultdict(lambda: defaultdict(dict)),
        'all_ranks': defaultdict(lambda: defaultdict(list)),
        'all_probabilities': defaultdict(lambda: defaultdict(list)),
        'top_token_texts': defaultdict(lambda: defaultdict(list)),
        'per_sample_exclusion_rates': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'statistics': defaultdict(lambda: defaultdict(dict)),
        'metadata': {
            'top_k_values': top_k_values, 'top_p_values': top_p_values,
            'max_length': max_length, 'timestamp': datetime.now().isoformat(),
            'models': list(models_config.keys()), 'datasets': list(datasets.keys())
        }
    }

    for model_name, model_path in models_config.items():
        print(f"\nProcessing {model_name}")
        try:
            model, tokenizer = load_model_and_tokenizer(model_name, model_path, hf_token=hf_token)

            for dataset_name, texts in datasets.items():
                if not texts:
                    continue

                print(f"  {dataset_name} ({len(texts)} samples)")
                dataset_ranks, dataset_probs, dataset_sorted_probs, dataset_top_tokens = [], [], [], []
                sample_exclusion_rates = {k: [] for k in top_k_values}

                for text in tqdm(texts, desc=f"    Processing"):
                    try:
                        info = compute_token_ranks_with_full_info(text, model, tokenizer, max_length)
                        if info['ranks']:
                            dataset_ranks.extend(info['ranks'])
                            dataset_probs.extend(info['probabilities'])
                            dataset_sorted_probs.extend(info['sorted_probs'])
                            dataset_top_tokens.extend(info['top_token_texts'])
                            for k in top_k_values:
                                sample_exclusion_rates[k].append(compute_exclusion_rate_topk(info['ranks'], k))
                    except:
                        continue

                if not dataset_ranks:
                    continue

                results['all_ranks'][model_name][dataset_name] = dataset_ranks
                results['all_probabilities'][model_name][dataset_name] = dataset_probs
                results['top_token_texts'][model_name][dataset_name] = dataset_top_tokens

                for k in top_k_values:
                    results['per_sample_exclusion_rates'][model_name][dataset_name][k] = sample_exclusion_rates[k]
                    rate = compute_exclusion_rate_topk(dataset_ranks, k)
                    results['exclusion_rates_topk'][model_name][dataset_name][k] = rate

                    if sample_exclusion_rates[k]:
                        mean, ci_low, ci_high = compute_confidence_interval(sample_exclusion_rates[k])
                        results['statistics'][model_name][dataset_name][f'topk_{k}'] = {
                            'mean': mean, 'ci_lower': ci_low, 'ci_upper': ci_high,
                            'n_samples': len(sample_exclusion_rates[k])
                        }

                for p in top_p_values:
                    excluded = 0
                    for rank, sorted_probs in zip(dataset_ranks, dataset_sorted_probs):
                        cumsum = sum(sorted_probs[:min(rank - 1, len(sorted_probs))])
                        if cumsum >= p:
                            excluded += 1
                    results['exclusion_rates_topp'][model_name][dataset_name][p] = excluded / len(dataset_ranks)

                rank_dist = {label: 0 for label in RANK_BIN_LABELS}
                for rank in dataset_ranks:
                    for (low, high), label in zip(RANK_BINS, RANK_BIN_LABELS):
                        if low <= rank <= high:
                            rank_dist[label] += 1
                            break

                total = len(dataset_ranks)
                results['rank_distributions'][model_name][dataset_name] = {
                    k: v/total*100 for k, v in rank_dist.items()
                }

                results['statistics'][model_name][dataset_name]['rank_stats'] = {
                    'median': np.median(dataset_ranks), 'mean': np.mean(dataset_ranks),
                    'std': np.std(dataset_ranks), 'n_tokens': len(dataset_ranks)
                }

            del model, tokenizer
            clear_gpu_memory()
        except Exception as e:
            print(f"  Error: {e}")
            continue

    def convert_to_dict(d):
        if isinstance(d, defaultdict):
            return {k: convert_to_dict(v) for k, v in d.items()}
        return d

    results = convert_to_dict(results)
    results_for_json = {k: v for k, v in results.items() if k != 'top_token_texts'}
    save_to_drive(results_for_json, f'{output_dir}/raw_data/exclusion_results.json', 'json')
    return results


def run_overlap_experiment(datasets: Dict[str, List[str]], models_config: Dict[str, str],
                          exclusion_results: Dict, k_values: List[int] = [10, 20, 50],
                          max_samples: int = 100, max_length: int = 128,
                          output_dir: str = './output', hf_token: Optional[str] = None) -> Dict:
    results = {
        'overlaps': defaultdict(lambda: defaultdict(dict)),
        'per_sample_overlaps': defaultdict(lambda: defaultdict(list)),
        'metadata': {'k_values': k_values, 'max_samples': max_samples,
                     'timestamp': datetime.now().isoformat(), 'method': 'text_level_overlap'}
    }

    model_names = list(models_config.keys())
    has_cached = 'top_token_texts' in exclusion_results

    if has_cached:
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i >= j:
                    continue

                print(f"\nOverlap: {name1} vs {name2}")
                for k in k_values:
                    all_overlaps = []
                    for ds_name in datasets.keys():
                        tokens_1 = exclusion_results['top_token_texts'].get(name1, {}).get(ds_name, [])
                        tokens_2 = exclusion_results['top_token_texts'].get(name2, {}).get(ds_name, [])

                        if not tokens_1 or not tokens_2:
                            continue

                        min_len = min(len(tokens_1), len(tokens_2))
                        chunk_size = min_len // max(len(datasets[ds_name]), 1)
                        
                        if chunk_size > 10:
                            for start in range(0, min_len - chunk_size, chunk_size):
                                overlap = compute_text_level_overlap(
                                    tokens_1[start:start+chunk_size],
                                    tokens_2[start:start+chunk_size], k
                                )
                                all_overlaps.append(overlap)
                        else:
                            all_overlaps.append(compute_text_level_overlap(
                                tokens_1[:min_len], tokens_2[:min_len], k
                            ))

                    if all_overlaps:
                        mean, ci_low, ci_high = compute_confidence_interval(all_overlaps)
                        results['overlaps'][f"{name1} -- {name2}"][k] = {
                            'mean': mean, 'std': np.std(all_overlaps),
                            'ci_lower': ci_low, 'ci_upper': ci_high, 'n_samples': len(all_overlaps)
                        }
                        results['per_sample_overlaps'][f"{name1} -- {name2}"][k] = all_overlaps
                        print(f"  k={k}: {mean:.4f}")
    else:
        all_texts = []
        for ds_name, texts in datasets.items():
            if texts:
                n_per_ds = max_samples // len(datasets)
                all_texts.extend([(text, ds_name) for text in texts[:n_per_ds]])
        all_texts = all_texts[:max_samples]

        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i >= j:
                    continue

                print(f"\nOverlap: {name1} vs {name2}")
                try:
                    model1, tok1 = load_model_and_tokenizer(name1, models_config[name1], hf_token=hf_token)
                    model2, tok2 = load_model_and_tokenizer(name2, models_config[name2], hf_token=hf_token)

                    for k in k_values:
                        overlaps = []
                        for text, _ in tqdm(all_texts, desc=f"  k={k}"):
                            try:
                                info1 = compute_token_ranks_with_full_info(text, model1, tok1, max_length)
                                info2 = compute_token_ranks_with_full_info(text, model2, tok2, max_length)
                                if info1['top_token_texts'] and info2['top_token_texts']:
                                    overlaps.append(compute_text_level_overlap(
                                        info1['top_token_texts'], info2['top_token_texts'], k
                                    ))
                            except:
                                continue

                        if overlaps:
                            mean, ci_low, ci_high = compute_confidence_interval(overlaps)
                            results['overlaps'][f"{name1} -- {name2}"][k] = {
                                'mean': mean, 'std': np.std(overlaps),
                                'ci_lower': ci_low, 'ci_upper': ci_high, 'n_samples': len(overlaps)
                            }
                            print(f"  k={k}: {mean:.4f}")

                    del model1, model2, tok1, tok2
                    clear_gpu_memory()
                except Exception as e:
                    print(f"  Error: {e}")

    def convert_to_dict(d):
        if isinstance(d, defaultdict):
            return {k: convert_to_dict(v) for k, v in d.items()}
        return d

    results = convert_to_dict(results)
    save_to_drive({k: v for k, v in results.items() if k != 'per_sample_overlaps'},
                  f'{output_dir}/raw_data/overlap_results.json', 'json')
    return results


def generate_table_1(results: Dict, output_dir: str = './output') -> pd.DataFrame:
    topk_rates = results['exclusion_rates_topk']
    topp_rates = results['exclusion_rates_topp']
    datasets_list = ['BookCorpus', 'WikiNews', 'WikiText']
    table_data = []

    for k in TOP_K_VALUES:
        row = {'Setting': f'k={k}'}
        all_rates = []
        for ds in datasets_list:
            rates = [topk_rates[m][ds][k] for m in topk_rates 
                    if ds in topk_rates[m] and k in topk_rates[m][ds]]
            if rates:
                row[ds] = f"{np.mean(rates)*100:.1f} +/- {np.std(rates)*100:.1f}"
                all_rates.extend(rates)
            else:
                row[ds] = "--"
        if all_rates:
            mean, ci_low, ci_high = compute_confidence_interval([r * 100 for r in all_rates])
            row['Avg. (95% CI)'] = f"{mean:.1f} [{ci_low:.1f}, {ci_high:.1f}]"
        else:
            row['Avg. (95% CI)'] = "--"
        table_data.append(row)

    table_data.append({col: '---' for col in ['Setting'] + datasets_list + ['Avg. (95% CI)']})

    for p in TOP_P_VALUES:
        row = {'Setting': f'p={p}'}
        all_rates = []
        for ds in datasets_list:
            rates = [topp_rates[m][ds][p] for m in topp_rates 
                    if ds in topp_rates[m] and p in topp_rates[m][ds]]
            if rates:
                row[ds] = f"{np.mean(rates)*100:.1f} +/- {np.std(rates)*100:.1f}"
                all_rates.extend(rates)
            else:
                row[ds] = "--"
        if all_rates:
            mean, ci_low, ci_high = compute_confidence_interval([r * 100 for r in all_rates])
            row['Avg. (95% CI)'] = f"{mean:.1f} [{ci_low:.1f}, {ci_high:.1f}]"
        else:
            row['Avg. (95% CI)'] = "--"
        table_data.append(row)

    df = pd.DataFrame(table_data)
    print("\nTable 1: Exclusion Rates (%)")
    print(df.to_string(index=False))
    save_to_drive(df, f'{output_dir}/tables/table1_exclusion_rates.csv', 'csv')
    save_to_drive(df.to_latex(index=False, escape=False), 
                  f'{output_dir}/tables/table1_exclusion_rates.tex', 'text')
    return df


def generate_table_2(results: Dict, output_dir: str = './output') -> pd.DataFrame:
    rank_dists = results['rank_distributions']
    all_ranks = results['all_ranks']
    datasets_list = ['BookCorpus', 'WikiNews', 'WikiText']
    table_data = []

    for label in RANK_BIN_LABELS:
        row = {'Rank Range': label}
        for ds in datasets_list:
            pcts = [rank_dists[m][ds][label] for m in rank_dists 
                   if ds in rank_dists[m] and label in rank_dists[m][ds]]
            row[ds] = f"{np.mean(pcts):.1f} +/- {np.std(pcts):.1f}" if pcts else "--"
        table_data.append(row)

    table_data.append({col: '---' for col in ['Rank Range'] + datasets_list})

    for stat_name, stat_func in [('Median', np.median), ('Mean', np.mean), ('Std', np.std)]:
        row = {'Rank Range': stat_name}
        for ds in datasets_list:
            values = [stat_func(all_ranks[m][ds]) for m in all_ranks 
                     if ds in all_ranks[m] and all_ranks[m][ds]]
            if values:
                row[ds] = f"{np.mean(values):.1f}" + (f" +/- {np.std(values):.1f}" if stat_name != 'Std' else "")
            else:
                row[ds] = "--"
        table_data.append(row)

    row = {'Rank Range': 'N tokens'}
    for ds in datasets_list:
        total = sum(len(all_ranks[m][ds]) for m in all_ranks if ds in all_ranks[m])
        row[ds] = f"{total:,}"
    table_data.append(row)

    df = pd.DataFrame(table_data)
    print("\nTable 2: Rank Distribution (%)")
    print(df.to_string(index=False))
    save_to_drive(df, f'{output_dir}/tables/table2_rank_distribution.csv', 'csv')
    save_to_drive(df.to_latex(index=False, escape=False), 
                  f'{output_dir}/tables/table2_rank_distribution.tex', 'text')
    return df


def generate_table_3(results: Dict, output_dir: str = './output') -> pd.DataFrame:
    overlaps = results['overlaps']
    k_values = results['metadata']['k_values']
    table_data = []

    for pair_name in overlaps:
        row = {'Model Pair': pair_name}
        for k in k_values:
            if k in overlaps[pair_name]:
                d = overlaps[pair_name][k]
                row[f'k={k}'] = f"{d['mean']:.3f} [{d.get('ci_lower', d['mean']):.3f}, {d.get('ci_upper', d['mean']):.3f}]"
            else:
                row[f'k={k}'] = "--"
        table_data.append(row)

    avg_row = {'Model Pair': 'Average'}
    for k in k_values:
        vals = [overlaps[p][k]['mean'] for p in overlaps if k in overlaps[p]]
        if vals:
            mean, ci_low, ci_high = compute_confidence_interval(vals)
            avg_row[f'k={k}'] = f"{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
        else:
            avg_row[f'k={k}'] = "--"
    table_data.append(avg_row)

    df = pd.DataFrame(table_data)
    print("\nTable 3: Model Overlap (text-level)")
    print(df.to_string(index=False))
    save_to_drive(df, f'{output_dir}/tables/table3_model_overlap.csv', 'csv')
    save_to_drive(df.to_latex(index=False, escape=False), 
                  f'{output_dir}/tables/table3_model_overlap.tex', 'text')
    return df


def generate_supplementary_tables(exclusion_results: Dict, overlap_results: Dict, 
                                  output_dir: str = './output') -> Dict[str, pd.DataFrame]:
    tables = {}

    # Per-model top-k
    rows = []
    for model in exclusion_results['exclusion_rates_topk']:
        for ds in exclusion_results['exclusion_rates_topk'][model]:
            row = {'Model': model, 'Dataset': ds}
            for k in TOP_K_VALUES:
                rate = exclusion_results['exclusion_rates_topk'][model][ds].get(k)
                row[f'k={k}'] = f"{rate*100:.2f}" if rate is not None else "--"
            rows.append(row)
    tables['S1'] = pd.DataFrame(rows)
    save_to_drive(tables['S1'], f'{output_dir}/tables/table_s1_per_model_topk.csv', 'csv')

    # Per-model top-p
    rows = []
    for model in exclusion_results['exclusion_rates_topp']:
        for ds in exclusion_results['exclusion_rates_topp'][model]:
            row = {'Model': model, 'Dataset': ds}
            for p in TOP_P_VALUES:
                rate = exclusion_results['exclusion_rates_topp'][model][ds].get(p)
                row[f'p={p}'] = f"{rate*100:.2f}" if rate is not None else "--"
            rows.append(row)
    tables['S2'] = pd.DataFrame(rows)
    save_to_drive(tables['S2'], f'{output_dir}/tables/table_s2_per_model_topp.csv', 'csv')

    # Statistical tests
    all_ranks = exclusion_results['all_ranks']
    datasets_list = ['BookCorpus', 'WikiNews', 'WikiText']
    rows = []
    for i, ds1 in enumerate(datasets_list):
        for ds2 in datasets_list[i+1:]:
            ranks1 = [r for m in all_ranks if ds1 in all_ranks[m] for r in all_ranks[m][ds1]]
            ranks2 = [r for m in all_ranks if ds2 in all_ranks[m] for r in all_ranks[m][ds2]]
            if ranks1 and ranks2:
                tests = compute_statistical_tests(ranks1, ranks2)
                row = {
                    'Comparison': f'{ds1} vs {ds2}', 'N1': len(ranks1), 'N2': len(ranks2),
                    'Mean1': f"{np.mean(ranks1):.2f}", 'Mean2': f"{np.mean(ranks2):.2f}",
                }
                if 'mann_whitney_u' in tests and 'p_value' in tests['mann_whitney_u']:
                    row['MW-U p'] = f"{tests['mann_whitney_u']['p_value']:.2e}"
                if 'cohens_d' in tests:
                    row["Cohen's d"] = f"{tests['cohens_d']:.3f}"
                rows.append(row)
    tables['S3'] = pd.DataFrame(rows)
    save_to_drive(tables['S3'], f'{output_dir}/tables/table_s3_statistical_tests.csv', 'csv')

    return tables


def generate_figures(results: Dict, output_dir: str = './output'):
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass

    datasets_list = ['BookCorpus', 'WikiNews', 'WikiText']
    colors = {'BookCorpus': '#E74C3C', 'WikiNews': '#3498DB', 'WikiText': '#2ECC71'}
    markers = {'BookCorpus': 'o', 'WikiNews': 's', 'WikiText': '^'}
    all_ranks = results['all_ranks']
    topk_rates = results['exclusion_rates_topk']
    topp_rates = results['exclusion_rates_topp']

    # Rank distribution histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, ds in enumerate(datasets_list):
        combined = [r for m in all_ranks if ds in all_ranks[m] for r in all_ranks[m][ds]]
        if combined:
            axes[idx].hist(combined, bins=50, range=(1, 200), alpha=0.7, 
                          color='steelblue', edgecolor='white')
            axes[idx].set_xlabel('Token Rank')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{ds} (n={len(combined):,})')
            axes[idx].set_xlim(1, 200)
            for k, c in [(5, 'red'), (10, 'orange'), (20, 'green'), (50, 'purple')]:
                axes[idx].axvline(x=k, color=c, linestyle='--', alpha=0.7, label=f'k={k}')
            axes[idx].axvline(x=np.median(combined), color='black', linestyle='-', 
                             alpha=0.8, label=f'Median={np.median(combined):.0f}')
            if idx == 0:
                axes[idx].legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/rank_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figures/rank_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Exclusion vs k
    fig, ax = plt.subplots(figsize=(10, 6))
    for ds in datasets_list:
        avg_rates, ci_lowers, ci_uppers = [], [], []
        for k in TOP_K_VALUES:
            rates = [topk_rates[m][ds][k] for m in topk_rates 
                    if ds in topk_rates[m] and k in topk_rates[m][ds]]
            if rates:
                mean, ci_low, ci_high = compute_confidence_interval([r * 100 for r in rates])
                avg_rates.append(mean)
                ci_lowers.append(mean - ci_low)
                ci_uppers.append(ci_high - mean)
            else:
                avg_rates.append(None)
                ci_lowers.append(None)
                ci_uppers.append(None)

        valid_k = [k for k, r in zip(TOP_K_VALUES, avg_rates) if r is not None]
        valid_rates = [r for r in avg_rates if r is not None]
        valid_ci = [[c for c in ci_lowers if c is not None], [c for c in ci_uppers if c is not None]]
        if valid_rates:
            ax.errorbar(valid_k, valid_rates, yerr=valid_ci, label=ds, color=colors[ds],
                       marker=markers[ds], markersize=8, capsize=4, linewidth=2)

    ax.set_xlabel('Top-k Parameter')
    ax.set_ylabel('Exclusion Rate (%)')
    ax.set_title('Human Token Exclusion Rate vs. Top-k Truncation')
    ax.legend()
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/exclusion_vs_k.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figures/exclusion_vs_k.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Exclusion vs p
    fig, ax = plt.subplots(figsize=(10, 6))
    for ds in datasets_list:
        avg_rates, ci_lowers, ci_uppers = [], [], []
        for p in TOP_P_VALUES:
            rates = [topp_rates[m][ds][p] for m in topp_rates 
                    if ds in topp_rates[m] and p in topp_rates[m][ds]]
            if rates:
                mean, ci_low, ci_high = compute_confidence_interval([r * 100 for r in rates])
                avg_rates.append(mean)
                ci_lowers.append(mean - ci_low)
                ci_uppers.append(ci_high - mean)
            else:
                avg_rates.append(None)
                ci_lowers.append(None)
                ci_uppers.append(None)

        valid_p = [p for p, r in zip(TOP_P_VALUES, avg_rates) if r is not None]
        valid_rates = [r for r in avg_rates if r is not None]
        valid_ci = [[c for c in ci_lowers if c is not None], [c for c in ci_uppers if c is not None]]
        if valid_rates:
            ax.errorbar(valid_p, valid_rates, yerr=valid_ci, label=ds, color=colors[ds],
                       marker=markers[ds], markersize=8, capsize=4, linewidth=2)

    ax.set_xlabel('Top-p Parameter')
    ax.set_ylabel('Exclusion Rate (%)')
    ax.set_title('Human Token Exclusion Rate vs. Top-p Truncation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/exclusion_vs_p.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figures/exclusion_vs_p.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 5))
    models = list(topk_rates.keys())
    matrix = []
    for model in models:
        row = []
        for k in TOP_K_VALUES:
            rates = [topk_rates[model][ds][k] for ds in datasets_list 
                    if ds in topk_rates[model] and k in topk_rates[model][ds]]
            row.append(np.mean(rates) * 100 if rates else np.nan)
        matrix.append(row)

    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(TOP_K_VALUES)))
    ax.set_xticklabels([f'k={k}' for k in TOP_K_VALUES])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(TOP_K_VALUES)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.1f}%', ha='center', va='center', fontsize=10)

    plt.colorbar(im, ax=ax, label='Exclusion Rate (%)')
    ax.set_title('Exclusion Rate by Model and Top-k')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/model_comparison_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figures/model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(exclusion_results: Dict, overlap_results: Dict, output_dir: str = './output') -> str:
    lines = [
        "TRUNCATION HYPOTHESIS EXPERIMENTS - RESULTS",
        "=" * 60,
        f"Generated: {datetime.now().isoformat()}",
        "",
        "SETUP",
        "-" * 40,
    ]

    metadata = exclusion_results.get('metadata', {})
    lines.append(f"Models: {', '.join(metadata.get('models', []))}")
    lines.append(f"Datasets: {', '.join(metadata.get('datasets', []))}")
    lines.append(f"Top-k values: {metadata.get('top_k_values', [])}")
    lines.append(f"Top-p values: {metadata.get('top_p_values', [])}")
    lines.append("")

    lines.append("TOP-K EXCLUSION RATES")
    lines.append("-" * 40)
    topk = exclusion_results['exclusion_rates_topk']
    for k in TOP_K_VALUES:
        rates = [topk[m][ds][k] for m in topk for ds in topk[m] if k in topk[m][ds]]
        if rates:
            mean, ci_low, ci_high = compute_confidence_interval([r * 100 for r in rates])
            lines.append(f"k={k}: {mean:.2f}% (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")
    lines.append("")

    lines.append("TOP-P EXCLUSION RATES")
    lines.append("-" * 40)
    topp = exclusion_results['exclusion_rates_topp']
    for p in TOP_P_VALUES:
        rates = [topp[m][ds][p] for m in topp for ds in topp[m] if p in topp[m][ds]]
        if rates:
            mean, ci_low, ci_high = compute_confidence_interval([r * 100 for r in rates])
            lines.append(f"p={p}: {mean:.2f}% (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")
    lines.append("")

    lines.append("RANK STATISTICS")
    lines.append("-" * 40)
    all_ranks = exclusion_results['all_ranks']
    combined = [r for m in all_ranks for ds in all_ranks[m] for r in all_ranks[m][ds]]
    if combined:
        lines.append(f"Total tokens: {len(combined):,}")
        lines.append(f"Median rank: {np.median(combined):.1f}")
        lines.append(f"Mean rank: {np.mean(combined):.1f} +/- {np.std(combined):.1f}")
        for pct in [25, 50, 75, 90, 95, 99]:
            lines.append(f"  {pct}th percentile: {np.percentile(combined, pct):.1f}")
    lines.append("")

    overlaps = overlap_results.get('overlaps', {})
    if overlaps:
        lines.append("CROSS-MODEL OVERLAP")
        lines.append("-" * 40)
        for k in overlap_results['metadata']['k_values']:
            vals = [overlaps[p][k]['mean'] for p in overlaps if k in overlaps[p]]
            if vals:
                mean, ci_low, ci_high = compute_confidence_interval(vals)
                lines.append(f"k={k}: mean overlap = {mean:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")

    report = "\n".join(lines)
    save_to_drive(report, f'{output_dir}/report.txt', 'text')
    return report


def main(use_small_models: bool = False, output_folder_name: str = 'truncation_experiments',
         max_samples: int = MAX_SAMPLES_PER_DATASET, skip_overlap: bool = False):
    set_seed(SEED)
    device = get_device()

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = setup_google_drive(output_folder_name)
    hf_token = get_hf_token()
    login_huggingface(hf_token)

    active_models = MODELS_SMALL if use_small_models else MODELS_FULL
    print(f"Models: {list(active_models.keys())}")

    datasets = load_and_prepare_datasets(max_samples=max_samples, min_length=MIN_SEQUENCE_LENGTH,
                                         max_length=MAX_SEQUENCE_LENGTH)

    if sum(len(t) for t in datasets.values()) == 0:
        print("No data loaded")
        return

    print("\nRunning exclusion rate experiment...")
    exclusion_results = run_exclusion_rate_experiment(
        datasets=datasets, models_config=active_models, top_k_values=TOP_K_VALUES,
        top_p_values=TOP_P_VALUES, max_length=MAX_SEQUENCE_LENGTH, output_dir=output_dir,
        hf_token=hf_token
    )

    if not skip_overlap:
        print("\nRunning overlap experiment...")
        overlap_results = run_overlap_experiment(
            datasets=datasets, models_config=active_models, exclusion_results=exclusion_results,
            k_values=[10, 20, 50], max_samples=100, max_length=128, output_dir=output_dir,
            hf_token=hf_token
        )
    else:
        overlap_results = {'overlaps': {}, 'metadata': {'k_values': [10, 20, 50]}}

    print("\nGenerating tables...")
    generate_table_1(exclusion_results, output_dir)
    generate_table_2(exclusion_results, output_dir)
    generate_table_3(overlap_results, output_dir)
    generate_supplementary_tables(exclusion_results, overlap_results, output_dir)

    print("\nGenerating figures...")
    generate_figures(exclusion_results, output_dir)

    print("\nGenerating report...")
    report = generate_report(exclusion_results, overlap_results, output_dir)
    print(report)

    print(f"\nResults saved to: {output_dir}")


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell in ('ZMQInteractiveShell', 'Shell')
    except:
        return False


if __name__ == '__main__':
    if is_notebook():
        main()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--small', action='store_true')
        parser.add_argument('--folder', type=str, default='truncation_experiments')
        parser.add_argument('--max_samples', type=int, default=MAX_SAMPLES_PER_DATASET)
        parser.add_argument('--skip_overlap', action='store_true')
        args = parser.parse_args()
        main(use_small_models=args.small, output_folder_name=args.folder,
             max_samples=args.max_samples, skip_overlap=args.skip_overlap)
