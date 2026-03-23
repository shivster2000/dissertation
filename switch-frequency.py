import json
import pandas as pd
import os
import re
from difflib import SequenceMatcher
import unicodedata
from ai4bharat.transliteration import XlitEngine
import torch
import argparse


torch.serialization.add_safe_globals([argparse.Namespace])

e = XlitEngine( beam_width=4, rescore=False, src_script_type = "indic")

#os.environ['PYTHONPATH'] += ":/content/fairseq/"

def get_all_candidates(hindi_word, e):
    out = e.translit_word(hindi_word, lang_code='hi', topk=4)
    
    if isinstance(out, dict):
        return [w.lower().strip() for w in out['hi']]
    return [w.lower().strip() for w in out]

def clean_text(text):
    return re.sub(r'[^\w\s]', '', str(text).lower())

def is_fuzzy_match(word1, word2, threshold=0.85):
    """Checks if two words are 'close enough' in Romanized spelling."""
    return SequenceMatcher(None, word1.lower(), word2.lower()).ratio() >= threshold

def strip_accents(s):
    """Converts 'khānā' to 'khana' to match model output."""
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def process_condition_logs(directory_path):
    all_data = []

    file_map = {
        '/local/scratch-2/sa2200/ezswitch/logs/en_matrix_20260216_201909_gemma.jsonl': 'MLF-Gemma-EN',
        '/local/scratch-2/sa2200/ezswitch/logs/en_matrix_20260217_004441_llama.jsonl': 'MLF-Llama-EN',
        '/local/scratch-2/sa2200/ezswitch/logs/en_matrix_20260217_004949_mistral.jsonl': 'MLF-Mistral-EN',
        '/local/scratch-2/sa2200/ezswitch/logs/hi_matrix_20260216_201159_gemma.jsonl': 'MLF-Gemma-HI',
        '/local/scratch-2/sa2200/ezswitch/logs/hi_matrix_20260217_000103_mistral.jsonl': 'MLF-Mistral-HI',
        '/local/scratch-2/sa2200/ezswitch/logs/hi_matrix_20260217_004017_llama.jsonl': 'MLF-Llama-HI',
    }

    for filename, condition in file_map.items():
        file_path = filename
        
        if not os.path.exists(file_path):
            print(f"Skipping {filename}: File not found.")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Extract raw fields
                    output = data.get('generated_output', '')
                    suggested_el = data.get('extracted_el_words', [])
                    output_tokens = clean_text(output).lower().split()
                    
                    # Cleaning for matching
                    clean_output = clean_text(output)
                    
                    # Calculate Compliance
                    used_words = []
                    compliance_rate = 0
                    if suggested_el:
                        for word in suggested_el:
                            
                            candidates = get_all_candidates(word, e)
                            
                            match_found = False
                            for cand in candidates:
                                if cand in output_tokens:
                                    match_found = True
                                    break

                                for token in output_tokens:
                                    if is_fuzzy_match(cand, token, threshold=0.85):
                                        match_found = True
                                        break
                                    
                                if match_found: 
                                        break

                            if match_found:
                                used_words.append(word)
                        compliance_rate = len(used_words) / len(suggested_el)
                    else:
                        compliance_rate = None

                    # Store results with metadata
                    all_data.append({
                        'index': data.get('index'),
                        'condition': condition,
                        'original_ml': data.get('original_ml_sentence'),
                        'output': output,
                        'direction': 'hi' if 'HI' in condition else 'en',
                        'model': condition.split('-')[1],
                        'suggested_count': len(suggested_el),
                        'used_count': len(used_words),
                        'compliance_rate': compliance_rate,
                        'ignored_words': list(set(suggested_el) - set(used_words))
                    })
                except json.JSONDecodeError:
                    continue

    return pd.DataFrame(all_data)

def process_ect_condition_logs(directory_path):
    all_data = []

    file_map = {
        '/local/scratch-2/sa2200/ezswitch/logs/hi_equivalence_20260216_202513_gemma.jsonl': 'ECT-Gemma-HI',
        '/local/scratch-2/sa2200/ezswitch/logs/hi_equivalence_20260216_202800_gemma.jsonl': 'ECT-Gemma-EN',
        '/local/scratch-2/sa2200/ezswitch/logs/hi_equivalence_20260216_234746_mistral.jsonl': 'ECT-Mistral-HI',
        '/local/scratch-2/sa2200/ezswitch/logs/hi_equivalence_20260216_235008_mistral.jsonl': 'ECT-Mistral-EN',
        '/local/scratch-2/sa2200/ezswitch/logs/hi_equivalence_20260217_003606_llama.jsonl': 'ECT-Llama-HI',
        '/local/scratch-2/sa2200/ezswitch/logs/hi_equivalence_20260217_003751_llama.jsonl': 'ECT-Llama-EN',
    }

    for filename, condition in file_map.items():
        file_path = filename
        
        if not os.path.exists(file_path):
            print(f"Skipping {filename}: File not found.")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Extract raw fields
                    output = data.get('generated_output', '')
                    suggested_el = data.get('extracted_src_words', [])
                    output_tokens = clean_text(output).lower().split()

                    # Cleaning for matching
                    clean_output = clean_text(output)
                    
                    # Calculate Compliance
                    used_words = []
                    compliance_rate = 0
                    if suggested_el:
                        for word in suggested_el:
                            
                            candidates = get_all_candidates(word, e)
                            
                            match_found = False
                            for cand in candidates:
                                if cand in output_tokens:
                                    match_found = True
                                    break

                                for token in output_tokens:
                                    if is_fuzzy_match(cand, token, threshold=0.85):
                                        match_found = True
                                        break
                                    
                                if match_found: 
                                        break

                            if match_found:
                                used_words.append(word)
                        compliance_rate = len(used_words) / len(suggested_el)
                    else:
                        compliance_rate = None

                
                    all_data.append({
                        'index': data.get('index'),
                        'condition': condition,
                        'original_src': data.get('original_src_sentence'),
                        'original_tgt': data.get('original_tgt_sentence'),
                        'output': output,
                        'direction': 'hi' if 'HI' in condition else 'en',
                        'model': condition.split('-')[1],
                        'suggested_count': len(suggested_el),
                        'used_count': len(used_words),
                        'compliance_rate': compliance_rate,
                        'ignored_words': list(set(suggested_el) - set(used_words))
                    })
                except json.JSONDecodeError:
                    continue

    return pd.DataFrame(all_data)

df_compliance = process_condition_logs('/local/scratch-2/sa2200/ezswitch/logs')

print(df_compliance)

avg_compliance = df_compliance.groupby('model')['compliance_rate'].mean()
print("Mean Compliance Rate:\n", avg_compliance)

avg_compliance_direction = df_compliance.groupby(['model', 'direction'])['compliance_rate'].mean()
print("\nMean Compliance Rate by Direction:\n", avg_compliance_direction)


df_ect_compliance = process_ect_condition_logs('/local/scratch-2/sa2200/ezswitch/logs')

print(df_ect_compliance)

avg_compliance = df_ect_compliance.groupby('model')['compliance_rate'].mean()
print("Mean ECT Compliance Rate:\n", avg_compliance)

avg_compliance_direction = df_ect_compliance.groupby(['model', 'direction'])['compliance_rate'].mean()
print("\nMean ECT Compliance Rate by Direction:\n", avg_compliance_direction)


 
"""

# 2. Load your 583 annotated lines
df_manual = pd.read_csv('/local/scratch-2/sa2200/ezswitch/analysis/filtered_changes.csv')

# 3. Merge on the Hindi sentence (ensure both are stripped of whitespace)
df_manual['original_ml_sentence'] = df_manual['original_ml_sentence'].str.strip()
df_compliance['original_ml_sentence'] = df_compliance['original_ml_sentence'].str.strip()

master_df = pd.merge(df_manual, df_compliance, on='original_ml_sentence', how='inner')

def calculate_utility(row):
    suggested = [word.lower() for word in row['extracted_el_words']]
    output = row['generated_output'].lower()
    
    # Check which suggested words actually appear in the output
    used_words = [word for word in suggested if word in output]
    
    utilization_count = len(used_words)
    total_suggested = len(suggested)
    
    # Handle division by zero if no EL words were suggested
    utilization_rate = (utilization_count / total_suggested) if total_suggested > 0 else 0
    
    return pd.Series([utilization_rate, used_words, list(set(suggested) - set(used_words))])

master_df[['util_rate', 'used_el', 'ignored_el']] = master_df.apply(calculate_utility, axis=1)

# Average utilization rate per human error category
analysis = master_df.groupby('human_category')['util_rate'].mean()
print(analysis)

"""

"""'/local/scratch-2/sa2200/ezswitch/logs/hi_baseline_20260216_195743_gemma.jsonl': 'Baseline-Gemma-HI',
'/local/scratch-2/sa2200/ezswitch/logs/hi_baseline_20260216_200035_gemma.jsonl': 'Baseline-Gemma-EN',
'/local/scratch-2/sa2200/ezswitch/logs/hi_baseline_20260216_232507_mistral.jsonl': 'Baseline-Mistral-HI',
'/local/scratch-2/sa2200/ezswitch/logs/hi_baseline_20260216_232739_mistral.jsonl': 'Baseline-Mistral-EN',
'/local/scratch-2/sa2200/ezswitch/logs/hi_baseline_20260217_003141_llama.jsonl': 'Baseline-Llama-HI',
'/local/scratch-2/sa2200/ezswitch/logs/hi_baseline_20260217_003336_llama.jsonl': 'Baseline-Llama-EN',
'/local/scratch-2/sa2200/ezswitch/logs/hi_equivalence_20260216_202513_gemma.jsonl': 'ECT-Gemma-HI',
'/local/scratch-2/sa2200/ezswitch/logs/hi_equivalence_20260216_202800_gemma.jsonl': 'ECT-Gemma-EN',
'/local/scratch-2/sa2200/ezswitch/logs/hi_equivalence_20260216_234746_mistral.jsonl': 'ECT-Mistral-HI',
'/local/scratch-2/sa2200/ezswitch/logs/hi_equivalence_20260216_235008_mistral.jsonl': 'ECT-Mistral-EN',
'/local/scratch-2/sa2200/ezswitch/logs/hi_equivalence_20260217_003606_llama.jsonl': 'ECT-Llama-HI',
'/local/scratch-2/sa2200/ezswitch/logs/hi_equivalence_20260217_003751_llama.jsonl': 'ECT-Llama-EN',"""