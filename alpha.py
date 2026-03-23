import pandas as pd
import nltk
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import interval_distance

df = pd.read_csv('/local/scratch-2/sa2200/ezswitch/analysis/Dissertation_Full_Data_FINAL_CLEAN.csv')

raw_unique = df['Sentence_Text'].nunique()
clean_unique = df['Sentence_Text'].str.replace('`', '').str.strip().nunique()

# This will show you which strings appeared more than once in your RAW data
duplicate_sentences = df[df.duplicated(subset=['Sentence_Text'], keep=False)]
print(duplicate_sentences['Sentence_Text'].value_counts())

def calculate_final_alpha(df, score_column):
    # Pre-processing: Standardize text to ensure "overlaps" are caught
    df['Clean_Text'] = df['Sentence_Text'].str.replace('`', '').str.strip()
    
    # Create a unique Numerical ID for every unique sentence string
    # This replaces your non-unique Item_ID
    df['Unique_Sentence_ID'] = df.groupby('Clean_Text').ngroup()
    
    # Filter out any rows missing a score
    clean_df = df.dropna(subset=[score_column])
    
    # Format into (Coder, Item, Label) triplets
    triplets = []
    for _, row in clean_df.iterrows():
        triplets.append((
            row['Participant'], 
            row['Unique_Sentence_ID'], 
            float(row[score_column])
        ))
    
    # Initialize task with interval_distance (Standard for Ordinal/Likert scales)
    task = AnnotationTask(data=triplets, distance=interval_distance)
    
    return task.alpha()

# --- RUN CALCULATION ---
# Treat both as ordinal as per your requirement
alpha_nat = calculate_final_alpha(df, 'Naturalness')
alpha_lik = calculate_final_alpha(df, 'Likelihood')

print(f"Krippendorff's Alpha (Naturalness): {alpha_nat:.4f}")
print(f"Krippendorff's Alpha (Likelihood): {alpha_lik:.4f}")

# --- VALIDATION CHECK ---
# It's good to report how many unique items were actually used
unique_items = df['Clean_Text'].nunique()
print(f"Calculation based on {unique_items} unique sentences.")

def calculate_alpha_by_model(df, score_column):
    results = {}
    
    # Pre-process once
    df['Clean_Text'] = df['Sentence_Text'].str.replace('`', '').str.strip()
    df['Unique_Sentence_ID'] = df.groupby('Clean_Text').ngroup()
    
    # Iterate through each model (e.g., Gemma, Llama, Human)
    for model_name in df['Model'].unique():
        model_df = df[df['Model'] == model_name].dropna(subset=[score_column])
        
        # We need at least some overlaps to calculate Alpha
        if len(model_df['Participant'].unique()) > 1:
            triplets = []
            for _, row in model_df.iterrows():
                triplets.append((row['Participant'], row['Unique_Sentence_ID'], float(row[score_column])))
            
            task = AnnotationTask(data=triplets, distance=interval_distance)
            results[model_name] = task.alpha()
        else:
            results[model_name] = "Not enough data"
            
    return results

# --- Run the breakdown ---
nat_by_model = calculate_alpha_by_model(df, 'Naturalness')
lik_by_model = calculate_alpha_by_model(df, 'Likelihood')

print("Alpha for Naturalness by Model:")
for model, val in nat_by_model.items():
    print(f" - {model}: {val:.4f}" if isinstance(val, float) else f" - {model}: {val}")

print("Alpha for Likelihood by Model:")
for model, val in lik_by_model.items():
    print(f" - {model}: {val:.4f}" if isinstance(val, float) else f" - {model}: {val}")

def calculate_alpha_by_condition(df, score_column):
    results = {}
    
    # Standardize and ID
    df['Clean_Text'] = df['Sentence_Text'].str.replace('`', '').str.strip()
    df['Unique_Sentence_ID'] = df.groupby('Clean_Text').ngroup()
    
    # Group by Condition
    for cond in df['Condition'].unique():
        cond_df = df[df['Condition'] == cond].dropna(subset=[score_column])
        
        # Check for sufficient overlaps (at least 2 raters, multiple items)
        if len(cond_df['Participant'].unique()) > 1 and cond_df['Unique_Sentence_ID'].nunique() > 1:
            triplets = []
            for _, row in cond_df.iterrows():
                triplets.append((row['Participant'], row['Unique_Sentence_ID'], float(row[score_column])))
            
            task = AnnotationTask(data=triplets, distance=interval_distance)
            results[cond] = task.alpha()
        else:
            results[cond] = "Insufficient overlaps"
            
    return results

# --- Run ---
nat_by_cond = calculate_alpha_by_condition(df, 'Naturalness')
lik_by_cond = calculate_alpha_by_condition(df, 'Likelihood')

print("Alpha for Naturalness by Condition:")
for cond, val in nat_by_cond.items():
    print(f" - {cond}: {val:.4f}" if isinstance(val, float) else f" - {cond}: {val}")

print("Alpha for Likelihood by Condition:")
for cond, val in lik_by_cond.items():
    print(f" - {cond}: {val:.4f}" if isinstance(val, float) else f" - {cond}: {val}")

def calculate_alpha_by_direction(df, score_column):
    results = []
    
    # Standardize
    df['Clean_Text'] = df['Sentence_Text'].str.replace('`', '').str.strip()
    df['Unique_Sentence_ID'] = df.groupby('Clean_Text').ngroup()

    for model in df['Model'].unique():
        for lang in df['Source_Lang'].unique():
            # Filter for specific Model + Source Direction
            subset = df[(df['Model'] == model) & (df['Source_Lang'] == lang)].dropna(subset=[score_column])
            
            if len(subset['Participant'].unique()) > 1 and subset['Unique_Sentence_ID'].nunique() > 1:
                triplets = [(r['Participant'], r['Unique_Sentence_ID'], float(r[score_column])) for _, r in subset.iterrows()]
                task = AnnotationTask(data=triplets, distance=interval_distance)
                alpha = task.alpha()
                results.append({'Model': model, 'Direction': lang, 'Alpha': round(alpha, 4)})
                
    return pd.DataFrame(results)

# Run for Naturalness
direction_alpha = calculate_alpha_by_direction(df, 'Naturalness')
print(direction_alpha)

direction_alpha_lik = calculate_alpha_by_direction(df, 'Likelihood')
print(direction_alpha_lik)

def calculate_granular_alpha_fixed(df, score_column):
    results = []
    
    # 1. Clean and ID
    df['Clean_Text'] = df['Sentence_Text'].str.replace('`', '').str.strip()
    df['Unique_Sentence_ID'] = df.groupby('Clean_Text').ngroup()
    
    # 2. Define our groups
    models = df['Model'].unique()
    conditions = df['Condition'].unique()
    # We use the specific directions found in the data
    directions = ['Hindi', 'English'] 

    for model in models:
        for cond in conditions:
            # For Human_Ref, we might need to bypass the 'Direction' filter 
            # if they are tagged as N/A
            current_directions = directions
            if cond == 'Human_Ref':
                # Calculate once for all Human Refs, or try to split if you know which are which
                current_directions = [None] 

            for lang in current_directions:
                if lang is None:
                    subset = df[(df['Condition'] == 'Human_Ref')].dropna(subset=[score_column])
                    dir_label = "Total"
                else:
                    subset = df[(df['Model'] == model) & 
                                (df['Condition'] == cond) & 
                                (df['Source_Lang'] == lang)].dropna(subset=[score_column])
                    dir_label = lang
                
                if subset['Participant'].nunique() > 1 and subset['Unique_Sentence_ID'].nunique() > 1:
                    triplets = [(r['Participant'], r['Unique_Sentence_ID'], float(r[score_column])) 
                                for _, r in subset.iterrows()]
                    task = AnnotationTask(data=triplets, distance=interval_distance)
                    alpha = task.alpha()
                    results.append({'Model': model, 'Condition': cond, 'Direction': dir_label, 'Alpha': round(alpha, 4)})
                
    return pd.DataFrame(results)

# Run the fixed version
final_granular_nat = calculate_granular_alpha_fixed(df, 'Naturalness')
print(final_granular_nat)

final_granular_lik = calculate_granular_alpha_fixed(df, 'Likelihood')
print(final_granular_lik)