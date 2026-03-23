import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv('/local/scratch-2/sa2200/ezswitch/analysis/Dissertation_Full_Data_FINAL_CLEAN.csv')

def calculate_cohens_d_paired(x, y):
    # Cohen's d for paired samples: mean(diff) / std(diff)
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def run_paired_analysis(data, pairing_col, cond_a, cond_b):
    # Pivot the data so each row is a pair (either an Item or a Participant)
    pivot = data.groupby([pairing_col, 'Condition'])['Naturalness'].mean().unstack()
    
    # Select the two conditions and drop any missing pairs
    paired_data = pivot[[cond_a, cond_b]].dropna()
    d = calculate_cohens_d_paired(paired_data[cond_a], paired_data[cond_b])

    # Run the Paired Samples T-Test
    t_stat, p_val = stats.ttest_rel(paired_data[cond_a], paired_data[cond_b])
    
    print(stats.shapiro(paired_data[cond_a]))
    print(stats.shapiro(paired_data[cond_b]))

    print(f"Comparison: {cond_a} vs {cond_b} (Paired by {pairing_col})")
    print(f"Pairs: {len(paired_data)}")
    print(f"Means: {cond_a}={paired_data[cond_a].mean():.3f}, {cond_b}={paired_data[cond_b].mean():.3f}")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    print(f"Cohen's d: {d:.4f} ({d})")
    print(f"Shapiro-Wilk: {cond_a}: {stats.shapiro(paired_data[cond_a])}, {cond_b}: {stats.shapiro(paired_data[cond_b])}")
    print("-" * 30)

# --- RUN THE TESTS ---

# 1. By-Items (Comparing the 30 sentences)
run_paired_analysis(df, 'Item_ID', 'Baseline', 'ECT')
run_paired_analysis(df, 'Item_ID', 'Baseline', 'Matrix')

# 2. By-Subjects (Comparing the 23 participants)
run_paired_analysis(df, 'Participant', 'Baseline', 'ECT')
run_paired_analysis(df, 'Participant', 'Baseline', 'Matrix')

for model_name in ['Gemma', 'Llama', 'Mistral']:
    # 1. Filter the data for one model
    model_df = df[df['Model'] == model_name]
    
    # 2. Pivot to get the pairs (By-Items)
    pairs = model_df.groupby(['Item_ID', 'Condition'])['Naturalness'].mean().unstack()
    
    print(stats.shapiro(pairs['Baseline']))
    print(stats.shapiro(pairs['ECT']))

    d = calculate_cohens_d_paired(pairs['Baseline'], pairs['ECT'])

    # 3. Run the test
    t_stat, p_val = stats.ttest_rel(pairs['Baseline'], pairs['ECT'])
    
    print(f"MODEL: {model_name}")
    print(f"Baseline vs ECT: p-value = {p_val:.4f}")
    print(f"Baseline vs ECT: t-statistic = {t_stat:.4f}")
    print(f"Baseline vs ECT: Cohen's d = {d:.4f}")

for model_name in ['Gemma', 'Llama', 'Mistral']:
    # 1. Filter the data for one model
    model_df = df[df['Model'] == model_name]
    
    # 2. Pivot to get the pairs (By-Items)
    pairs = model_df.groupby(['Item_ID', 'Condition'])['Naturalness'].mean().unstack()
    
    print(stats.shapiro(pairs['Baseline']))
    print(stats.shapiro(pairs['Matrix']))

    d = calculate_cohens_d_paired(pairs['Baseline'], pairs['Matrix'])

    # 3. Run the test
    t_stat, p_val = stats.ttest_rel(pairs['Baseline'], pairs['Matrix'])
    
    print(f"MODEL: {model_name}")
    print(f"Baseline vs MLF: p-value = {p_val:.4f}")
    print(f"Baseline vs MLF: t-statistic = {t_stat:.4f}")
    print(f"Baseline vs MLF: Cohen's d = {d:.4f}")

    