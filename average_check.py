human_scores = []

import pandas as pd

df = pd.read_csv('/local/scratch-2/sa2200/ezswitch/analysis/Dissertation_Full_Data_FINAL_CLEAN.csv')

metrics = ['Naturalness', 'Likelihood']

for _, row in df.iterrows():
    
    is_human_config = 'translation' in str(row['Config']).lower()
    is_human_group = 'human' in str(row['Group']).lower()
    is_ai = any(m in str(row['Config']).lower() for m in ['gemma', 'llama', 'mistral'])

    if (is_human_config or is_human_group) and not is_ai:
        human_scores.append(row['Likelihood'])

if human_scores:
    final_mean = sum(human_scores) / len(human_scores)
    print(f"Strict Human Mean: {final_mean:.2f}")
    print(f"Total Human Ratings Found: {len(human_scores)}")
else:
    print("No Human Reference items found with strict settings.")
