import pandas as pd
import numpy as np
import string
from collections import Counter
from transformers import pipeline
import nltk
from tqdm import tqdm
tqdm.pandas()

df = pd.read_csv('/local/scratch-2/sa2200/ezswitch/analysis/Dissertation_Full_Data_FINAL_CLEAN.csv')

original_count = len(df)
df = df.drop_duplicates(subset=['Sentence_Text'])
new_count = len(df)

print(f"Removed {original_count - new_count} duplicate sentences.")

lid_pipeline = pipeline(
    "token-classification", 
    model="l3cube-pune/hing-bert-lid", 
    aggregation_strategy="simple",
    device=0 # Set to 0 if you have a GPU, otherwise -1 for CPU
)
def calculate_exact_cmi(text):
    if pd.isna(text) or text.strip() == "":
        return 0
    
    # Pre-clean: Remove the prompt if it's still there
    if ":" in text:
        text = text.split(":")[-1].strip()

    # 2. Get predictions
    # Typical labels: 'en', 'hi', 'other' (for punctuation/numbers)
    results = lid_pipeline(text)
    
    n = 0 # Total tokens
    u = 0 # Language-independent tokens (punctuation/numbers)
    lang_counts = Counter()

    for chunk in results:
        # Get the label (EN, HI, or OTHER)
        label = chunk.get('entity_group', chunk.get('entity', 'other')).lower()
        words_in_chunk = chunk['word'].split()
        
        for word in words_in_chunk:
            if word.startswith("##"):
                continue

            n += 1
            # Strip punctuation to check if it's a real word or just 'u'
            clean_word = word.strip(string.punctuation)
            
            # Identify if the token is language-independent (u)
            if not clean_word or clean_word.isdigit() or label == 'other':
                u += 1
            elif label in ['en', 'hi', 'label_0', 'label_1']:
                # Valid language token
                lang_counts[label] += 1
            else:
                # Fallback: if label is unknown, treat as u
                u += 1

    # 3. Apply the conditional formula
    if n > u:
        max_wi = lang_counts.most_common(1)[0][1] if lang_counts else 0
        
        cmi = 100 * (1 - (max_wi / (n - u)))
    else:
        cmi = 0
    return cmi
    

# Apply to your cleaned dataframe
df['CMI'] = df['Sentence_Text'].progress_apply(calculate_exact_cmi)

print("\n" + "="*30)
print("TOP 5 MOST MIXED SENTENCES")
print("="*30)
# Showing the highest CMI values
top_mixed = df.sort_values(by='CMI', ascending=False).head(5)
for idx, row in top_mixed.iterrows():
    print(f"CMI: {row['CMI']}% | Text: {row['Sentence_Text'][:100]}...")

print("\n" + "="*30)
print("5 MONOLINGUAL SENTENCES (CMI = 0)")
print("="*30)
mono = df[df['CMI'] == 0].head(5)
for idx, row in mono.iterrows():
    print(f"Text: {row['Sentence_Text'][:100]}...")

summary_df = df.groupby(['Condition', 'Model'])['CMI'].mean().unstack()

df.to_csv('/local/scratch-2/sa2200/ezswitch/analysis/Dissertation_Data_WITH_CMI.csv', index=False)

# Export to LaTeX code
latex_table = summary_df.to_latex(
    index=True, 
    float_format="%.2f",
    caption="Mean Code-Mixing Index (CMI) across Conditions and Models",
    label="tab:cmi_results"
)

print(latex_table)