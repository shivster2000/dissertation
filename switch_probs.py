import os
import pandas as pd
import csv
from transformers import pipeline

lid_pipeline = pipeline(
    "token-classification", 
    model="l3cube-pune/hing-bert-lid", 
    aggregation_strategy="simple",
    device=0 
)

def process_code_switching_data(input_dir):
    all_switch_data = []

    print("Scanning CSVs and identifying switch boundaries...")
    
    for filename in os.listdir(input_dir):
        if not filename.endswith("scores.csv"):
            continue
            
        parts = filename.replace(".csv", "").split("_")
        if len(parts) < 3: 
            continue
            
        model = parts[1]
        condition = parts[2]
        lang = parts[3] 

        try:
            df = pd.read_csv(
                os.path.join(input_dir, filename), 
                sep=None, engine='python', quoting=csv.QUOTE_NONE, on_bad_lines='skip'
            )
            df.columns = [c.strip().replace('"', '').replace("'", "") for c in df.columns]
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        if 'prompt_id' not in df.columns or 'token_text' not in df.columns:
            continue

        grouped = df.groupby('prompt_id')
        
        for sentence_idx, group in grouped:
            
            group = group.sort_values('token_index').copy()
            
            # 1. Reconstruct the full sentence while tracking character offsets for each sub-token
            full_text = ""
            token_offsets = []
            
            for idx, row in group.iterrows():
                # Clean the sub-token to match how it will look in the full string
                clean_tok = str(row['token_text']).replace('\u2581', ' ')
                
                start_char = len(full_text)
                full_text += clean_tok
                end_char = len(full_text)
                
                token_offsets.append({
                    'index': idx,
                    'midpoint': (start_char + end_char) / 2 # Use midpoint for safe overlap checking
                })

            # 2. Pass the FULL context to HingBERT
            # aggregation_strategy="simple" will return start/end character indices
            pipeline_out = lid_pipeline(full_text)
            
            # If pipeline fails or returns empty, skip
            if not isinstance(pipeline_out, list) or len(pipeline_out) == 0:
                continue

            # 3. Map the pipeline entities back to the specific LLM tokens using offsets
            label_map = {}
            for t in token_offsets:
                assigned_label = 'UNK'
                # Find which HingBERT entity covers this token's midpoint
                for entity in pipeline_out:
                    if entity['start'] <= t['midpoint'] <= entity['end']:
                        assigned_label = entity.get('entity_group', entity.get('entity', 'UNK'))
                        break
                label_map[t['index']] = assigned_label


            # Apply labels to dataframe
            group['lang_id'] = group.index.map(label_map)
            
            # 4. Shift columns to calculate the switch
            group['prev_lang_id'] = group['lang_id'].shift(1)
            group['prev_prob'] = group['probability'].shift(1)

            print(f"Sentence ID: {sentence_idx}")
            print(f"Full Reconstructed Text: '{full_text}'\n")
            
            # Create a clean subset of the dataframe to print
            # Replacing the literal spaces so they are easier to read in the terminal
            display_df = group[['token_index', 'token_text', 'probability', 'lang_id', 'prev_lang_id']].copy()
            display_df['token_text'] = display_df['token_text'].astype(str).str.replace('\u2581', '_')
            
            # Print the dataframe nicely
            print(display_df.to_string(index=False, justify='left'))
            print("\n" + "="*60 + "\n")
            
            
            # 5. Detect the switch point
            # Ignore switches to/from UNK or punctuation (often labeled 'O' by BERT)
            valid_labels = ['EN', 'HI'] # Add any other specific labels your model uses
            
            is_switch = (
                (group['lang_id'] != group['prev_lang_id']) & 
                (group['prev_lang_id'].notna()) &
                (group['lang_id'].str.upper().isin(valid_labels)) &
                (group['prev_lang_id'].str.upper().isin(valid_labels))
            )
            
            switch_points = group[is_switch]
            
            for _, row in switch_points.iterrows():
                all_switch_data.append({
                    'Model': model.capitalize(),
                    'Condition': condition.capitalize(),
                    'Source': lang.capitalize(),
                    'From_Lang': row['prev_lang_id'].upper(),
                    'To_Lang': row['lang_id'].upper(),
                    'Prob_Before': row['prev_prob'],
                    'Prob_After': row['probability']
                })


    return pd.DataFrame(all_switch_data)

# ==========================================
# 3. LATEX TABLE GENERATOR
# ==========================================
def generate_latex_table(df):
    if df.empty:
        print("No switch points detected. Check your LangID pipeline.")
        return

    # Calculate the Drop (Delta) for every individual switch
    df['Prob_Drop'] = df['Prob_Before'] - df['Prob_After']

    # Group and aggregate the metrics
    summary = df.groupby(['Model', 'Condition', 'Source']).agg(
        N_Switches=('Prob_Before', 'count'),
        Avg_Prob_Before=('Prob_Before', 'mean'),
        Avg_Prob_After=('Prob_After', 'mean'),
        Avg_Drop=('Prob_Drop', 'mean')
    ).reset_index()

    # Format numbers for LaTeX
    summary['Avg_Prob_Before'] = summary['Avg_Prob_Before'].map('{:.3f}'.format)
    summary['Avg_Prob_After'] = summary['Avg_Prob_After'].map('{:.3f}'.format)
    summary['Avg_Drop'] = summary['Avg_Drop'].map('{:+.3f}'.format) # includes + or - sign

    # Build the LaTeX string manually for perfect dissertation formatting (booktabs)
    latex_str = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Average LLM Confidence Before and After Romanized Code-Switching Points}",
        "\\label{tab:switch_probs}",
        "\\begin{tabular}{lllc rrr}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Condition} & \\textbf{Source} & \\textbf{$N$} & \\textbf{Prob. Before} & \\textbf{Prob. After} & \\textbf{$\\Delta$ Drop} \\\\",
        "\\midrule"
    ]

    # Iterate through the rows and add them to the table
    current_model = ""
    for _, row in summary.iterrows():
        # Only print the model name once per block for a cleaner look
        mod_display = row['Model'] if row['Model'] != current_model else ""
        current_model = row['Model']
        
        line = f"{mod_display} & {row['Condition']} & {row['Source']} & {row['N_Switches']} & {row['Avg_Prob_Before']} & {row['Avg_Prob_After']} & {row['Avg_Drop']} \\\\"
        latex_str.append(line)

    latex_str.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    print("\n" + "="*50)
    print("="*50 + "\n")
    print("\n".join(latex_str))
    
    # Save a CSV backup just in case
    summary.to_csv("switch_point_metrics_latex_backup.csv", index=False)


if __name__ == "__main__":
    # Update this path to where your final_data folder is
    INPUT_PATH = "/local/scratch-2/sa2200/ezswitch/output/final_data/"
    
    # Run the pipeline
    final_df = process_code_switching_data(INPUT_PATH)
    generate_latex_table(final_df)