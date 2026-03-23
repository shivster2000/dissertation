import pandas as pd


df = pd.read_csv('/local/scratch-2/sa2200/ezswitch/analysis/Dissertation_Full_Data_FINAL_CLEAN.csv')

filtered_df = df[df['Changes'].notna() & (df['Changes'] != "")]

filtered_df.to_csv('filtered_changes.csv', index=False)

print(f"Filtered {len(df)} rows down to {len(filtered_df)}.")