import pandas as pd

df = pd.read_csv('/local/scratch-2/sa2200/ezswitch/analysis/Dissertation_Full_Data_FINAL_CLEAN.csv')

metrics = ['Naturalness', 'Likelihood']

stats_by_condition_direction_model = df.groupby(['Condition', 'Source_Lang', 'Model'])[metrics].agg(['mean', 'std', 'count']).round(3)

stats_by_condition_direction_model.to_csv('table_condition_direction_model_stats.csv')