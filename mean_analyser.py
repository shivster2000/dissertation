import pandas as pd

df = pd.read_csv('/local/scratch-2/sa2200/ezswitch/analysis/Dissertation_Full_Data_FINAL_CLEAN.csv')

metrics = ['Naturalness', 'Likelihood']

stats_by_condition = df.groupby('Condition')[metrics].agg(['mean', 'std', 'count']).round(3)

stats_by_model = df.groupby('Model')[metrics].agg(['mean', 'std', 'count']).round(3)

stats_by_condition_direction = df.groupby(['Condition', 'Source_Lang'])[metrics].agg(['mean', 'std', 'count']).round(3)
stats_detailed = df.groupby(['Condition', 'Model'])[metrics].agg(['mean', 'std', 'count']).round(3)

print("AVERAGES PER CONDITION:")
print(stats_by_condition)

print("\nAVERAGES PER MODEL:")
print(stats_by_model)

print("\nDETAILED BREAKDOWN:")
print(stats_detailed)

stats_by_condition.to_csv('table_condition_stats.csv')
stats_by_model.to_csv('table_model_stats.csv')
stats_detailed.to_csv('table_detailed_stats.csv')
stats_by_condition_direction.to_csv('table_condition_direction_stats.csv')