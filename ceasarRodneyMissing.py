import pandas as pd

df = pd.read_csv('merged_educator_data.csv', low_memory=False)
df.columns = df.columns.str.strip()

df_10_2025 = df[
    (df['District Code'] == 10) &
    (df['School Year'] == 2025)
]

missing_turnover = df_10_2025[
    df_10_2025['Turnover Rate'].isna()
]

print("\nRows for District 10 (2025) with missing Turnover Rate:")
print(missing_turnover)

print("\nNumber of missing rows:", len(missing_turnover))

missing_turnover.to_csv("district10_2025_missing_turnover.csv", index=False)

print("\nSaved to: district10_2025_missing_turnover.csv")