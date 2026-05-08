import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('merged_educator_data.csv', low_memory=False)
df.columns = df.columns.str.strip()

df = df[df['District Code'] != 0]
df = df[df['Job Classification'].str.contains('Teacher', na=False)]
df = df[df['Job Classification'] != 'ALL']
df = df[df['Gender'].isin(['Male', 'Female'])]

df['Average Total Salary'] = pd.to_numeric(
    df['Average Total Salary'].astype(str).str.replace(',', ''),
    errors='coerce'
)

cols = [
    'Average Years of Experience',
    'Same School Retention Rate',
    'Transfer Rate Within District',
    'Transfer Rate Between Districts',
    'Turnover Rate'
]

for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.dropna(subset=cols + ['Average Total Salary'])

# convert to decimal if needed
if df['Turnover Rate'].mean() > 1:
    df['Turnover Rate'] = df['Turnover Rate'] / 100

grouped = df.groupby('District Code')[[
    'Average Years of Experience',
    'Same School Retention Rate',
    'Transfer Rate Within District',
    'Transfer Rate Between Districts',
    'Average Total Salary',
    'Turnover Rate'
]].mean().reset_index()

X = grouped.drop(columns=['Turnover Rate', 'District Code'])
y = grouped['Turnover Rate']

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

print("\nModel R²:", r2_score(y, model.predict(X)))

d10_avg = grouped[grouped['District Code'] == 10]

if len(d10_avg) > 0:
    d10_avg = d10_avg.iloc[0]
else:
    d10_avg = grouped.mean()

mob = pd.read_csv('Educator_Mobility.csv', low_memory=False)
mob.columns = mob.columns.str.strip()

for col in [
    'Average Years of Experience',
    'Same School Retention Rate',
    'Transfer Rate Within District',
    'Transfer Rate Between Districts',
    'Average Total Salary',
    'Turnover Rate'
]:
    if col in mob.columns:
        mob[col] = pd.to_numeric(mob[col], errors='coerce')

mask_d10 = mob['District Code'] == 10

mob.loc[mask_d10, 'Average Years of Experience'] = d10_avg['Average Years of Experience']
mob.loc[mask_d10, 'Average Total Salary'] = d10_avg['Average Total Salary']

mob.loc[mask_d10, 'Same School Retention Rate'] = mob.loc[mask_d10, 'Same School Retention Rate'].fillna(
    d10_avg['Same School Retention Rate']
)

mob.loc[mask_d10, 'Transfer Rate Within District'] = mob.loc[mask_d10, 'Transfer Rate Within District'].fillna(
    d10_avg['Transfer Rate Within District']
)

mob.loc[mask_d10, 'Transfer Rate Between Districts'] = mob.loc[mask_d10, 'Transfer Rate Between Districts'].fillna(
    d10_avg['Transfer Rate Between Districts']
)

features = [
    'Average Years of Experience',
    'Same School Retention Rate',
    'Transfer Rate Within District',
    'Transfer Rate Between Districts',
    'Average Total Salary'
]

X_d10 = mob.loc[mask_d10, features]

predicted = model.predict(X_d10)

# convert to % if needed
if mob['Turnover Rate'].dropna().mean() > 1:
    predicted = predicted * 100

missing_mask = mask_d10 & mob['Turnover Rate'].isna()

mob.loc[missing_mask, 'Turnover Rate'] = predicted[missing_mask[mask_d10].values]

mob.to_csv('Educator_Mobility_ALL_WITH_D10_FILLED.csv', index=False)

print("\n--- DONE ---")
print("Total rows in dataset:", len(mob))
print("District 10 rows:", mask_d10.sum())
print("Filled missing D10 turnover rows:", missing_mask.sum())
print("\nSaved as: Educator_Mobility_ALL_WITH_D10_FILLED.csv")