import pandas as pd

salary = pd.read_csv('Educator_Salary.csv', low_memory=False)
mobility = pd.read_csv('Educator_Mobility.csv', low_memory=False)

salary.columns = salary.columns.str.strip()
mobility.columns = mobility.columns.str.strip()

salary['Average Total Salary'] = (
    salary['Average Total Salary'].astype(str).str.replace(',', '')
)
salary['Average Total Salary'] = pd.to_numeric(salary['Average Total Salary'], errors='coerce')

salary['Average Years of Experience'] = pd.to_numeric(
    salary['Average Years of Experience'], errors='coerce'
)

mobility['Turnover Rate'] = pd.to_numeric(mobility['Turnover Rate'], errors='coerce')
mobility['Same School Retention Rate'] = pd.to_numeric(
    mobility['Same School Retention Rate'], errors='coerce'
)

keys = [
    'School Year',
    'District Code',
    'Race',
    'Gender',
    'School Code',
    'Staff Category',
    'Job Classification'
]

salary_clean = salary.groupby(keys, as_index=False).mean(numeric_only=True)

mobility_clean = mobility.groupby(keys, as_index=False).mean(numeric_only=True)

merged = salary_clean.merge(
    mobility_clean,
    on=keys,
    how='inner'
)

print("Merged rows:", len(merged))
print("Shape:", merged.shape)
print(merged.head())

# SAVE
merged.to_csv('merged_educator_data.csv', index=False)
print("Saved as merged_educator_data.csv")