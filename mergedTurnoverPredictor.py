import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

df['Average Years of Experience'] = pd.to_numeric(df['Average Years of Experience'], errors='coerce')
df['Transfer Rate Within District'] = pd.to_numeric(df['Transfer Rate Within District'], errors='coerce')
df['Transfer Rate Between Districts'] = pd.to_numeric(df['Transfer Rate Between Districts'], errors='coerce')
df['Turnover Rate'] = pd.to_numeric(df['Turnover Rate'], errors='coerce')

df = df[df['Average Years of Experience'] > 0]

df = df.dropna(subset=[
    'Average Years of Experience',
    'Average Total Salary',
    'Transfer Rate Within District',
    'Transfer Rate Between Districts',
    'Turnover Rate'
])

cleaned = []

for d in df['District Code'].unique():
    temp = df[df['District Code'] == d].copy()

    Q1 = temp['Average Total Salary'].quantile(0.25)
    Q3 = temp['Average Total Salary'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    temp = temp[(temp['Average Total Salary'] >= lower) &
                (temp['Average Total Salary'] <= upper)]

    cleaned.append(temp)

df = pd.concat(cleaned)

grouped = df.groupby('District Code')[[
    'Average Years of Experience',
    'Transfer Rate Within District',
    'Transfer Rate Between Districts',
    'Average Total Salary',
    'Turnover Rate'
]].mean().reset_index()

X = grouped[[
    'Average Years of Experience',
    'Transfer Rate Within District',
    'Transfer Rate Between Districts',
    'Average Total Salary'
]]

y = grouped['Turnover Rate']

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)

y_pred = model.predict(X)
grouped['Predicted Turnover'] = y_pred

r2 = r2_score(y, y_pred)
print("\nR^2:", r2)

def salary_needed_for_reduction(model, X, target_reduction=1.0, step=1000, max_steps=50):

    base_pred = model.predict(X)
    base_mean = np.mean(base_pred)

    current_X = X.copy()
    total_salary_increase = 0

    last_reduction = 0

    for _ in range(max_steps):

        current_X['Average Total Salary'] += step
        total_salary_increase += step

        new_pred = model.predict(current_X)
        new_mean = np.mean(new_pred)

        reduction = base_mean - new_mean
        marginal_gain = reduction - last_reduction

        # plateau detection
        if marginal_gain < 0.001:
            print("\n⚠️ Plateau reached: salary no longer meaningfully reduces turnover.")
            break

        last_reduction = reduction

        if reduction >= target_reduction:
            return total_salary_increase, reduction

    return total_salary_increase, last_reduction

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nFeature Importance:")
print(importances)

sns.set_theme(style="whitegrid", context="talk")

fig, ax = plt.subplots(figsize=(12, 8))

# compute residuals
grouped['residual'] = grouped['Turnover Rate'] - grouped['Predicted Turnover']

# color mapping:
# negative residual → model overpredicted → GOOD outcome (green)
# positive residual → model underpredicted → BAD outcome (red)

colors = grouped['residual']

scatter = ax.scatter(
    grouped['Turnover Rate'],
    grouped['Predicted Turnover'],
    c=colors,
    cmap="RdYlGn_r",   # reversed so green = better, red = worse
    s=90,
    edgecolor="black",
    linewidths=0.5,
    alpha=0.85
)

# perfect prediction line
ax.plot(
    [grouped['Turnover Rate'].min(), grouped['Turnover Rate'].max()],
    [grouped['Turnover Rate'].min(), grouped['Turnover Rate'].max()],
    linestyle="--",
    color="gray",
    linewidth=2
)

# label ONLY extreme cases (top 8 worst + best)
top_worst = grouped.sort_values('residual', ascending=False).head(5)
top_best = grouped.sort_values('residual', ascending=True).head(5)

label_set = pd.concat([top_worst, top_best])

for _, row in label_set.iterrows():
    ax.text(
        row['Turnover Rate'],
        row['Predicted Turnover'],
        str(int(row['District Code'])),
        fontsize=9,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
    )

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Turnover Performance Gap (Actual − Expected)")

ax.set_xlabel("Actual Turnover Rate")
ax.set_ylabel("Predicted Turnover Rate")
ax.set_title("District Turnover Performance Relative to Model Expectations")

plt.tight_layout()
plt.show()

print("\n--- SALARY PLANNER ---")

target = float(input("Enter desired turnover reduction (% points): "))

salary_needed, achieved = salary_needed_for_reduction(
    model,
    X,
    target_reduction=target
)

print("\n--- RESULT ---")
print(f"Target reduction: {target}%")
print(f"Estimated salary increase needed: ${salary_needed:,.2f}")
print(f"Achieved reduction in model: {achieved:.4f}%")