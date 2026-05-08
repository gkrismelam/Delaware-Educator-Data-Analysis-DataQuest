import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('merged_educator_data.csv', low_memory=False)
df.columns = df.columns.str.strip()

df = df[df['District Code'] != 0]
df = df[df['District Code'] != 12]
df = df[df['District Code'] <= 9000]
df = df[df['Job Classification'].str.contains('Teacher', na=False)]
df = df[df['Job Classification'] != 'ALL']

df = df[df['Gender'].isin(['Male', 'Female'])]

df['Average Total Salary'] = pd.to_numeric(
    df['Average Total Salary'].astype(str).str.replace(',', ''),
    errors='coerce'
)

df['Average Years of Experience'] = pd.to_numeric(
    df['Average Years of Experience'],
    errors='coerce'
)

df = df[df['Average Years of Experience'] > 0]

df = df.dropna(subset=['Average Years of Experience', 'Average Total Salary'])

cleaned = []

for d in df['District Code'].unique():
    temp = df[df['District Code'] == d].copy()

    Q1 = temp['Average Total Salary'].quantile(0.25)
    Q3 = temp['Average Total Salary'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    temp = temp[
        (temp['Average Total Salary'] >= lower) &
        (temp['Average Total Salary'] <= upper)
    ]

    cleaned.append(temp)

df = pd.concat(cleaned)

def linear_regression(x, y):
    x = np.array(x)
    y = np.array(y)

    x_mean = x.mean()
    y_mean = y.mean()

    m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - m * x_mean

    y_pred = m * x + b

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return m, b, r2

plt.figure(figsize=(10,6))

for d in df['District Code'].unique():
    temp = df[df['District Code'] == d]

    grouped = temp.groupby('Average Years of Experience')[
        'Average Total Salary'
    ].mean().reset_index()

    grouped = grouped.sort_values('Average Years of Experience')

    if len(grouped) < 2:
        continue

    x = grouped['Average Years of Experience'].values
    y = grouped['Average Total Salary'].values

    m, b, r2 = linear_regression(x, y)

    print(f"District {d} | R^2 = {r2:.4f}")

    # regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b

    # line graph (no scatter)
    plt.plot(x, y, alpha=0.6)
    plt.plot(x_line, y_line, linewidth=2)

    # label district
    plt.text(x[-1], y[-1], str(d), fontsize=8)

plt.title('Teacher Salary vs Experience (new.csv)')
plt.xlabel('Years of Experience')
plt.ylabel('Average Total Salary')
plt.grid(True)

y_max = df['Average Total Salary'].max()
plt.yticks(np.arange(0, y_max + 10000, 10000))

plt.show()