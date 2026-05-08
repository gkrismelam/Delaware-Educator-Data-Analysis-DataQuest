import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('merged_educator_data.csv', low_memory=False)
df.columns = df.columns.str.strip()

# Filters
df = df[df['District Code'] != 0]
df = df[df['Job Classification'].str.contains('Teacher', na=False)]
df = df[df['Job Classification'] != 'ALL']
df = df[df['Gender'].isin(['Male', 'Female'])]

# Clean numeric columns
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

# Detect format of turnover rate (percentage vs decimal)
temp_check = df['Turnover Rate'].dropna()
sample_mean = temp_check.mean()

if sample_mean > 10:
    print(f"Detected: PERCENTAGE format (mean = {sample_mean:.1f}%)")
    df['Turnover Rate'] = df['Turnover Rate'] / 100
    turnover_multiplier = 100
else:
    print(f"Detected: DECIMAL format (mean = {sample_mean:.3f})")
    turnover_multiplier = 100

# Outlier removal
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

# Group by district and average features
grouped = df.groupby('District Code')[[
    'Average Years of Experience',
    'Transfer Rate Within District',
    'Transfer Rate Between Districts',
    'Average Total Salary',
    'Turnover Rate'
]].mean().reset_index()

# Features and target
X = grouped[[
    'Average Years of Experience',
    'Transfer Rate Within District',
    'Transfer Rate Between Districts',
    'Average Total Salary'
]]
y = grouped['Turnover Rate']

# Train model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Get actual local baseline
local_baseline_pct = y.mean() * 100
print(f"\nModel R² Score: {r2:.4f}")
print(f"LOCAL district baseline: {local_baseline_pct:.1f}%")

# Use statewide baseline (21.66%)
STATEWIDE_BASELINE_PCT = 21.66  # Your statewide average
USE_STATEWIDE_BASELINE = True   # Set to False to use local data

if USE_STATEWIDE_BASELINE:
    baseline_turnover_pct = STATEWIDE_BASELINE_PCT
    print(f"USING STATEWIDE baseline: {baseline_turnover_pct:.1f}%")
    print(f"   (Local districts average {local_baseline_pct:.1f}%)")
else:
    baseline_turnover_pct = local_baseline_pct

# Generate curve
def generate_turnover_curve_scaled(model, X, target_baseline_pct, max_salary_increase=50000, steps=100):
    """
    Generate curve scaled to start at target_baseline_pct
    Uses relative changes from model, not absolute offsets
    """
    salary_increases = np.linspace(0, max_salary_increase, steps)
    turnover_rates_pct = []
    
    # Get model's prediction at current salaries (local baseline in decimal)
    model_baseline_decimal = model.predict(X).mean()
    model_baseline_pct = model_baseline_decimal * 100
    
    # Calculate scaling factor to map model's baseline to target baseline
    scaling_factor = target_baseline_pct / model_baseline_pct
    
    for salary_add in salary_increases:
        X_sim_temp = X.copy()
        X_sim_temp['Average Total Salary'] = X['Average Total Salary'] + salary_add
        pred_decimal = model.predict(X_sim_temp).mean()
        pred_pct = pred_decimal * 100
        
        # Scale the prediction to match target baseline
        # This preserves the model's relative change pattern
        scaled_pct = pred_pct * scaling_factor
        turnover_rates_pct.append(max(0, scaled_pct))
    
    return salary_increases, turnover_rates_pct

salary_increases, turnover_rates_pct = generate_turnover_curve_scaled(
    model, X,
    target_baseline_pct=baseline_turnover_pct,
    max_salary_increase=50000,
    steps=100
)

# Find salary for the target
def find_salary_for_reduction(target_pct, salary_increases, turnover_rates_pct):
    """Find salary needed to reach target turnover percentage"""
    if target_pct >= turnover_rates_pct[0]:
        return 0, "Target is higher than baseline"
    
    if target_pct <= turnover_rates_pct[-1]:
        return salary_increases[-1], f"Minimum achievable: ${salary_increases[-1]:,.0f}"
    
    idx = np.argmin(np.abs(np.array(turnover_rates_pct) - target_pct))
    return salary_increases[idx], f"${salary_increases[idx]:,.0f}"

# Create Visualization
sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(14, 9))

# Plot curve
ax.plot(salary_increases, turnover_rates_pct, 
        linewidth=3, color='#2E86AB', 
        label=f'Model Projection (R² = {r2:.3f})',
        zorder=2)

# Fill area
ax.fill_between(salary_increases, turnover_rates_pct, baseline_turnover_pct, 
                alpha=0.2, color='#2E86AB', label='Reduction Potential')

# Mark starting point
ax.scatter([0], [baseline_turnover_pct], 
          color='green', s=200, zorder=5, 
          marker='o', edgecolor='black', linewidth=2)
ax.annotate(f'Statewide Baseline\n{baseline_turnover_pct:.1f}% Turnover', 
            xy=(0, baseline_turnover_pct),
            xytext=(5000, baseline_turnover_pct + 1),
            fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Calculate and mark key reduction targets
targets = [16, 15, 14, 13, 12]
for target in targets:
    if target >= min(turnover_rates_pct) and target <= max(turnover_rates_pct):
        idx = np.argmin(np.abs(np.array(turnover_rates_pct) - target))
        salary_at_target = salary_increases[idx]
        
        if target == 15:  # Highlight 15% as an example
            ax.scatter([salary_at_target], [target], 
                      color='red', s=150, zorder=5, 
                      marker='*', edgecolor='black', linewidth=1)
            ax.annotate(f'Target: {target}%\n${salary_at_target:,.0f}', 
                        xy=(salary_at_target, target),
                        xytext=(salary_at_target + 5000, target + 0.5),
                        fontsize=10, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Styling
ax.set_xlim(0, max(salary_increases))
ax.set_ylim(max(10, min(turnover_rates_pct) - 1), baseline_turnover_pct + 2)

# Format axes
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.xaxis.set_major_locator(plt.MultipleLocator(10000))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.1f}%'))
ax.yaxis.set_major_locator(plt.MultipleLocator(1))

# Labels
ax.set_title("Teacher Turnover Reduction from Statewide Baseline (21.66%)", 
             fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel("Additional Salary Investment per Teacher", 
              fontsize=14, labelpad=10)
ax.set_ylabel("Projected Turnover Rate", 
              fontsize=14, labelpad=10)

ax.grid(True, alpha=0.3, linestyle='--', zorder=1)

# Baseline line
ax.axhline(y=baseline_turnover_pct, color='gray', linestyle='--', alpha=0.5, zorder=1)

ax.legend(loc='upper right', framealpha=0.95, fontsize=11)

plt.tight_layout()
plt.show()

# Planner
print("\n" + "="*70)
print("STATEWIDE SALARY PLANNER TOOL")
print("="*70)
print(f"Starting turnover rate (statewide baseline): {baseline_turnover_pct:.1f}%")
if USE_STATEWIDE_BASELINE:
    print(f"   (Local district average is {local_baseline_pct:.1f}%)")
print(f"Minimum achievable turnover: {min(turnover_rates_pct):.1f}%")
print("="*70)

while True:
    try:
        print(f"\n💡 Suggested targets: 16%, 15%, 14%, 13%, 12%")
        target = float(input(f"Enter target turnover rate (% must be < {baseline_turnover_pct:.1f}): "))
        
        if target < 0:
            print("Turnover rate cannot be negative.")
            continue
        if target > baseline_turnover_pct:
            print(f"⚠️ Target ({target}%) is higher than baseline ({baseline_turnover_pct:.1f}%).")
            continue
        if target < min(turnover_rates_pct):
            print(f"⚠️ Target ({target}%) is below minimum achievable ({min(turnover_rates_pct):.1f}%).")
            print(f"   Using minimum achievable instead.")
            target = min(turnover_rates_pct)
        
        salary_needed, message = find_salary_for_reduction(target, salary_increases, turnover_rates_pct)
        reduction_pct = baseline_turnover_pct - target
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Target Turnover Rate: {target:.1f}%")
        print(f"Reduction Needed: {reduction_pct:.1f} percentage points")
        print(f"Required Investment Per Teacher: {message}")
        
        if salary_needed > 0:
            # Calculate additional metrics
            percent_reduction = (reduction_pct / baseline_turnover_pct) * 100
            print(f"Percent reduction: {percent_reduction:.1f}%")
            
            # Cost per percentage point reduction
            cost_per_point = salary_needed / reduction_pct
            print(f"Cost per percentage point reduction: ${cost_per_point:,.0f}")
            
            # For a district with 100 teachers
            total_cost_100 = salary_needed * 100
            print(f"For 100 teachers, total investment: ${total_cost_100:,.0f}")
        
        show = input("\nDisplay this point on the graph? (y/n): ").lower()
        if show == 'y':
            fig2, ax2 = plt.subplots(figsize=(14, 9))
            ax2.plot(salary_increases, turnover_rates_pct, 
                    linewidth=3, color='#2E86AB', zorder=2)
            
            ax2.scatter([salary_needed], [target], 
                       color='red', s=300, zorder=5, 
                       marker='D', edgecolor='black', linewidth=2)
            ax2.annotate(f'Your Target: {target:.1f}%\n${salary_needed:,.0f} per teacher', 
                        xy=(salary_needed, target),
                        xytext=(salary_needed + 5000, target + 0.8),
                        fontsize=11, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
            
            ax2.set_xlim(0, max(salary_increases))
            ax2.set_ylim(max(10, min(turnover_rates_pct) - 1), baseline_turnover_pct + 2)
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax2.xaxis.set_major_locator(plt.MultipleLocator(10000))
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.1f}%'))
            ax2.yaxis.set_major_locator(plt.MultipleLocator(1))
            ax2.set_title(f"Salary Investment to Reach {target:.1f}% Turnover", 
                         fontsize=20, fontweight='bold')
            ax2.set_xlabel("Additional Salary Investment per Teacher", fontsize=14)
            ax2.set_ylabel("Projected Turnover Rate", fontsize=14)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.axhline(y=baseline_turnover_pct, color='gray', linestyle='--', alpha=0.5)
            
            if USE_STATEWIDE_BASELINE:
                ax2.text(max(salary_increases) * 0.02, baseline_turnover_pct - 1.2, 
                        f'Based on statewide baseline: {baseline_turnover_pct:.1f}%\n'
                        f'Local district avg: {local_baseline_pct:.1f}%', 
                        fontsize=8, ha='left', color='gray',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
        
        again = input("\nCalculate another target? (y/n): ").lower()
        if again != 'y':
            break
            
    except ValueError:
        print("Please enter a valid number.")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        break

print("\nAnalysis complete!")
print("\nInterpretation:")
print(f"   - Starting from {baseline_turnover_pct:.1f}% (statewide average)")
print(f"   - Each ${salary_increases[10]:,.0f} invested reduces turnover by ~{baseline_turnover_pct - turnover_rates_pct[10]:.1f}%")
print(f"   - Returns diminish as turnover approaches {min(turnover_rates_pct):.1f}%")