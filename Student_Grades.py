"""Simple, linear analysis of student scores.

Improvements made:
- clearer comments and sectioning
- axis labels and titles on plots
- safer prediction call using a DataFrame
- prints model parameters and R-squared
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path


# Create figures directory
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Improve global aesthetics
# use a clean style without gridlines and a friendly notebook context
sns.set_style('white')
sns.set_context('notebook', font_scale=1.1)
sns.set_palette('deep')


# --- Configuration: change filename if needed ---
DATA_FILE = 'codecademy.csv'


# --- Load data ---
try:
	codecademy = pd.read_csv(DATA_FILE)
except FileNotFoundError:
	raise FileNotFoundError(f"Could not find data file: {DATA_FILE}")

print("Data preview:")
print(codecademy.head())
print(codecademy.dtypes)


# --- Scatter: score vs completed (raw) ---
plt.figure(figsize=(8, 5), dpi=150)
sns.scatterplot(x='completed', y='score', data=codecademy)
plt.xlabel('Completed prior lessons')
plt.ylabel('Score')
plt.title('Score vs Completed (raw scatter)')
plt.grid(False)
sns.despine()
scatter_path = FIG_DIR / 'scatter_raw.png'
plt.savefig(scatter_path, bbox_inches='tight')
plt.show()
plt.clf()


# --- Linear regression: score ~ completed ---
# using formula interface via OLS.from_formula keeps code short
model = sm.OLS.from_formula('score ~ completed', data=codecademy).fit()
print('\nModel: score ~ completed')
print(model.params)
print(f"R-squared: {model.rsquared:.3f}")


# --- Scatter with fitted line ---
plt.figure(figsize=(8, 5), dpi=150)
sns.scatterplot(x='completed', y='score', data=codecademy)
# sort x for a smooth line
xs = np.sort(codecademy['completed'].values)
preds = model.predict(pd.DataFrame({'completed': xs}))
plt.plot(xs, preds, color='red')
plt.xlabel('Completed prior lessons')
plt.ylabel('Score')
plt.title('Score vs Completed with OLS fit')
plt.grid(False)
sns.despine()
fit_path = FIG_DIR / 'scatter_fit.png'
plt.savefig(fit_path, bbox_inches='tight')
plt.show()
plt.clf()


# --- Example prediction ---
new_df = pd.DataFrame({'completed': [20]})
pred20 = model.predict(new_df).iloc[0]
print(f"Predicted score for completed=20: {pred20:.2f}")


# --- Residual diagnostics ---
fitted_values = model.predict(codecademy)
residuals = codecademy['score'] - fitted_values

plt.figure(figsize=(8, 5), dpi=150)
plt.hist(residuals, bins=20)
plt.xlabel('Residual')
plt.title('Residuals histogram')
plt.grid(False)
sns.despine()
hist_path = FIG_DIR / 'residuals_hist.png'
plt.savefig(hist_path, bbox_inches='tight')
plt.show()
plt.clf()

plt.figure(figsize=(8, 5), dpi=150)
plt.scatter(fitted_values, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.grid(False)
sns.despine()
resid_path = FIG_DIR / 'residuals_vs_fitted.png'
plt.savefig(resid_path, bbox_inches='tight')
plt.show()
plt.clf()


# --- Compare scores by lesson ---
plt.figure(figsize=(8, 5), dpi=150)
# add a more colorful palette and slightly thicker box outlines for clarity
sns.boxplot(x='lesson', y='score', data=codecademy, palette='Set2', saturation=0.9, width=0.6)
plt.title('Score distribution by Lesson')
plt.xlabel('Lesson')
plt.ylabel('Score')
plt.grid(False)
sns.despine()
boxplot_path = FIG_DIR / 'boxplot_lesson.png'
plt.savefig(boxplot_path, bbox_inches='tight')
plt.show()
plt.clf()

model2 = sm.OLS.from_formula('score ~ lesson', data=codecademy).fit()
print('\nModel: score ~ lesson')
print(model2.params)

print('\nGroup means:')
print(codecademy.groupby('lesson')['score'].mean())


# --- Scatter + separate regression lines by lesson ---
g = sns.lmplot(x='completed', y='score', hue='lesson', data=codecademy, height=5, aspect=1.4)
g.fig.suptitle('Score vs Completed by Lesson')
plt.grid(False)
sns.despine()
lmplot_path = FIG_DIR / 'lmplot_by_lesson.png'
g.savefig(lmplot_path, bbox_inches='tight')
plt.show()