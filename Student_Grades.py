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
from statsmodels.stats.stattools import jarque_bera
from pathlib import Path
import os
from statsmodels.nonparametric.smoothers_lowess import lowess


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


# --- Pre-fit assumption checks (visual + numeric) ---
# 1) Scatter with LOWESS to inspect linearity
plt.figure(figsize=(8, 5), dpi=150)
sns.scatterplot(x='completed', y='score', data=codecademy, s=50)
lw = lowess(codecademy['score'], codecademy['completed'], frac=0.3)
plt.plot(lw[:, 0], lw[:, 1], color='red', lw=2, label='LOWESS')
plt.xlabel('Completed prior lessons')
plt.ylabel('Score')
plt.title('Pre-fit: Score vs Completed (with LOWESS)')
plt.legend()
plt.grid(False)
sns.despine()
pre_scatter_path = FIG_DIR / 'prefit_scatter_lowess.png'
plt.savefig(pre_scatter_path, bbox_inches='tight')
plt.show()
plt.clf()

# 2) Distribution checks: score and completed
plt.figure(figsize=(8, 4), dpi=150)
ax1 = plt.subplot(1, 2, 1)
sns.histplot(codecademy['score'], bins=20, kde=True, color='#74a9cf', ax=ax1)
ax1.set_title('Distribution of score')
ax1.set_xlabel('Score')

ax2 = plt.subplot(1, 2, 2)
sns.histplot(codecademy['completed'], bins=20, kde=True, color='#a1d99b', ax=ax2)
ax2.set_title('Distribution of completed')
ax2.set_xlabel('Completed prior lessons')

dist_path = FIG_DIR / 'prefit_distributions.png'
plt.tight_layout()
plt.savefig(dist_path, bbox_inches='tight')
plt.show()
plt.clf()

# 3) Numeric checks: Pearson correlation and simple outlier flags
corr = codecademy['score'].corr(codecademy['completed'])
print(f"Pre-fit Pearson correlation (score, completed): {corr:.3f}")

# Flag extreme values (z-score > 3) for manual inspection
z_score = (codecademy['score'] - codecademy['score'].mean()) / codecademy['score'].std(ddof=0)
z_completed = (codecademy['completed'] - codecademy['completed'].mean()) / codecademy['completed'].std(ddof=0)
outliers = codecademy[(z_score.abs() > 3) | (z_completed.abs() > 3)]
if not outliers.empty:
    print("Pre-fit outliers (|z|>3) found:\n", outliers)
else:
    print("No extreme outliers found by z-score > 3.")



# --- Fit model and create annotated figures (kept minimal) ---
# Fit OLS once and use results for both annotated scatter and residual histogram
model = sm.OLS.from_formula('score ~ completed', data=codecademy).fit()
print('\nModel: score ~ completed')
print(model.params)
print(f"R-squared: {model.rsquared:.3f}")

# Predictions and residuals
codecademy['pred'] = model.predict(codecademy)
residuals = codecademy['score'] - codecademy['pred']


# --- Annotated scatter: score vs completed with fit and metrics ---
plt.figure(figsize=(8, 5), dpi=150)
sns.scatterplot(x='completed', y='score', data=codecademy, s=50)
# smooth fit line
xs = np.linspace(codecademy['completed'].min(), codecademy['completed'].max(), 200)
plt.plot(xs, model.predict(pd.DataFrame({'completed': xs})), color='C1', lw=2)
plt.xlabel('Completed prior lessons')
plt.ylabel('Score')
plt.title('Score vs Completed with OLS fit (annotated)')

# metrics
r2 = float(model.rsquared)
intercept = float(model.params.get('Intercept', np.nan))
slope = float(model.params.get('completed', np.nan))
slope_p = float(model.pvalues.get('completed', np.nan))
rmse = float(np.sqrt(np.mean((codecademy['score'] - codecademy['pred']) ** 2)))
mae = float(np.mean(np.abs(codecademy['score'] - codecademy['pred'])))
annot = (
    f"R² = {r2:.3f}\n"
    f"slope = {slope:.3f}\n"
    f"intercept = {intercept:.2f}\n"
    f"RMSE = {rmse:.2f}\n"
    f"MAE = {mae:.2f}\n"
    f"p (slope) = {slope_p:.2g}"
)
ax = plt.gca()
ax.text(0.02, 0.98, annot, transform=ax.transAxes, va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.85))
plt.grid(False)
sns.despine()
scatter_metrics_path = FIG_DIR / 'scatter_fit_metrics.png'
plt.savefig(scatter_metrics_path, bbox_inches='tight')
plt.show()
plt.clf()


# --- Compare scores by lesson ---
plt.figure(figsize=(8, 5), dpi=150)
# colorful boxplot
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


# --- Scatter + separate regression lines by lesson (lmplot) ---
g = sns.lmplot(x='completed', y='score', hue='lesson', data=codecademy, height=5, aspect=1.4)
g.fig.suptitle('Score vs Completed by Lesson')
plt.grid(False)
sns.despine()
lmplot_path = FIG_DIR / 'lmplot_by_lesson.png'
g.savefig(lmplot_path, bbox_inches='tight')
plt.show()

# --- Extended analysis: regression metrics and residuals (using available columns) ---

df = codecademy.copy()

# Fit model using the available predictor `completed` and outcome `score`
model_ext = sm.OLS.from_formula('score ~ completed', data=df).fit()
df['pred'] = model_ext.predict(df)
residuals = df['score'] - df['pred']

# compute summary metrics
r2_ext = float(model_ext.rsquared)
intercept_ext = float(model_ext.params.get('Intercept', np.nan))
slope_ext = float(model_ext.params.get('completed', np.nan))
slope_p_ext = float(model_ext.pvalues.get('completed', np.nan))
rmse_ext = float(np.sqrt(np.mean((df['score'] - df['pred']) ** 2)))
mae_ext = float(np.mean(np.abs(df['score'] - df['pred'])))

# plot scatter + fit with annotations
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.scatter(df['completed'], df['score'], alpha=0.75)
# plot sorted fit line for clarity
xs_sorted = np.sort(df['completed'].values)
ax.plot(xs_sorted, model_ext.predict(pd.DataFrame({'completed': xs_sorted})), color='C1', lw=2)

annot_text = (
    f"R² = {r2_ext:.3f}\n"
    f"Slope = {slope_ext:.3f}\n"
    f"Intercept = {intercept_ext:.2f}\n"
    f"RMSE = {rmse_ext:.2f}\n"
    f"MAE = {mae_ext:.2f}\n"
    f"p (slope) = {slope_p_ext:.2g}"
)
ax.text(0.02, 0.98, annot_text, transform=ax.transAxes, fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.85))

sns.despine(ax=ax)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'scatter_fit_metrics.png'))
plt.close()


# residuals histogram with normal overlay and stats
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
counts, bins, patches = ax.hist(residuals, bins=30, density=True, color='#74a9cf', edgecolor='white', alpha=0.9)

# overlay normal pdf using residual mean and std
mu = float(residuals.mean())
sigma = float(residuals.std(ddof=0))
x_vals = np.linspace(residuals.min(), residuals.max(), 200)
pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)
ax.plot(x_vals, pdf, color='k', lw=1.25, linestyle='--', label='Normal PDF')

# summary stats for residuals
skew = float(residuals.skew())
kurt = float(residuals.kurt())
# jarque_bera may return either (jb, pvalue) or a longer tuple depending on statsmodels version
jb_res = jarque_bera(residuals)
if isinstance(jb_res, (list, tuple)) and len(jb_res) >= 2:
    jb_stat = float(jb_res[0])
    jb_p = float(jb_res[1])
else:
    jb_stat = float(jb_res)
    jb_p = np.nan

# vertical lines for mean/median
ax.axvline(mu, color='C0', lw=1.25, linestyle='-', label=f"mean={mu:.2f}")
median_val = float(residuals.median())
ax.axvline(median_val, color='C1', lw=1.0, linestyle=':', label=f"median={median_val:.2f}")

# annotation box
hist_annot = (
    f"N = {len(residuals)}\n"
    f"mean = {mu:.2f}\n"
    f"median = {median_val:.2f}\n"
    f"skew = {skew:.2f}\n"
    f"kurt = {kurt:.2f}\n"
    f"Jarque–Bera p = {jb_p:.2g}"
)
ax.text(0.98, 0.98, hist_annot, transform=ax.transAxes, fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.85))

ax.set_xlabel('Residual')
ax.set_title('Residuals histogram')
ax.legend(frameon=False, fontsize=8)
sns.despine(ax=ax)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'residuals_hist_metrics.png'))
plt.show()
plt.close()

# --- Homoscedasticity check: residuals vs fitted values ---
fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=150)
fitted_vals = df['pred']
ax2.scatter(fitted_vals, residuals, alpha=0.75)
ax2.axhline(0, color='red', linestyle='--', linewidth=1)
ax2.set_xlabel('Fitted values')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals vs Fitted (Homoscedasticity check)')
plt.grid(False)
sns.despine(ax=ax2)
resid_vs_fitted_path = FIG_DIR / 'residuals_vs_fitted_metrics.png'
plt.tight_layout()
plt.savefig(resid_vs_fitted_path, bbox_inches='tight')
plt.show()
plt.close()