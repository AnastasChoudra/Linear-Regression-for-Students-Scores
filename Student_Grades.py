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
sns.set_style('whitegrid')
plt.figure()
sns.scatterplot(x='completed', y='score', data=codecademy)
plt.xlabel('Completed prior lessons')
plt.ylabel('Score')
plt.title('Score vs Completed (raw scatter)')
plt.show()
plt.clf()


# --- Linear regression: score ~ completed ---
# using formula interface via OLS.from_formula keeps code short
model = sm.OLS.from_formula('score ~ completed', data=codecademy).fit()
print('\nModel: score ~ completed')
print(model.params)
print(f"R-squared: {model.rsquared:.3f}")


# --- Scatter with fitted line ---
plt.figure()
sns.scatterplot(x='completed', y='score', data=codecademy)
# sort x for a smooth line
xs = np.sort(codecademy['completed'].values)
preds = model.predict(pd.DataFrame({'completed': xs}))
plt.plot(xs, preds, color='red')
plt.xlabel('Completed prior lessons')
plt.ylabel('Score')
plt.title('Score vs Completed with OLS fit')
plt.show()
plt.clf()


# --- Example prediction ---
new_df = pd.DataFrame({'completed': [20]})
pred20 = model.predict(new_df).iloc[0]
print(f"Predicted score for completed=20: {pred20:.2f}")


# --- Residual diagnostics ---
fitted_values = model.predict(codecademy)
residuals = codecademy['score'] - fitted_values

plt.figure()
plt.hist(residuals, bins=20)
plt.xlabel('Residual')
plt.title('Residuals histogram')
plt.show()
plt.clf()

plt.figure()
plt.scatter(fitted_values, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()
plt.clf()


# --- Compare scores by lesson ---
plt.figure()
sns.boxplot(x='lesson', y='score', data=codecademy)
plt.title('Score distribution by Lesson')
plt.xlabel('Lesson')
plt.ylabel('Score')
plt.show()
plt.clf()

model2 = sm.OLS.from_formula('score ~ lesson', data=codecademy).fit()
print('\nModel: score ~ lesson')
print(model2.params)

print('\nGroup means:')
print(codecademy.groupby('lesson')['score'].mean())


# --- Scatter + separate regression lines by lesson ---
sns.lmplot(x='completed', y='score', hue='lesson', data=codecademy)
plt.title('Score vs Completed by Lesson')
plt.show()