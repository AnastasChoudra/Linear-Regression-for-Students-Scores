# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Read in the data
codecademy = pd.read_csv('codecademy.csv')
# Print the first five rows
print(codecademy.head())
# Create a scatter plot of score vs completed
plt.scatter(y=codecademy.score, x=codecademy.completed)
# Show then clear plot
plt.show()
plt.clf()
# Fit a linear regression to predict score based on prior lessons completed
model = sm.OLS.from_formula('score ~ completed', data=codecademy).fit()
print(model.params)
# Intercept interpretation:
# for reach completed course the score was 1.3 higher
# Slope interpretation:
# see above
# Plot the scatter plot with the line on top
plt.scatter(y=codecademy.score, x=codecademy.completed)
plt.plot(codecademy.completed, model.predict(codecademy))
plt.show()
plt.clf()
# Predict score for learner who has completed 20 prior lessons
new_data = {'completed':[20]}
print(model.predict(new_data))
# Calculate fitted values
fitted_values = model.predict(codecademy)
# Calculate residuals
residuals = codecademy.score - fitted_values
# Check normality assumption
plt.hist(residuals)
plt.show()
plt.clf() # I don't think it meets the assumption, hard to say
# Check homoscedasticity assumption
plt.scatter(y=residuals, x=fitted_values)
plt.show()
plt.clf() # assumption met
# Create a boxplot of score vs lesson
sns.boxplot(y=codecademy.score, x=codecademy.lesson)
plt.show()
plt.clf() # lesson a looks better
# Fit a linear regression to predict score based on which lesson they took
model2 = sm.OLS.from_formula('score ~ lesson', data=codecademy).fit()
print(model2.params)
# Calculate and print the group means and mean difference (for comparison)
'''
mean_score_lessonA = np.mean(codecademy.score[codecademy.lesson == 'Lesson A'])
mean_score_lessonB = np.mean(codecademy.score[codecademy.lesson == 'Lesson B'])
print('Mean score (A): ', mean_score_lessonA)
print('Mean score (B): ', mean_score_lessonB)
print('Mean score difference: ', mean_score_lessonA - mean_score_lessonB)
'''
#OR alternatively
print(codecademy.groupby('lesson').mean().score)
# Use `sns.lmplot()` to plot `score` vs. `completed` colored by `lesson`
sns.lmplot(x = 'completed', y = 'score', hue = 'lesson', data = codecademy)
plt.show()