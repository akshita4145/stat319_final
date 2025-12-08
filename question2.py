"""
Question 2: One-Sample Hypothesis Test
Wine Quality Dataset - Testing mean pH of red wines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import t

#load the data
#note: file uses semicolon as delimiter
df = pd.read_csv('winequality-red.csv', sep=';')

print("\nQUESTION 2: ONE-SAMPLE HYPOTHESIS TEST\n")

#variable for analysis
variable = 'pH'
data = df[variable].dropna()

print(f"\nVariable selected: {variable}")
print(f"Sample size (n): {len(data)}")
print(f"Sample mean: {data.mean():.4f}")
print(f"Sample standard deviation: {data.std(ddof=1):.4f}")

#1. STATE HYPOTHESES
print("\n1. HYPOTHESIS STATEMENT")

#hypothesized population mean
mu_0 = 3.5

print(f"\nResearch Question:")
print(f"Is the mean pH of red wines different from {mu_0}?")
print(f"\nNull Hypothesis (H₀): μ = {mu_0}")
print(f"Alternative Hypothesis (H₁): μ ≠ {mu_0}")
print(f"\nThis is a two-tailed test at α = 0.05 significance level.")

#2.CHECK ASSUMPTIONS
print("\n\n2. CHECKING ASSUMPTIONS")

#a. independence
print("\nAssumption 1 - Independence:")
print("The wine samples are assumed to be independently collected.")
print("Each observation represents a different wine sample.")

#b. large sample size (n >= 30)
print(f"\nAssumption 2 - Sample Size:")
print(f"Sample size n = {len(data)}")
if len(data) >= 30:
    print("✓ Sample size is large (n ≥ 30), so CLT applies.")
else:
    print("✗ Sample size is small (n < 30), normality should be checked.")

#c. normality check
print("\nAssumption 3 - Normality:")
print("Visual checks using histogram and QQ plot...")

#create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#histogram
axes[0].hist(data, bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {data.mean():.3f}')
axes[0].axvline(mu_0, color='green', linestyle='--', linewidth=2, label=f'H₀: μ = {mu_0}')
axes[0].set_xlabel(variable)
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Histogram of {variable}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

#QQ plot
stats.probplot(data, dist="norm", plot=axes[1])
axes[1].set_title(f'QQ Plot of {variable}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('question2_assumptions.png', dpi=300, bbox_inches='tight')
print("Plots saved as 'question2_assumptions.png'")
plt.close()

#visuakl assessment of normality
print("\nVisual Assessment:")
print("  From the histogram: The distribution appears roughly bell-shaped and")
print("  approximately symmetric, though with some slight left skewness.")
print("  From the QQ plot: Most points fall close to the reference line,")
print("  suggesting the data is approximately normally distributed.")
print("\n  Conclusion: The normality assumption is reasonably satisfied.")
print("  Note: With a large sample size (n = 1599), the Central Limit Theorem")
print("  ensures that the t-test is valid even if data isn't perfectly normal.")

#3. CONDUCT ONE-SAMPLE T-TEST
print("\n\n3. HYPOTHESIS TEST")

#test statistic manually
n = len(data)
sample_mean = data.mean()
sample_std = data.std(ddof=1)
se = sample_std / np.sqrt(n)

# Test statistic
t_stat = (sample_mean - mu_0) / se

#degrees of freedom
df = n - 1

#p-value (two-tailed)
p_value = 2 * (1 - t.cdf(abs(t_stat), df))

#critical value at a = 0.05
alpha = 0.05
t_critical = t.ppf(1 - alpha/2, df)

print(f"\nTest: One-sample t-test")
print(f"Test statistic (t): {t_stat:.4f}")
print(f"Degrees of freedom: {df}")
print(f"p-value: {p_value:.4f}")
print(f"Critical value (α = {alpha}): ±{t_critical:.4f}")
print(f"Rejection region: |t| > {t_critical:.4f}")

#4. CONFIDENCE INTERVAL
print("\n\n4. CONFIDENCE INTERVAL")

#95% confidence interval
confidence_level = 0.95
margin_of_error = t_critical * se
ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print(f"\n{confidence_level*100:.0f}% Confidence Interval for μ:")
print(f"({ci_lower:.4f}, {ci_upper:.4f})")
print(f"\nInterpretation:")
print(f"We are {confidence_level*100:.0f}% confident that the true mean {variable} of red wines")
print(f"lies between {ci_lower:.4f} and {ci_upper:.4f}.")

#check if mu_0 is in the CI
if ci_lower <= mu_0 <= ci_upper:
    print(f"\nNote: The hypothesized value μ₀ = {mu_0} IS within the confidence interval.")
else:
    print(f"\nNote: The hypothesized value μ₀ = {mu_0} is NOT within the confidence interval.")

#5. CONCLUSION
print("\n\n5. CONCLUSION")

print(f"\nDecision rule: Reject H₀ if p-value < α = {alpha}")
print(f"p-value = {p_value:.4f}")

if p_value < alpha:
    print(f"\nDecision: REJECT H₀ (p-value = {p_value:.4f} < {alpha})")
    print(f"\nConclusion in context:")
    print(f"At the {alpha} significance level, there is sufficient evidence to conclude")
    print(f"that the mean {variable} of red wines is significantly different from {mu_0}.")
    print(f"The sample data suggests the true mean {variable} is approximately {sample_mean:.4f},")
    print(f"which is {'higher' if sample_mean > mu_0 else 'lower'} than the hypothesized value of {mu_0}.")
else:
    print(f"\nDecision: FAIL TO REJECT H₀ (p-value = {p_value:.4f} ≥ {alpha})")
    print(f"\nConclusion in context:")
    print(f"At the {alpha} significance level, there is insufficient evidence to conclude")
    print(f"that the mean {variable} of red wines is different from {mu_0}.")
    print(f"The data is consistent with the hypothesis that the mean {variable} is {mu_0}.")

#VISUALIZATION OF T- TEST
print("\n\nCreating visualization of hypothesis test...")

#t-distribution plot
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(-5, 5, 1000)
y = t.pdf(x, df)

ax.plot(x, y, 'b-', linewidth=2, label=f't-distribution (df={df})')
ax.fill_between(x[x <= -t_critical], 0, t.pdf(x[x <= -t_critical], df),
                alpha=0.3, color='red', label=f'Rejection region (α/2 = {alpha/2})')
ax.fill_between(x[x >= t_critical], 0, t.pdf(x[x >= t_critical], df),
                alpha=0.3, color='red')
ax.axvline(t_stat, color='green', linestyle='--', linewidth=2,
           label=f'Test statistic = {t_stat:.3f}')
ax.axvline(-t_critical, color='red', linestyle=':', linewidth=1.5)
ax.axvline(t_critical, color='red', linestyle=':', linewidth=1.5)
ax.set_xlabel('t-value')
ax.set_ylabel('Probability Density')
ax.set_title(f'One-Sample t-Test: H₀: μ = {mu_0} vs H₁: μ ≠ {mu_0}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('question2_test_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Test visualization saved as 'question2_test_visualization.png'")
plt.close()

print("\nAnalysis complete!")
