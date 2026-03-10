# =============================================================================
# MEDICAL INSURANCE COST ANALYSIS
# Story: "Who Pays the Price? Uncovering What Drives Insurance Costs"
# =============================================================================

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Global style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.family'] = 'DejaVu Sans'

# ── 2. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\yoges\OneDrive\Documents\Medical_Insurance_Model\insurance.csv")

print("=" * 55)
print("  MEDICAL INSURANCE COST ANALYSIS")
print("  'Who Pays the Price?'")
print("=" * 55)
print(f"\n Dataset Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
print(f" Missing Values  : {df.isnull().sum().sum()}")
print(f" Duplicates      : {df.duplicated().sum()}")
print(f"\n── Sample Data ──")
print(df.head())
print(f"\n── Data Types ──")
print(df.dtypes)
print(f"\n── Statistical Summary ──")
print(df.describe())

# ── 3. DATA CLEANING ──────────────────────────────────────────────────────────
# Drop duplicates if any
df = df.drop_duplicates().reset_index(drop=True)

# ── 4. ENCODING & FEATURE SCALING ─────────────────────────────────────────────
df_model = df.copy()

# Label Encoding
le = LabelEncoder()
df_model['sex']    = le.fit_transform(df_model['sex'])        # female=0, male=1
df_model['smoker'] = le.fit_transform(df_model['smoker'])     # no=0, yes=1

# One-Hot Encoding for region (4 categories)
df_model = pd.get_dummies(df_model, columns=['region'], drop_first=True)

print("\n── Encoded Feature Columns ──")
print(df_model.columns.tolist())

# Feature Scaling (StandardScaler on numeric features)
features   = [c for c in df_model.columns if c != 'charges']
target     = 'charges'

X = df_model[features]
y = df_model[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

print("\n── After Scaling (first 3 rows) ──")
print(X_scaled.head(3))


# =============================================================================
# ── 5. EDA — THE STORY ────────────────────────────────────────────────────────
# =============================================================================

# ── CHAPTER 1: How Are Charges Distributed? ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chapter 1 — The Cost Landscape: How Are Charges Distributed?",
             fontsize=14, fontweight='bold', y=1.01)

# Histogram
axes[0].hist(df['charges'], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(df['charges'].mean(),   color='red',    linestyle='--', linewidth=1.5, label=f"Mean: ${df['charges'].mean():,.0f}")
axes[0].axvline(df['charges'].median(), color='orange', linestyle='--', linewidth=1.5, label=f"Median: ${df['charges'].median():,.0f}")
axes[0].set_title("Distribution of Insurance Charges")
axes[0].set_xlabel("Charges (USD)")
axes[0].set_ylabel("Count")
axes[0].legend()

# Log-scale to reveal skew
axes[1].hist(np.log1p(df['charges']), bins=40, color='teal', edgecolor='white', alpha=0.85)
axes[1].set_title("Log-Transformed Charges (reveals hidden clusters)")
axes[1].set_xlabel("Log(Charges)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("chapter1_distribution.png", bbox_inches='tight')
plt.show()
print("\nInsight 1: Charges are right-skewed with two distinct clusters —")
print("   a low-cost majority and a high-cost minority. This suggests")
print("   a specific subgroup is driving up average costs significantly.")


# ── CHAPTER 2: The Smoking Gun ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chapter 2 — The Smoking Gun: Smokers vs Non-Smokers",
             fontsize=14, fontweight='bold')

# Boxplot
sns.boxplot(data=df, x='smoker', y='charges', palette={'yes': '#e74c3c', 'no': '#2ecc71'},
            ax=axes[0])
axes[0].set_title("Charges by Smoking Status")
axes[0].set_xlabel("Smoker")
axes[0].set_ylabel("Charges (USD)")

# Mean comparison bar
smoker_avg = df.groupby('smoker')['charges'].mean().reset_index()
colors = ['#2ecc71' if s == 'no' else '#e74c3c' for s in smoker_avg['smoker']]
bars = axes[1].bar(smoker_avg['smoker'], smoker_avg['charges'], color=colors, edgecolor='white', width=0.4)
for bar, val in zip(bars, smoker_avg['charges']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                 f"${val:,.0f}", ha='center', fontweight='bold')
axes[1].set_title("Average Charges: Smoker vs Non-Smoker")
axes[1].set_xlabel("Smoker")
axes[1].set_ylabel("Average Charges (USD)")

plt.tight_layout()
plt.savefig("chapter2_smoking.png", bbox_inches='tight')
plt.show()

smoker_mean    = df[df['smoker'] == 'yes']['charges'].mean()
nonsmoker_mean = df[df['smoker'] == 'no']['charges'].mean()
print(f"\nInsight 2: Smokers cost ${smoker_mean:,.0f} on average vs ${nonsmoker_mean:,.0f} for non-smokers.")
print(f"   That's {smoker_mean/nonsmoker_mean:.1f}x MORE — smoking is the single biggest cost driver.")
print("   Industry implication: Insurers heavily penalize smokers — and the data justifies it.")


# ── CHAPTER 3: The Age Factor ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chapter 3 — Age & BMI: The Gradual Climb",
             fontsize=14, fontweight='bold')

# Age vs Charges colored by smoker
colors_map = df['smoker'].map({'yes': '#e74c3c', 'no': '#3498db'})
axes[0].scatter(df['age'], df['charges'], c=colors_map, alpha=0.5, edgecolors='none', s=25)
axes[0].set_title("Age vs Charges (Red = Smoker, Blue = Non-Smoker)")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Charges (USD)")

# BMI vs Charges
axes[1].scatter(df['bmi'], df['charges'], c=colors_map, alpha=0.5, edgecolors='none', s=25)
axes[1].axvline(30, color='black', linestyle='--', linewidth=1.2, label='BMI=30 (Obese threshold)')
axes[1].set_title("BMI vs Charges (Red = Smoker, Blue = Non-Smoker)")
axes[1].set_xlabel("BMI")
axes[1].set_ylabel("Charges (USD)")
axes[1].legend()

plt.tight_layout()
plt.savefig("chapter3_age_bmi.png", bbox_inches='tight')
plt.show()
print("\nInsight 3: Three distinct charge bands emerge by age — driven entirely by smoking status.")
print("   Non-smokers show a gentle rise with age. Smokers show a steep, high-cost band.")
print("   BMI above 30 (obese) combined with smoking creates the most expensive profiles.")


# ── CHAPTER 4: Does Gender or Region Matter? ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chapter 4 — Gender & Region: Smaller Than You Think",
             fontsize=14, fontweight='bold')

# Gender
sns.boxplot(data=df, x='sex', y='charges', palette='pastel', ax=axes[0])
axes[0].set_title("Charges by Gender")
axes[0].set_xlabel("Sex")
axes[0].set_ylabel("Charges (USD)")

# Region
region_avg = df.groupby('region')['charges'].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=region_avg, x='region', y='charges', palette='Blues_d', ax=axes[1])
axes[1].set_title("Average Charges by Region")
axes[1].set_xlabel("Region")
axes[1].set_ylabel("Average Charges (USD)")
for i, row in region_avg.iterrows():
    axes[1].text(i, row['charges'] + 100, f"${row['charges']:,.0f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("chapter4_gender_region.png", bbox_inches='tight')
plt.show()
print("\nInsight 4: Gender has minimal impact on charges — a surprisingly fair outcome.")
print("   Region shows slight variation but nothing dramatic.")
print("   This tells insurers: geography and gender are weak pricing levers.")


# ── CHAPTER 5: Correlation Heatmap ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("Chapter 5 — What Correlates Most With High Charges?",
             fontsize=14, fontweight='bold')

corr = df_model.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("chapter5_heatmap.png", bbox_inches='tight')
plt.show()
print("\nInsight 5: Smoker status has the highest correlation with charges (0.79).")
print("   Age (0.30) and BMI (0.20) follow. Children, sex, region are weak predictors.")


# =============================================================================
# ── 6. LINEAR REGRESSION MODEL ───────────────────────────────────────────────
# =============================================================================
print("\n" + "=" * 55)
print("  MODEL: LINEAR REGRESSION")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n  R² Score  : {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"  MAE       : ${mae:,.2f}")
print(f"  RMSE      : ${rmse:,.2f}")

# ── Actual vs Predicted ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Performance — How Well Does the Model Predict?",
             fontsize=14, fontweight='bold')

# Scatter
axes[0].scatter(y_test, y_pred, alpha=0.5, color='steelblue', edgecolors='none', s=25)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', linewidth=1.5, label='Perfect Prediction')
axes[0].set_title(f"Actual vs Predicted (R²={r2:.3f})")
axes[0].set_xlabel("Actual Charges")
axes[0].set_ylabel("Predicted Charges")
axes[0].legend()

# Residuals
residuals = y_test - y_pred
axes[1].hist(residuals, bins=40, color='coral', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='black', linestyle='--', linewidth=1.5)
axes[1].set_title("Residual Distribution")
axes[1].set_xlabel("Residual (Actual - Predicted)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("model_performance.png", bbox_inches='tight')
plt.show()

# ── Feature Importance (Coefficients) ────────────────────────────────────────
coef_df = pd.DataFrame({
    'Feature'    : features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#e74c3c' if c > 0 else '#3498db' for c in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title("Feature Coefficients — What Drives Charges Up or Down?",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Coefficient Value (after scaling)")
plt.tight_layout()
plt.savefig("feature_importance.png", bbox_inches='tight')
plt.show()

print("\n── Top Drivers of Insurance Cost ──")
print(coef_df.to_string(index=False))


# =============================================================================
# ── 7. FINAL CONCLUSIONS ─────────────────────────────────────────────────────
# =============================================================================
print("\n" + "=" * 55)
print("  FINAL INDUSTRY INSIGHTS")
print("=" * 55)
print(f"""
1. SMOKING IS THE #1 COST DRIVER
   Smokers pay {smoker_mean/nonsmoker_mean:.1f}x more than non-smokers on average.
   Avg smoker charge: ${smoker_mean:,.0f} vs ${nonsmoker_mean:,.0f} for non-smokers.
   → Insurers should invest heavily in smoking cessation programs
     to reduce the highest-cost claims.

2. AGE COMPOUNDS COSTS — BUT ONLY FOR SMOKERS
   Non-smokers show a gentle, manageable cost increase with age.
   Smokers show a steep, accelerating cost curve.
   → Age-based pricing alone is insufficient — smoking × age
     interaction is the real risk multiplier.

3. OBESITY (BMI > 30) + SMOKING = HIGHEST RISK PROFILE
   The most expensive patients are smokers with high BMI.
   → Wellness programs targeting this combined profile
     could yield the biggest cost savings for insurers.

4. GENDER & REGION ARE WEAK PRICING FACTORS
   Minimal difference in charges by sex or region.
   → These should not be primary underwriting criteria.
     Risk-based pricing should focus on smoking + BMI + age.

5. MODEL EXPLAINS {r2*100:.1f}% OF COST VARIANCE
   Linear regression captures the broad patterns well.
   Residuals suggest non-linear interactions exist
   (e.g. smoking × BMI synergy).
   → A more advanced model (Random Forest, XGBoost) could
     push accuracy further for production use.
""")