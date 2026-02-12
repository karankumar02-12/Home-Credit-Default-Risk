#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install imbalanced-learn')


# In[2]:


# =============================================================================
# Libraries
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Modeling
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report, 
                             confusion_matrix, precision_recall_curve)

# Imbalance handling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Display settings
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# Plot styling (version-safe)
try:
    # Try newer seaborn style
    plt.style.use('seaborn-v0_8')
except:
    try:
        # Try older seaborn style
        plt.style.use('seaborn-darkgrid')
    except:
        # Fall back to default
        sns.set_style("darkgrid")
        print("Using seaborn style configuration")

# Set default figure size
plt.rcParams['figure.figsize'] = (10, 6)

print("✓ All libraries imported successfully!")
print(f"✓ Pandas version: {pd.__version__}")
print(f"✓ NumPy version: {np.__version__}")
print(f"✓ Matplotlib version: {plt.matplotlib.__version__}")
print(f"✓ Seaborn version: {sns.__version__}")


# In[3]:


# =============================================================================
# DATA LOADING
# =============================================================================
# Download from: https://www.kaggle.com/c/home-credit-default-risk/data
# We'll work primarily with application_train.csv (main dataset)

# Load main application data
app_train = pd.read_csv('application_train.csv')
app_test = pd.read_csv('application_test.csv')

print(f"Training data shape: {app_train.shape}")
print(f"Test data shape: {app_test.shape}")
print(f"\nTarget distribution:")
print(app_train['TARGET'].value_counts(normalize=True))


# In[4]:


# =============================================================================
# STEP 1: Understand Data Types
# WHY: Different preprocessing for numerical vs categorical features
# =============================================================================
print("Data Types Summary:")
print(app_train.dtypes.value_counts())

# Identify feature types
numeric_features = app_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = app_train.select_dtypes(include=['object']).columns.tolist()

# Remove TARGET from numeric features
numeric_features.remove('TARGET')
numeric_features.remove('SK_ID_CURR')  # Customer ID, not a feature

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"\nFirst 5 categorical features: {categorical_features[:5]}")


# In[5]:


# =============================================================================
# STEP 2: Missing Value Analysis
# WHY: Banking context - missing values have business meaning
# - Missing employment = unemployed (higher risk)
# - Missing building info = temporary address (higher risk)
# =============================================================================
def analyze_missing_values(df):
    """Comprehensive missing value analysis"""
    missing = pd.DataFrame({
        'feature': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df)) * 100,
        'dtype': df.dtypes
    })
    missing = missing[missing['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    return missing

missing_summary = analyze_missing_values(app_train)
print("Top 10 features with missing values:")
print(missing_summary.head(10))

# Visualize missing patterns
plt.figure(figsize=(12, 6))
missing_summary.head(20).plot(x='feature', y='missing_pct', kind='barh', 
                              color='coral', figsize=(10, 8))
plt.xlabel('Missing Percentage (%)')
plt.title('Top 20 Features with Missing Values\n(Banking Context: May indicate data quality or applicant behavior)')
plt.tight_layout()
plt.show()


# In[6]:


# =============================================================================
# STEP 3: Analyze Target Distribution by Key Features
# WHY: Understand which customer segments have higher default rates
# =============================================================================

# Helper function for target analysis
def plot_target_analysis(feature, df, figsize=(14, 5)):
    """
    Creates two plots:
    1. Distribution of feature by target
    2. Default rate by feature category
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Distribution
    if df[feature].dtype == 'object':
        # Categorical
        df.groupby([feature, 'TARGET']).size().unstack().plot(kind='bar', ax=axes[0], 
                                                               color=['skyblue', 'salmon'])
        axes[0].set_title(f'Distribution of {feature} by Target')
        axes[0].set_ylabel('Count')
        
        # Plot 2: Default rate
        default_rate = df.groupby(feature)['TARGET'].mean() * 100
        default_rate.plot(kind='bar', ax=axes[1], color='crimson')
        axes[1].axhline(y=df['TARGET'].mean()*100, color='blue', linestyle='--', 
                       label=f'Overall Default Rate: {df["TARGET"].mean()*100:.2f}%')
        axes[1].set_title(f'Default Rate by {feature}')
        axes[1].set_ylabel('Default Rate (%)')
        axes[1].legend()
    else:
        # Numerical - use bins
        for target in [0, 1]:
            df[df['TARGET'] == target][feature].hist(bins=30, alpha=0.6, 
                                                      label=f'Target {target}', ax=axes[0])
        axes[0].set_title(f'Distribution of {feature} by Target')
        axes[0].legend(['Non-Default', 'Default'])
        
        # Binned default rate
        df['temp_bins'] = pd.qcut(df[feature], q=10, duplicates='drop')
        default_rate = df.groupby('temp_bins')['TARGET'].mean() * 100
        default_rate.plot(kind='line', ax=axes[1], color='crimson', marker='o')
        axes[1].set_title(f'Default Rate across {feature} Quantiles')
        axes[1].set_ylabel('Default Rate (%)')
        df.drop('temp_bins', axis=1, inplace=True)
    
    plt.tight_layout()
    plt.show()

# Analyze key features
print("=" * 80)
print("ANALYZING KEY RISK INDICATORS")
print("=" * 80)

# 1. Income
plot_target_analysis('AMT_INCOME_TOTAL', app_train)


# In[7]:


# 2. Credit Amount
plot_target_analysis('AMT_CREDIT', app_train)


# In[8]:


# 3. Contract Type
plot_target_analysis('NAME_CONTRACT_TYPE', app_train)


# In[9]:


# 4. Income Type (Employment)
plot_target_analysis('NAME_INCOME_TYPE', app_train)


# In[10]:


# 5. Education Level
plot_target_analysis('NAME_EDUCATION_TYPE', app_train)


# In[11]:


# =============================================================================
# STEP 4: Correlation with Target
# WHY: Identify strongest predictors for feature selection
# =============================================================================

# Calculate correlations
correlations = app_train[numeric_features + ['TARGET']].corr()['TARGET'].sort_values(ascending=False)

print("Top 15 Positive Correlations with Default:")
print(correlations.head(15))
print("\nTop 15 Negative Correlations with Default:")
print(correlations.tail(15))

# Visualize
plt.figure(figsize=(10, 12))
correlations.drop('TARGET').plot(kind='barh', color=['green' if x < 0 else 'red' for x in correlations.drop('TARGET')])
plt.title('Feature Correlations with Default Risk\n(Green = Protective, Red = Risky)')
plt.xlabel('Correlation Coefficient')
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.show()


# In[12]:


# =============================================================================
# STEP 5: Deep Dive into EXT_SOURCE Features
# WHY: These are normalized credit bureau scores (0-1 scale)
# Banks use these heavily in practice
# =============================================================================

# Check availability
ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
for col in ext_sources:
    missing_pct = app_train[col].isnull().sum() / len(app_train) * 100
    print(f"{col}: {missing_pct:.2f}% missing")

# Distribution by target
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(ext_sources):
    app_train.boxplot(column=col, by='TARGET', ax=axes[i])
    axes[i].set_title(f'{col} by Default Status')
    axes[i].set_xlabel('Target (0=Good, 1=Default)')
plt.suptitle('External Credit Scores Distribution\n(Lower scores = Higher default risk)', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Statistical test
from scipy.stats import ttest_ind
for col in ext_sources:
    good = app_train[app_train['TARGET'] == 0][col].dropna()
    bad = app_train[app_train['TARGET'] == 1][col].dropna()
    t_stat, p_value = ttest_ind(good, bad)
    print(f"{col}: Mean difference is statistically significant (p < 0.001): {p_value < 0.001}")


# In[13]:


# =============================================================================
# STEP 6: Detect and Handle Outliers
# WHY: Outliers can represent:
# - Data errors (365,243 years employed = clearly wrong)
# - Fraud (income 10,000x median = suspicious)
# - Genuine edge cases (billionaires applying for small loans)
# =============================================================================

def detect_outliers(df, features, method='IQR'):
    """
    Banking approach to outlier detection:
    - Use domain knowledge first
    - Then statistical methods
    """
    outlier_summary = []
    
    for col in features:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            # Statistics
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr  # Using 3*IQR (more conservative)
            upper_bound = q3 + 3 * iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_pct = (len(outliers) / len(df)) * 100
            
            outlier_summary.append({
                'feature': col,
                'outlier_count': len(outliers),
                'outlier_pct': outlier_pct,
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            })
    
    return pd.DataFrame(outlier_summary).sort_values('outlier_pct', ascending=False)

outlier_df = detect_outliers(app_train, numeric_features)
print("Features with Outliers (>5% of data):")
print(outlier_df[outlier_df['outlier_pct'] > 5].head(10))


# In[14]:


# =============================================================================
# STEP 7: Domain-Specific Outlier Handling
# =============================================================================

# DAYS_EMPLOYED: Has value 365243 for unemployed (encoding error)
print(f"Unique DAYS_EMPLOYED values with 365243: {(app_train['DAYS_EMPLOYED'] == 365243).sum()}")

# Create flag for unemployed and fix
app_train['FLAG_UNEMPLOYED'] = (app_train['DAYS_EMPLOYED'] == 365243).astype(int)
app_train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

# Apply same to test set
app_test['FLAG_UNEMPLOYED'] = (app_test['DAYS_EMPLOYED'] == 365243).astype(int)
app_test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

print("✓ Unemployment flag created and DAYS_EMPLOYED corrected")

# AMT_INCOME_TOTAL: Cap extreme values (banking practice: 99th percentile)
income_99 = app_train['AMT_INCOME_TOTAL'].quantile(0.99)
print(f"99th percentile income: {income_99:,.0f}")

app_train['AMT_INCOME_TOTAL'] = app_train['AMT_INCOME_TOTAL'].clip(upper=income_99)
app_test['AMT_INCOME_TOTAL'] = app_test['AMT_INCOME_TOTAL'].clip(upper=income_99)

print("✓ Income capped at 99th percentile")


# In[15]:


# =============================================================================
# STEP 8: Strategic Missing Value Imputation
# Banking Logic:
# - Demographics: Use mode (most common category)
# - Financial: Use median (robust to outliers)
# - External scores: Use median BUT create missing flag
# - Employment: Missing = 0 days (unemployed)
# =============================================================================

def impute_missing_values(df):
    """
    Industry-standard imputation strategy
    """
    df = df.copy()
    
    # 1. Categorical: Mode (most frequent)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # 2. External Sources: Median + Create missing flag
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    for col in ext_sources:
        if col in df.columns:
            df[f'{col}_MISSING'] = df[col].isnull().astype(int)
            df[col].fillna(df[col].median(), inplace=True)
    
    # 3. AMT features: Median
    amt_cols = [col for col in df.columns if col.startswith('AMT_')]
    for col in amt_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # 4. DAYS features: 0 (meaning "not applicable")
    days_cols = [col for col in df.columns if col.startswith('DAYS_')]
    for col in days_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(0, inplace=True)
    
    # 5. Remaining numerical: Median
    remaining_num = df.select_dtypes(include=['float64', 'int64']).columns
    for col in remaining_num:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    return df

print("Before imputation:")
print(f"Total missing values: {app_train.isnull().sum().sum()}")

app_train = impute_missing_values(app_train)
app_test = impute_missing_values(app_test)

print("\nAfter imputation:")
print(f"Total missing values in train: {app_train.isnull().sum().sum()}")
print(f"Total missing values in test: {app_test.isnull().sum().sum()}")


# In[16]:


# =============================================================================
# STEP 9: Create Banking-Specific Features
# WHY: Raw features don't capture risk relationships
# Banks use financial ratios extensively
# =============================================================================

def engineer_features(df):
    """
    Create features based on banking domain knowledge
    """
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # 1. INCOME RATIOS (Debt Serviceability)
    # -------------------------------------------------------------------------
    # Credit to Income: How much are they borrowing relative to income?
    # Rule: Should be < 3x income
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    
    # Annuity (EMI) to Income: Can they afford monthly payments?
    # Rule: Should be < 40% of income
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    
    # Loan-to-Value (LTV): How much of the goods price are they financing?
    # Lower LTV = more skin in the game = lower risk
    df['LTV_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
    
    # -------------------------------------------------------------------------
    # 2. EMPLOYMENT STABILITY
    # -------------------------------------------------------------------------
    # Employment to Age ratio: How much of their life have they worked?
    # Higher = more stable
    df['EMPLOYMENT_TO_AGE_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)
    
    # Income per family member
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    
    # -------------------------------------------------------------------------
    # 3. EXTERNAL SOURCE COMBINATIONS (Most Powerful)
    # -------------------------------------------------------------------------
    # Average of all external sources
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
    
    # Weighted average (EXT_SOURCE_2 and 3 are stronger predictors)
    df['EXT_SOURCE_WEIGHTED'] = (df['EXT_SOURCE_1'] * 0.2 + 
                                   df['EXT_SOURCE_2'] * 0.4 + 
                                   df['EXT_SOURCE_3'] * 0.4)
    
    # -------------------------------------------------------------------------
    # 4. AGE GROUPS (Non-linear relationship with risk)
    # -------------------------------------------------------------------------
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'], 
                              bins=[0, 25, 35, 45, 55, 100], 
                              labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elderly'])
    
    # -------------------------------------------------------------------------
    # 5. DOCUMENT FLAGS (Application Completeness)
    # -------------------------------------------------------------------------
    # Count how many documents were provided
    doc_cols = [col for col in df.columns if col.startswith('FLAG_DOCUMENT')]
    df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)
    
    # -------------------------------------------------------------------------
    # 6. ENQUIRY FLAGS (Credit Hungry Indicator)
    # -------------------------------------------------------------------------
    enquiry_cols = [col for col in df.columns if col.startswith('AMT_REQ_CREDIT_BUREAU')]
    df['TOTAL_ENQUIRIES'] = df[enquiry_cols].sum(axis=1)
    
    return df

# Apply feature engineering
print("Engineering features...")
app_train = engineer_features(app_train)
app_test = engineer_features(app_test)

print(f"New feature count: {app_train.shape[1]}")

# Show new features
new_features = ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'LTV_RATIO',
                'EXT_SOURCE_MEAN', 'EMPLOYMENT_TO_AGE_RATIO', 'INCOME_PER_PERSON']
print("\nNew Features Summary:")
print(app_train[new_features].describe())


# In[17]:


# =============================================================================
# STEP 10: Label Encoding for Categorical Features
# WHY: ML models need numerical inputs
# Using Label Encoding (not One-Hot) to avoid dimensionality explosion
# =============================================================================

def encode_categorical(df, encoder_dict=None):
    """
    Label encode categorical variables
    Return encoded df and encoder dict for test set
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove AGE_GROUP if it exists (will handle separately)
    if 'AGE_GROUP' in categorical_cols:
        categorical_cols.remove('AGE_GROUP')
    
    if encoder_dict is None:
        encoder_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoder_dict[col] = le
    else:
        for col in categorical_cols:
            if col in encoder_dict:
                # Transform using existing encoder
                df[col] = encoder_dict[col].transform(df[col].astype(str))
    
    # Handle AGE_GROUP separately (ordinal encoding)
    if 'AGE_GROUP' in df.columns:
        age_map = {'Very Young': 0, 'Young': 1, 'Middle': 2, 'Senior': 3, 'Elderly': 4}
        df['AGE_GROUP'] = df['AGE_GROUP'].map(age_map)
    
    return df, encoder_dict

# Encode training data
app_train_encoded, encoders = encode_categorical(app_train)

# Encode test data using same encoders
app_test_encoded, _ = encode_categorical(app_test, encoder_dict=encoders)

print(f"✓ Categorical encoding complete")
print(f"Encoded columns: {len(encoders)}")


# In[18]:


# =============================================================================
# STEP 11: Prepare Data for Modeling
# =============================================================================

# Separate features and target
X = app_train_encoded.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
y = app_train_encoded['TARGET']

# Train-validation split (Stratified to preserve class imbalance)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"\nClass distribution in train:")
print(y_train.value_counts(normalize=True))
print(f"\nClass distribution in validation:")
print(y_val.value_counts(normalize=True))

# Feature Scaling (important for Logistic Regression, not for tree models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)

print("\n✓ Data scaling complete")


# In[19]:


# =============================================================================
# STEP 12: Baseline Logistic Regression 
# WHY: Establishes performance floor
# Banks love interpretability - LR provides odds ratios
# =============================================================================

print("=" * 80)
print("BASELINE MODEL: LOGISTIC REGRESSION")
print("=" * 80)

# Train baseline model
lr_baseline = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_baseline.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_lr = lr_baseline.predict_proba(X_train_scaled)[:, 1]
y_val_pred_lr = lr_baseline.predict_proba(X_val_scaled)[:, 1]

# Evaluate
train_auc_lr = roc_auc_score(y_train, y_train_pred_lr)
val_auc_lr = roc_auc_score(y_val, y_val_pred_lr)

print(f"Training AUC: {train_auc_lr:.4f}")
print(f"Validation AUC: {val_auc_lr:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_lr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {val_auc_lr:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve - Baseline Logistic Regression')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Feature Importance (Top 20)
feature_importance_lr = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lr_baseline.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance_lr.head(20))


# In[20]:


# =============================================================================
# STEP 13: Address Class Imbalance
# WHY: Model is biased toward majority class (non-defaulters)
# Banking Solution: SMOTE (Synthetic Minority Oversampling)
# =============================================================================

from imblearn.over_sampling import SMOTE

print("=" * 80)
print("HANDLING CLASS IMBALANCE WITH SMOTE")
print("=" * 80)

print("Before SMOTE:")
print(y_train.value_counts())

# Apply SMOTE (create synthetic defaulters to balance classes)
smote = SMOTE(random_state=42, sampling_strategy=0.5)  # 50% defaulters after resampling
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Retrain Logistic Regression on balanced data
lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_train_resampled, y_train_resampled)

y_val_pred_lr_smote = lr_smote.predict_proba(X_val_scaled)[:, 1]
val_auc_lr_smote = roc_auc_score(y_val, y_val_pred_lr_smote)

print(f"\nValidation AUC after SMOTE: {val_auc_lr_smote:.4f}")
print(f"Improvement: {val_auc_lr_smote - val_auc_lr:.4f}")


# In[21]:


# =============================================================================
# STEP 14: Random Forest Classifier
# WHY: Handles non-linear relationships, feature interactions
# No need for scaling, robust to outliers
# =============================================================================

print("=" * 80)
print("ADVANCED MODEL: RANDOM FOREST")
print("=" * 80)

# Train Random Forest on SMOTE data
rf_model = RandomForestClassifier(
    n_estimators=100,        # 100 trees (balance between performance and speed)
    max_depth=10,            # Prevent overfitting
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,               # Use all CPU cores
    class_weight='balanced'  # Handle remaining imbalance
)

# Note: RF doesn't need scaled data
rf_model.fit(X_train_resampled, y_train_resampled)

# Predictions
y_val_pred_rf = rf_model.predict_proba(X_val)[:, 1]
val_auc_rf = roc_auc_score(y_val, y_val_pred_rf)

print(f"Random Forest Validation AUC: {val_auc_rf:.4f}")

# Feature Importance from RF
feature_importance_rf = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Important Features (Random Forest):")
print(feature_importance_rf.head(20))

# Visualize
plt.figure(figsize=(10, 8))
feature_importance_rf.head(20).plot(x='feature', y='importance', kind='barh', color='forestgreen')
plt.xlabel('Importance Score')
plt.title('Top 20 Features - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[22]:


# =============================================================================
# STEP 15: XGBoost - Industry Standard for Credit Risk
# WHY: Best performance, handles imbalance well, provides feature importance
# Used by most banks and fintech companies
# =============================================================================

# Check current data type
print("AGE_GROUP dtype before:", X_train['AGE_GROUP'].dtype)

# Convert category to numeric (if it exists)
if 'AGE_GROUP' in X_train.columns:
    X_train['AGE_GROUP'] = X_train['AGE_GROUP'].astype(int)
    X_val['AGE_GROUP'] = X_val['AGE_GROUP'].astype(int)
    print("✓ AGE_GROUP converted to int")

# Verify all columns are numeric
print("\nData types check:")
print(X_train.dtypes.value_counts())

# Now train XGBoost
print("=" * 80)
print("PRODUCTION MODEL: XGBOOST")
print("=" * 80)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc',
    enable_categorical=False  # Changed to False since we converted to int
)

# Train with early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# Predictions
y_val_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
val_auc_xgb = roc_auc_score(y_val, y_val_pred_xgb)

print(f"\nXGBoost Validation AUC: {val_auc_xgb:.4f}")

# Feature Importance
feature_importance_xgb = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Important Features (XGBoost):")
print(feature_importance_xgb.head(20))


# In[23]:


# =============================================================================
# STEP 16: Compare All Models
# =============================================================================

# Collect predictions
models = {
    'Logistic Regression': y_val_pred_lr,
    'LR + SMOTE': y_val_pred_lr_smote,
    'Random Forest': y_val_pred_rf,
    'XGBoost': y_val_pred_xgb
}

# Calculate AUC for all
results = []
for name, preds in models.items():
    auc = roc_auc_score(y_val, preds)
    results.append({'Model': name, 'AUC-ROC': auc})

results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)
print(results_df)

# Plot comparison
plt.figure(figsize=(12, 6))
for name, preds in models.items():
    fpr, tpr, _ = roc_curve(y_val, preds)
    auc = roc_auc_score(y_val, preds)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve Comparison - All Models', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[24]:


# =============================================================================
# STEP 17: Precision-Recall Trade-off
# WHY: In imbalanced datasets, PR curve is more informative than ROC
# Banking focus: Maximize recall (catch defaulters) while maintaining precision
# =============================================================================

plt.figure(figsize=(12, 6))

for name, preds in models.items():
    precision, recall, _ = precision_recall_curve(y_val, preds)
    plt.plot(recall, precision, label=name, linewidth=2)

plt.xlabel('Recall (True Positive Rate)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve\n(Higher = Better at catching defaulters without false alarms)', 
          fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[25]:


# =============================================================================
# STEP 18: Find Optimal Decision Threshold
# Banking Logic: Maximize F2-score (prioritizes recall over precision)
# =============================================================================

from sklearn.metrics import f1_score, fbeta_score

def find_optimal_threshold(y_true, y_pred_proba, beta=2):
    """
    Find threshold that maximizes F-beta score
    beta=2 means recall is 2x more important than precision
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []
    
    for thresh in thresholds:
        y_pred_class = (y_pred_proba >= thresh).astype(int)
        score = fbeta_score(y_true, y_pred_class, beta=beta)
        scores.append({'threshold': thresh, 'f2_score': score})
    
    scores_df = pd.DataFrame(scores)
    optimal = scores_df.loc[scores_df['f2_score'].idxmax()]
    return optimal['threshold'], optimal['f2_score']

# Find optimal threshold for XGBoost
optimal_thresh, optimal_f2 = find_optimal_threshold(y_val, y_val_pred_xgb)
print(f"Optimal Threshold: {optimal_thresh:.3f}")
print(f"F2-Score at optimal threshold: {optimal_f2:.4f}")

# Generate predictions at optimal threshold
y_val_pred_class = (y_val_pred_xgb >= optimal_thresh).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred_class)
print("\nConfusion Matrix:")
print(cm)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted: Good', 'Predicted: Default'],
            yticklabels=['Actual: Good', 'Actual: Default'])
plt.title(f'Confusion Matrix at Threshold = {optimal_thresh:.3f}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
print(f"\nPerformance Metrics:")
print(f"True Negatives (Correct rejections): {tn}")
print(f"False Positives (Good customers rejected): {fp}")
print(f"False Negatives (Defaulters approved - COSTLY): {fn}")
print(f"True Positives (Defaulters caught): {tp}")
print(f"\nRecall (Catch rate for defaulters): {tp/(tp+fn):.2%}")
print(f"Precision: {tp/(tp+fp):.2%}")
print(f"False Negative Rate (Miss rate): {fn/(fn+tp):.2%}")


# In[26]:


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

# Method 1: XGBoost Built-in Feature Importance
feature_importance_xgb = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize top 20
plt.figure(figsize=(12, 8))
top_20 = feature_importance_xgb.head(20)
colors = ['red' if 'EXT_SOURCE' in feat else 'steelblue' for feat in top_20['feature']]
plt.barh(range(len(top_20)), top_20['importance'], color=colors)
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Importance Score')
plt.title('Top 20 Most Important Features\n(Red = External Credit Scores)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("Top 20 Important Features:")
print(feature_importance_xgb.head(20))

# Method 2: Permutation Importance (more reliable than feature_importances_)
from sklearn.inspection import permutation_importance

print("\nCalculating permutation importance (this may take a minute)...")
perm_importance = permutation_importance(
    xgb_model, X_val.head(5000), y_val.head(5000), 
    n_repeats=10, random_state=42, n_jobs=-1
)

perm_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

print("\nTop 20 Features (Permutation Importance):")
print(perm_importance_df.head(20))

# Method 3: Individual Prediction Explanation
def explain_customer_decision(customer_idx):
    """
    Explain why a customer was approved/rejected
    """
    customer = X_val.iloc[customer_idx:customer_idx+1]
    prob = xgb_model.predict_proba(customer)[0, 1]
    decision = "REJECT - High Risk" if prob > 0.5 else "APPROVE - Low Risk"
    
    print("="*80)
    print(f"CUSTOMER PREDICTION EXPLANATION")
    print("="*80)
    print(f"Default Probability: {prob:.2%}")
    print(f"Decision: {decision}")
    print(f"\nTop 10 Risk Indicators for this customer:")
    
    top_features = feature_importance_xgb.head(10)['feature']
    for i, feat in enumerate(top_features, 1):
        val = customer[feat].values[0]
        avg_val = X_train[feat].mean()
        diff = val - avg_val
        direction = "↑ Higher" if diff > 0 else "↓ Lower"
        print(f"{i:2d}. {feat:30s}: {val:10.4f} ({direction} than average by {abs(diff):.2f})")
    
    return prob

# Test explanation on a high-risk customer
high_risk_idx = y_val[y_val == 1].index[0]  # Find a defaulter
explain_customer_decision(high_risk_idx)


# In[27]:


# =============================================================================
# SIMPLER VERSION: Just using positions 0-9
# =============================================================================

def explain_customer(position):
    """Simple explanation using position in validation set"""
    customer = X_val.iloc[position:position+1]
    prob = xgb_model.predict_proba(customer)[0, 1]
    actual = y_val.iloc[position]
    
    print("="*80)
    print(f"CUSTOMER #{position} RISK ASSESSMENT")
    print("="*80)
    print(f"Default Probability: {prob:.2%}")
    print(f"Recommendation: {'REJECT ❌' if prob > 0.5 else 'APPROVE ✅'}")
    print(f"Actual Outcome: {'Defaulted' if actual == 1 else 'Repaid'}")
    print(f"Model was: {'CORRECT ✓' if (prob > 0.5) == actual else 'WRONG ✗'}")
    
    print(f"\nTop 8 Risk Drivers:")
    top_features = feature_importance_xgb.head(8)['feature']
    for i, feat in enumerate(top_features, 1):
        val = customer[feat].values[0]
        avg_val = X_train[feat].mean()
        
        if 'EXT_SOURCE' in feat:
            status = "LOW (Risky)" if val < 0.5 else "HIGH (Good)"
        else:
            status = f"{val:.3f}"
        
        print(f"  {i}. {feat:30s}: {status}")
    
    return prob

# Find some defaulters manually
print("Finding defaulters in validation set...")
for i in range(100):
    if y_val.iloc[i] == 1:
        print(f"\n🔴 DEFAULTER FOUND AT POSITION {i}")
        explain_customer(i)
        break

# Find a good customer
print("\n" + "="*80)
for i in range(100):
    if y_val.iloc[i] == 0:
        print(f"\n🟢 GOOD CUSTOMER FOUND AT POSITION {i}")
        explain_customer(i)
        break


# In[32]:


# =============================================================================
# Save Model and Preprocessing Pipeline
# =============================================================================

import joblib

# Save model
joblib.dump(xgb_model, 'credit_risk_xgboost_model.pkl')

# Save scaler (if using LR in production)
joblib.dump(scaler, 'feature_scaler.pkl')

# Save feature names (critical for production)
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')

# Save encoders
joblib.dump(encoders, 'categorical_encoders.pkl')

print("✓ Model artifacts saved:")
print("  - credit_risk_xgboost_model.pkl")
print("  - feature_scaler.pkl")
print("  - feature_names.pkl")
print("  - categorical_encoders.pkl")


# In[28]:


# =============================================================================
# SAVE PRODUCTION MODEL
# =============================================================================

import joblib

# Save final model
joblib.dump(xgb_model, 'credit_risk_xgboost_final.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')

# Save feature names
feature_list = X_train.columns.tolist()
joblib.dump(feature_list, 'feature_names.pkl')

# Save performance metrics
metrics = {
    'validation_auc': val_auc_xgb,
    'model_type': 'XGBoost',
    'n_features': len(feature_list),
    'training_date': '2025-01-31'
}
joblib.dump(metrics, 'model_metrics.pkl')

print("✅ Model saved successfully!")
print(f"   • Model: credit_risk_xgboost_final.pkl")
print(f"   • Features: {len(feature_list)}")
print(f"   • AUC: {val_auc_xgb:.4f}")


# In[29]:


# =============================================================================
# Portfolio-Level Risk Assessment
# =============================================================================

# Predict default probabilities for entire validation set
val_portfolio = X_val.copy()
val_portfolio['DEFAULT_PROBABILITY'] = y_val_pred_xgb
val_portfolio['ACTUAL_DEFAULT'] = y_val.values
val_portfolio['CREDIT_AMOUNT'] = app_train_encoded.loc[X_val.index, 'AMT_CREDIT']

# Risk segmentation
val_portfolio['RISK_CATEGORY'] = pd.cut(
    val_portfolio['DEFAULT_PROBABILITY'],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)

# Portfolio analysis
portfolio_summary = val_portfolio.groupby('RISK_CATEGORY').agg({
    'DEFAULT_PROBABILITY': ['mean', 'count'],
    'ACTUAL_DEFAULT': 'mean',
    'CREDIT_AMOUNT': 'sum'
}).round(3)

portfolio_summary.columns = ['Avg Predicted Prob', 'Count', 'Actual Default Rate', 'Total Exposure']
print("\nPortfolio Risk Summary:")
print(portfolio_summary)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Distribution by risk category
portfolio_summary['Count'].plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Customer Distribution by Risk Category')
axes[0].set_ylabel('Number of Customers')
axes[0].set_xlabel('Risk Category')

# Plot 2: Exposure by risk category
portfolio_summary['Total Exposure'].plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Credit Exposure by Risk Category')
axes[1].set_ylabel('Total Credit Amount ($)')
axes[1].set_xlabel('Risk Category')

plt.tight_layout()
plt.show()


# In[31]:


# =============================================================================
# Calculate Business Impact
# Banking Economics:
# - Average loan size: $15,000
# - Loss Given Default (LGD): 70% (bank recovers 30% through collateral)
# - Cost of false positive: Lost interest income (~$500)
# - Cost of false negative: $15,000 * 0.7 = $10,500
# =============================================================================

# Assumptions (adjust based on your bank's data)
AVG_LOAN_AMOUNT = 15000
LOSS_GIVEN_DEFAULT = 0.7  # 70% loss
INTEREST_RATE = 0.12  # 12% annual interest
LOAN_TENURE = 3  # years

# Costs
COST_PER_DEFAULT = AVG_LOAN_AMOUNT * LOSS_GIVEN_DEFAULT  # $10,500
COST_PER_FALSE_POSITIVE = AVG_LOAN_AMOUNT * INTEREST_RATE * LOAN_TENURE * 0.3  # ~$1,620 (lost profit)

print("=" * 80)
print("BUSINESS IMPACT ANALYSIS")
print("=" * 80)

# Calculate costs for different thresholds
thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
impact_results = []

for thresh in thresholds_to_test:
    y_pred_class = (y_val_pred_xgb >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_class).ravel()
    
    # Calculate total cost
    cost_fn = fn * COST_PER_DEFAULT  # Missed defaulters
    cost_fp = fp * COST_PER_FALSE_POSITIVE  # Rejected good customers
    total_cost = cost_fn + cost_fp
    
    # Calculate saved amount (true positives)
    amount_saved = tp * COST_PER_DEFAULT
    
    impact_results.append({
        'Threshold': thresh,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Recall': tp/(tp+fn),
        'Precision': tp/(tp+fp) if (tp+fp) > 0 else 0,
        'Cost (FN)': cost_fn,
        'Cost (FP)': cost_fp,
        'Total Cost': total_cost,
        'Amount Saved': amount_saved,
        'Net Benefit': amount_saved - total_cost
    })

impact_df = pd.DataFrame(impact_results)
print(impact_df[['Threshold', 'Recall', 'Total Cost', 'Amount Saved', 'Net Benefit']])

# Find threshold with maximum net benefit
best_threshold_idx = impact_df['Net Benefit'].idxmax()
best_threshold_row = impact_df.iloc[best_threshold_idx]

print(f"\n{'='*80}")
print(f"OPTIMAL BUSINESS THRESHOLD: {best_threshold_row['Threshold']}")
print(f"Net Benefit: ${best_threshold_row['Net Benefit']:,.0f}")
print(f"Amount Saved by Catching Defaulters: ${best_threshold_row['Amount Saved']:,.0f}")
print(f"Cost of False Rejections: ${best_threshold_row['Cost (FP)']:,.0f}")
print(f"Cost of Missed Defaults: ${best_threshold_row['Cost (FN)']:,.0f}")
print(f"Recall: {best_threshold_row['Recall']:.2%}")
print(f"{'='*80}")


# In[28]:


# =============================================================================
# FINAL BUSINESS SUMMARY
# =============================================================================

print("="*80)
print("CREDIT RISK MODEL - BUSINESS IMPACT SUMMARY")
print("="*80)

print("\n MODEL PERFORMANCE:")
print(f"   • XGBoost AUC-ROC: {val_auc_xgb:.4f}")
print(f"   • Random Forest AUC: {val_auc_rf:.4f}")
print(f"   • Logistic Regression AUC: {val_auc_lr:.4f}")
print(f"   • Improvement over baseline: {(val_auc_xgb - val_auc_lr):.4f}")

print("\n KEY PREDICTORS:")
print("   1. External Credit Scores (EXT_SOURCE) - 20% importance")
print("   2. Education Level - 1.7% importance")
print("   3. Loan-to-Value Ratio - 1.3% importance")

print("\n BUSINESS RECOMMENDATIONS:")
print("   • Integrate with ALL 3 credit bureaus (not just 1-2)")
print("   • Require credit history for approval")
print("   • Cap LTV ratios for high-risk segments")
print("   • Use education as secondary risk factor")

print("\n DEPLOYMENT PLAN:")
print("   1. A/B test against current system (30 days)")
print("   2. Monitor default rate in approved applications")
print("   3. Set decision threshold at 0.45 (balances precision/recall)")
print("   4. Retrain quarterly with new data")

print("\n REGULATORY COMPLIANCE:")
print("   • Model is explainable (feature importance available)")
print("   • Can provide rejection reasons to customers")
print("   • Adverse action notices can reference top 3 risk factors")
print("="*80)


# In[ ]:


# =============================================================================
# IMPROVEMENT: ADDING CROSS-VALIDATION
# =============================================================================

from sklearn.model_selection import StratifiedKFold, cross_val_score

print("="*80)
print("CROSS-VALIDATION ANALYSIS (5-Fold Stratified)")
print("="*80)

# Define stratified k-fold (preserves class distribution)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. Logistic Regression with CV
print("\n1. Logistic Regression")
lr_cv = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_cv_scores = cross_val_score(lr_cv, X_train_scaled, y_train, 
                                 cv=skf, scoring='roc_auc', n_jobs=-1)
print(f"   CV AUC Scores: {lr_cv_scores}")
print(f"   Mean AUC: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std():.4f})")

# 2. Random Forest with CV
print("\n2. Random Forest")
rf_cv = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                random_state=42, class_weight='balanced', n_jobs=-1)
rf_cv_scores = cross_val_score(rf_cv, X_train, y_train, 
                                cv=skf, scoring='roc_auc', n_jobs=-1)
print(f"   CV AUC Scores: {rf_cv_scores}")
print(f"   Mean AUC: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

# 3. XGBoost with CV (manual loop for better control)
print("\n3. XGBoost (5-Fold CV)")
xgb_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    # Split data
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    # Train XGBoost
    xgb_fold = XGBClassifier(
        n_estimators=100,  # Reduced for speed
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=(y_fold_train == 0).sum() / (y_fold_train == 1).sum(),
        random_state=42,
        eval_metric='auc'
    )
    xgb_fold.fit(X_fold_train, y_fold_train, verbose=False)
    
    # Evaluate
    y_fold_pred = xgb_fold.predict_proba(X_fold_val)[:, 1]
    fold_auc = roc_auc_score(y_fold_val, y_fold_pred)
    xgb_cv_scores.append(fold_auc)
    print(f"   Fold {fold}: {fold_auc:.4f}")

print(f"\n   Mean AUC: {np.mean(xgb_cv_scores):.4f} (+/- {np.std(xgb_cv_scores):.4f})")

print("\n" + "="*80)
print("SUMMARY: Cross-Validation Results")
print("="*80)
print(f"Logistic Regression: {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")
print(f"Random Forest:       {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
print(f"XGBoost:             {np.mean(xgb_cv_scores):.4f} ± {np.std(xgb_cv_scores):.4f}")
print("="*80)


# In[33]:


print("="*80)
print("MODEL PERFORMANCE")
print("="*80)
print(f"XGBoost AUC:        {val_auc_xgb:.4f}")
print(f"Random Forest AUC:  {val_auc_rf:.4f}")
print(f"Logistic Reg AUC:   {val_auc_lr:.4f}")
print("="*80)


# In[ ]:




