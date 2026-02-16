
# AI Credit Model Fairness Audit: A CFA Charterholder's Workflow

## Case Study: Detecting and Mitigating Bias in Credit Decisions at GlobalFin Bank

### Introduction

As Ms. Anya Sharma, CFA, Head of Model Governance at GlobalFin Bank, you are tasked with ensuring that the bank's AI-powered credit scoring model operates not only efficiently but also ethically and in full compliance with evolving regulations. Recent scrutiny, particularly from the Equal Credit Opportunity Act (ECOA) in the US and the forthcoming EU AI Act, demands robust bias testing and documentation for high-risk AI systems like credit models.

Your primary objective is to conduct a comprehensive fairness audit of GlobalFin Bank's existing credit model. This involves identifying potential disparate impact against protected demographic groups, detecting subtle proxy variables, understanding individual-level discrimination through counterfactual analysis, and quantifying the trade-offs between model accuracy and fairness. The ultimate goal is to produce a regulatory-ready "Fairness Audit Report" that clearly articulates findings, compliance status, and recommended actions, bridging the gap between technical data science teams and compliance/legal departments.

This notebook will guide you through a step-by-step workflow that simulates a real-world fairness audit, empowering you with the tools and insights to uphold GlobalFin Bank's commitment to fair and responsible AI.

---

## 1. Environment Setup and Initial Model Loading

As a CFA Charterholder focused on model governance, the first step is to prepare your analytical environment. This includes installing necessary libraries and loading the pre-existing credit model and its associated data. For this exercise, we will simulate a credit dataset and a pre-trained model that Ms. Sharma would receive from the data science team.

### Code Cell (Function Definition + Execution)

```python
# Install required libraries
!pip install pandas numpy scikit-learn matplotlib seaborn shap fairlearn xgboost
```

### Code Cell (Function Definition + Execution)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import shap
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# Set random seed for reproducibility
np.random.seed(42)
pd.set_option('display.max_columns', None)

# Simulate a base credit dataset (representing D1-T2-C2 output)
def load_and_prepare_credit_data(n_samples=10000, test_size=0.3, random_state=42):
    """
    Simulates a credit dataset and a pre-trained XGBoost model.
    """
    data = {
        'income': np.random.normal(70000, 20000, n_samples),
        'debt_ratio': np.random.beta(2, 5, n_samples) * 0.5,
        'revolving_utilization': np.random.beta(1, 10, n_samples) * 0.8,
        'employment_length': np.random.randint(0, 30, n_samples),
        'num_dependents': np.random.randint(0, 5, n_samples),
        'credit_score': np.random.normal(680, 50, n_samples),
        'loan_amount': np.random.normal(15000, 5000, n_samples),
        'loan_duration_months': np.random.randint(12, 60, n_samples),
    }
    X_base = pd.DataFrame(data)

    # Scale features for model training (except categorical, if any, but all are numeric here)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_base), columns=X_base.columns)

    # Simulate a 'default' target variable (1=default, 0=no default)
    # Target heavily depends on credit_score, debt_ratio, revolving_utilization, income
    default_prob = (
        (X_scaled['debt_ratio'] * 0.4) +
        (X_scaled['revolving_utilization'] * 0.3) -
        (X_scaled['credit_score'] * 0.2) +
        (X_scaled['income'] * -0.1) +
        np.random.normal(0, 0.5, n_samples) # Add some noise
    )
    y_true = (default_prob > default_prob.median()).astype(int)

    # Train a simple XGBoost model to get realistic predictions
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_true, test_size=test_size, random_state=random_state, stratify=y_true)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int) # Standard threshold for prediction
    
    # We will treat 0 as 'approved' (no default) and 1 as 'denied' (default)
    # For fairness metrics, a favorable outcome is 'approved' (0).
    # So, we need to invert y_pred for 'approval_rate' where 1 means approved.
    y_pred_favorable = 1 - y_pred # 1 if approved, 0 if denied/default predicted
    y_true_favorable = 1 - y_test # 1 if actually approved, 0 if actually defaulted

    print("Initial Model Performance (on simulated test set):")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Accuracy: {(tp+tn)/(tn+fp+fn+tp):.4f}")
    print(f"Predicted Approval Rate: {y_pred_favorable.mean():.2%}")
    print(f"Actual Approval Rate: {y_true_favorable.mean():.2%}")

    return X_test, y_test, y_pred, y_pred_proba, y_pred_favorable, y_true_favorable, model, X_test.columns.tolist(), X_train, y_train, scaler

X_test_base, y_test, y_pred, y_pred_proba, y_pred_favorable, y_true_favorable, model, feature_cols, X_train_scaled, y_train_true, scaler = load_and_prepare_credit_data()

# Ensure we use the original (unscaled) X_test for synthetic demographic generation
# As the augment_with_demographics function works with raw feature values like 'income'
X_test_original = scaler.inverse_transform(X_test_base)
X_test_original = pd.DataFrame(X_test_original, columns=feature_cols)

print("\nSample of initial test data features:")
print(X_test_base.head())
```

### Markdown Cell (Explanation of Execution)

Ms. Sharma has successfully set up her environment and loaded the simulated credit dataset, which includes features like income, debt ratio, and credit score. A pre-trained `XGBClassifier` model, representing the bank's production credit scoring model, has also been loaded. The model's initial performance, including ROC AUC and predicted approval rates, provides a baseline for the fairness audit. Note that for fairness analysis, we define 'approval' as the favorable outcome, which is `y_pred_favorable` (1 if approved, 0 if denied). The original `y_pred` (1 for default, 0 for no default) is inverted for this purpose.

---

## 2. Augmenting Data with Synthetic Demographic Attributes

GlobalFin Bank, like many financial institutions, does not explicitly collect sensitive demographic data (race, gender, age) for credit decisions. However, for internal fairness auditing and methodology demonstration, Ms. Sharma needs to assess potential bias. She will augment the dataset with *synthetic* demographic attributes, ensuring a clear disclaimer that these are for analytical demonstration only and not for real production systems without legally obtained data.

### Markdown Cell (Story + Context + Real-World Relevance)

Ms. Sharma understands the sensitivity around demographic data. While production credit models at GlobalFin Bank intentionally exclude such attributes to avoid direct discrimination, regulatory bodies require robust evidence that no indirect discrimination occurs. To perform a comprehensive fairness audit, she needs to simulate how the model might behave across different demographic groups. For this lab, she will generate synthetic `gender`, `race_group`, and `age_group` attributes, carefully correlating them with existing financial features to mimic real-world proxy relationships (e.g., income with gender, revolving utilization with race, employment length with age).

The mathematical basis for creating these correlations is rooted in simple logistic functions or direct correlations, designed to create discernible (but not perfectly deterministic) relationships for testing purposes. For example, `gender_prob` is linked to `income` using a sigmoid-like function:
$$ \text{gender\_prob} = \frac{1}{1 + e^{-c \times \text{income\_norm}}} $$
where $c$ is a constant controlling the correlation strength, and $\text{income\_norm}$ is the normalized income. Similar logic applies to `race_prob` (correlated with `revolving_utilization`) and `age_group` (correlated with `employment_length`). This approach allows Ms. Sharma to test for "disparate impact" even when protected attributes are not direct model inputs.

### Code Cell (Function Definition + Execution)

```python
def augment_with_demographics(X, seed=42):
    """
    Adds synthetic demographic attributes (gender, race_group, age_group) to the DataFrame X.
    These attributes are weakly correlated with existing features to simulate realistic proxy relationships.
    In production: actual demographics from HMDA data or Bayesian Improved Surname Geocoding (BISG)
    would be used if legally permissible.
    """
    np.random.seed(seed)
    n = len(X)

    # Gender: weakly correlated with income
    income_norm = (X['income'] - X['income'].mean()) / X['income'].std()
    gender_prob = 1 / (1 + np.exp(-0.3 * income_norm)) # Slight correlation
    X['gender'] = np.random.binomial(1, gender_prob) # 1=Male, 0=Female

    # Race/ethnicity: correlated with ZIP code (simulated via revolving_utilization as proxy)
    util_norm = (X['revolving_utilization'] - X['revolving_utilization'].mean()) / X['revolving_utilization'].std()
    race_prob = 1 / (1 + np.exp(0.4 * util_norm))
    X['race_group'] = np.where(np.random.random(n) < race_prob, 'Group_A', 'Group_B')

    # Age: correlated with employment length
    # If 'employment_length' is not present, use a uniform random distribution for age.
    # Otherwise, use its value to correlate with age_group.
    employment_length_series = X.get('employment_length', pd.Series(np.random.uniform(0, 30, n), index=X.index))
    X['age_group'] = np.where(employment_length_series > 10, 'Over_40', 'Under_40')

    print("Demographic augmentation details:")
    print(f" Gender: {(X['gender']==1).mean():.0%} Male, {(X['gender']==0).mean():.0%} Female")
    print(f" Race: {(X['race_group']=='Group_A').mean():.0%} Group A, {(X['race_group']=='Group_B').mean():.0%} Group B")
    print(f" Age: {(X['age_group']=='Over_40').mean():.0%} Over 40, {(X['age_group']=='Under_40').mean():.0%} Under 40")

    return X

# Apply the augmentation
X_test_fair = augment_with_demographics(X_test_original.copy())

print("\nSample of augmented data with synthetic demographics:")
print(X_test_fair.head())
```

### Markdown Cell (Explanation of Execution)

The `augment_with_demographics` function has successfully added synthetic `gender`, `race_group`, and `age_group` attributes to the credit dataset (`X_test_fair`). The output shows the distribution of these newly created groups. Ms. Sharma now has the necessary (synthetic) attributes to proceed with fairness testing, acknowledging that in a real audit, legally and ethically sourced demographic data (e.g., HMDA, BISG) would be used. This step is crucial for methodological demonstration, allowing her to detect potential biases that might otherwise remain hidden.

---

## 3. Quantifying Group Fairness and Regulatory Compliance (Four-Fifths Rule)

Ms. Sharma's core responsibility is to ensure the credit model complies with fair lending laws. This requires quantifying how the model's approval rates differ across various demographic groups and applying regulatory thresholds like the "four-fifths rule" to identify disparate impact.

### Markdown Cell (Story + Context + Real-World Relevance)

As a CFA Charterholder, Ms. Sharma knows that "fair lending is not optional—it is the law." Disparate impact occurs when a neutral policy or model disproportionately affects a protected group, even without explicit discriminatory intent. The Equal Credit Opportunity Act (ECOA) prohibits such discrimination. The Equal Employment Opportunity Commission (EEOC) "four-fifths rule" is a widely recognized regulatory threshold: if the selection rate for a protected group is less than 80% (or four-fifths) of the selection rate for the most favored group, it generally indicates disparate impact.

Ms. Sharma needs to compute several key fairness metrics:

1.  **Disparate Impact Ratio (DIR)**: Measures the ratio of the favorable outcome rate (e.g., approval rate) for the disadvantaged group to the favorable outcome rate for the advantaged group.
    $$ \text{DIR} = \frac{P(\hat{Y} = \text{approve} \mid G = \text{disadvantaged})}{P(\hat{Y} = \text{approve} \mid G = \text{advantaged})} $$
    The **Four-fifths rule** states that $\text{DIR} \ge 0.80$ is required for compliance. Below 0.80 constitutes prima facie evidence of disparate impact.
2.  **Statistical Parity Difference (SPD)**: The absolute difference in favorable outcome rates between groups. Ideally, this should be close to zero.
    $$ \text{SPD} = |P(\hat{Y} = 1 \mid G = A) - P(\hat{Y} = 1 \mid G = B)| $$
    Where $\hat{Y} = 1$ is the favorable outcome (approval), and $G=A$ and $G=B$ represent two different demographic groups.
3.  **Equal Opportunity Difference (EOD)**: Measures the absolute difference in False Negative Rates (FNR) between groups. A high FNR for a disadvantaged group means truly creditworthy applicants are unfairly denied.
    $$ \text{EOD} = |\text{FNR}_A - \text{FNR}_B| $$
4.  **False Positive Rate (FPR) Parity**: Measures the absolute difference in False Positive Rates between groups. A high FPR for a disadvantaged group means truly uncreditworthy applicants are unfairly approved (less common concern in credit, but relevant for other domains).
    $$ \text{FPR Parity} = |\text{FPR}_A - \text{FPR}_B| $$
5.  **Predictive Parity**: Measures the absolute difference in Positive Predictive Values (PPV) between groups. If PPV differs, it means that among those predicted as creditworthy, the actual default rates differ across groups.
    $$ \text{Predictive Parity} = |\text{PPV}_A - \text{PPV}_B| $$
These metrics allow Ms. Sharma to quantitatively assess the model's fairness across different dimensions.

### Code Cell (Function Definition + Execution)

```python
def compute_fairness_metrics(y_true, y_pred_favorable, y_prob_favorable, sensitive_feature, feature_name):
    """
    Compute comprehensive fairness metrics for a protected attribute.
    y_true: actual outcomes (0=default, 1=no default)
    y_pred_favorable: predicted favorable outcome (1=approved, 0=denied)
    y_prob_favorable: predicted probability of favorable outcome
    sensitive_feature: the protected attribute (e.g., gender, race_group)
    feature_name: name of the sensitive feature for display
    """
    groups = np.unique(sensitive_feature)
    results = {}

    # Define favorable outcome for MetricFrame
    # y_true should be 0/1 where 1 is the positive class (non-default/approved)
    # y_pred_favorable should also be 0/1 where 1 is the positive class
    # For fairlearn metrics, we need to pass the true labels and predicted scores/labels.
    # Our y_true_favorable and y_pred_favorable are already aligned where 1 is 'approved'.

    for group in groups:
        mask = (sensitive_feature == group)
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred_favorable[mask]).ravel()
        
        # Calculate rates for the current group (1 = Favorable Outcome, 0 = Unfavorable Outcome)
        # Approval Rate = P(pred=1)
        # Base Rate = P(true=1)
        # FNR = False Negatives / (True Positives + False Negatives) = fn / (tp + fn)
        # FPR = False Positives / (True Negatives + False Positives) = fp / (tn + fp)
        # PPV = True Positives / (True Positives + False Positives) = tp / (tp + fp)

        group_approval_rate = tp / (tp + tn + fp + fn) # Equivalent to y_pred_favorable[mask].mean()
        group_base_rate = tp / (tp + fn) if (tp + fn) > 0 else 0 # (Actual Favorable Rate)

        group_fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        group_fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
        group_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

        results[group] = {
            'n': mask.sum(),
            'base_rate': y_true[mask].mean(), # Actual rate of being non-default (approved)
            'approval_rate': y_pred_favorable[mask].mean(), # Predicted rate of being approved
            'fnr': group_fnr,
            'fpr': group_fpr,
            'ppv': group_ppv,
            'auc': roc_auc_score(y_true[mask], y_prob_favorable[mask]) if len(np.unique(y_true[mask])) > 1 else np.nan,
        }

    # Identify advantaged and disadvantaged groups based on approval rate
    g_rates = {g: results[g]['approval_rate'] for g in groups}
    advantaged_group = max(g_rates, key=g_rates.get)
    disadvantaged_group = min(g_rates, key=g_rates.get)

    # 1. Disparate Impact Ratio (four-fifths rule)
    dir_val = results[disadvantaged_group]['approval_rate'] / max(0.001, results[advantaged_group]['approval_rate'])
    if dir_val > 1: # Ensure DIR is always <= 1 by making the disadvantaged group the numerator
        dir_val = 1 / dir_val 

    # 2. Statistical Parity Difference
    spd_val = abs(results[groups[0]]['approval_rate'] - results[groups[1]]['approval_rate'])

    # 3. Equal Opportunity Difference (FNR gap)
    eod_val = abs(results[groups[0]]['fnr'] - results[groups[1]]['fnr'])

    # 4. FPR Parity
    fpr_parity_val = abs(results[groups[0]]['fpr'] - results[groups[1]]['fpr'])

    # 5. Predictive Parity
    ppv_parity_val = abs(results[groups[0]]['ppv'] - results[groups[1]]['ppv'])

    # Fairlearn's combined Equalized Odds (FPR + FNR gap)
    # Using MetricFrame to get these from fairlearn
    sensitive_features_df = pd.DataFrame({'sensitive': sensitive_feature})
    mf = MetricFrame(metrics={
        'roc_auc_score': roc_auc_score,
        'demographic_parity_difference': demographic_parity_difference,
        'equalized_odds_difference': equalized_odds_difference,
    }, y_true=y_true, y_pred=y_pred_favorable, y_probas=y_prob_favorable, sensitive_features=sensitive_features_df['sensitive'])

    # Print results
    print(f"\nFAIRNESS METRICS: {feature_name}")
    print("=" * 55)
    for group, stats in results.items():
        print(f" {group}: n={stats['n']}, approval={stats['approval_rate']:.1%}, FNR={stats['fnr']:.3f}, FPR={stats['fpr']:.3f}, PPV={stats['ppv']:.3f}")

    print(f"\n Disparate Impact Ratio: {dir_val:.3f} ({'PASS' if dir_val >= 0.80 else 'FAIL'} four-fifths rule)")
    print(f" Statistical Parity Difference: {spd_val:.4f}")
    print(f" Equal Opportunity Difference (FNR Gap): {eod_val:.4f}")
    print(f" FPR Parity Difference: {fpr_parity_val:.4f}")
    print(f" Predictive Parity Difference: {ppv_parity_val:.4f}")
    print(f" Fairlearn Equalized Odds Difference: {mf.difference(metric_name='equalized_odds_difference'):.4f}")


    fairness_report_entry = {
        'feature': feature_name,
        'disparate_impact_ratio': round(dir_val, 3),
        'four_fifths_pass': dir_val >= 0.80,
        'statistical_parity_diff': round(spd_val, 4),
        'equal_opportunity_diff': round(eod_val, 4),
        'fpr_parity_diff': round(fpr_parity_val, 4),
        'predictive_parity_diff': round(ppv_parity_val, 4),
        'equalized_odds_diff_fairlearn': round(mf.difference(metric_name='equalized_odds_difference'), 4),
        'group_details': results,
        'advantaged_group': advantaged_group,
        'disadvantaged_group': disadvantaged_group,
    }
    return fairness_report_entry, results, advantaged_group, disadvantaged_group

# Store fairness metric results for the final report
metrics_by_group = {}
advantaged_groups = {}
disadvantaged_groups = {}

# Test fairness across all protected attributes
protected_attributes = [('Gender', 'gender'), ('Race', 'race_group'), ('Age', 'age_group')]

for attr_name, col_name in protected_attributes:
    entry, group_results, adv_g, dis_g = compute_fairness_metrics(
        y_true_favorable, y_pred_favorable, y_pred_proba, X_test_fair[col_name], attr_name
    )
    metrics_by_group[attr_name] = entry
    advantaged_groups[attr_name] = adv_g
    disadvantaged_groups[attr_name] = dis_g


# Visualization: Approval Rates by Group with Four-Fifths Rule
def plot_approval_rates(metrics_by_group_dict, protected_attributes):
    fig, axes = plt.subplots(1, len(protected_attributes), figsize=(18, 6), sharey=True)
    if len(protected_attributes) == 1:
        axes = [axes] # Ensure axes is iterable even for single subplot

    for i, (attr_name, _) in enumerate(protected_attributes):
        ax = axes[i]
        details = metrics_by_group_dict[attr_name]['group_details']
        groups = list(details.keys())
        approval_rates = [details[g]['approval_rate'] for g in groups]

        sns.barplot(x=groups, y=approval_rates, ax=ax, palette='viridis')
        ax.set_title(f'Approval Rate by {attr_name}')
        ax.set_ylabel('Approval Rate')
        ax.set_ylim(0, max(approval_rates) * 1.2) # Set y-limit for better visualization

        # Add Four-fifths rule line
        advantaged_rate = details[metrics_by_group_dict[attr_name]['advantaged_group']]['approval_rate']
        four_fifths_threshold = advantaged_rate * 0.80
        ax.axhline(four_fifths_threshold, color='red', linestyle='--', label='4/5ths Rule Threshold')
        ax.legend()
        ax.text(0.5, 0.95, f"DIR: {metrics_by_group_dict[attr_name]['disparate_impact_ratio']:.3f}",
                horizontalalignment='center', verticalalignment='top', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="red", lw=1, alpha=0.6))

    plt.tight_layout()
    plt.show()

plot_approval_rates(metrics_by_group, protected_attributes)
```

### Markdown Cell (Explanation of Execution)

Ms. Sharma has executed the `compute_fairness_metrics` function across all three synthetic protected attributes (`gender`, `race_group`, `age_group`). The output details the approval rates, False Negative Rates (FNR), False Positive Rates (FPR), and other key metrics for each group. Crucially, the Disparate Impact Ratio (DIR) is calculated, and its compliance with the "four-fifths rule" is immediately assessed (PASS/FAIL).

The visualizations provide a clear, intuitive understanding of the approval rate disparities. Ms. Sharma can see at a glance which demographic groups have lower approval rates and whether these fall below the critical 80% threshold. For instance, if 'Group B' for race has an approval rate significantly below 80% of 'Group A's rate, it signals a potential disparate impact violation requiring further investigation. This hands-on analysis allows her to identify compliance risks directly.

---

## 4. Identifying Potential Proxy Variables via SHAP Analysis

Even if direct demographic information is excluded, other features in the model might inadvertently act as "proxies" for protected attributes, leading to indirect discrimination. Ms. Sharma needs to identify these potential proxies, which are features that are both correlated with a protected attribute and highly important to the model's decision-making.

### Markdown Cell (Story + Context + Real-World Relevance)

The challenge of proxy variables is one of the hardest problems in fair lending. A feature like 'revolving_utilization' (how much of available credit is being used) is a legitimate business factor for creditworthiness. However, if 'revolving_utilization' is also highly correlated with a protected attribute (e.g., 'race_group' due to historical lending patterns or access to financial services in certain neighborhoods), and if the model relies heavily on this feature, it can become a proxy, perpetuating historical biases.

Ms. Sharma must identify features that exhibit a "dual condition":
1.  **Significant Correlation**: The feature is statistically correlated with a protected attribute.
2.  **High Model Importance**: The feature has a high impact on the model's predictions (e.g., measured by SHAP values).

The **Proxy Risk Score** quantifies this dual condition:
$$ \text{Proxy Risk Score} = |\text{Correlation with Protected Attribute}| \times \frac{\text{SHAP Importance}}{\text{Max SHAP Importance}} $$
where SHAP Importance is the mean absolute SHAP value for a feature. A high proxy risk score indicates a feature that warrants close scrutiny and potential business justification or mitigation. The decision to keep or remove a proxy variable is a policy judgment requiring legal, compliance, and business input, not purely a technical one.

### Code Cell (Function Definition + Execution)

```python
def detect_proxy_variables(model, X_data, protected_col_series, feature_cols, threshold=0.15):
    """
    Identifies features that correlate with the protected attribute AND have high model importance
    -- potential proxies.

    Args:
        model: Trained model (e.g., XGBClassifier).
        X_data: DataFrame of features used for prediction.
        protected_col_series: Series of the protected attribute.
        feature_cols: List of feature columns the model was trained on.
        threshold: Correlation threshold to flag as high-risk.

    Returns:
        DataFrame of proxy scores.
    """
    proxy_scores = []

    # Calculate correlation with protected attribute
    for feat in feature_cols:
        # Correlation with protected attribute (handle categorical vs numeric)
        if protected_col_series.dtype == 'object' or protected_col_series.nunique() <= 2:
            # Use point-biserial correlation for binary protected attribute and numeric feature
            # Or chi2 for two categorical variables (but we usually want correlation with a numeric feature)
            # For simplicity, we'll convert binary categorical to int for pointbiserialr
            # For multi-category (e.g. race_group with >2 groups), this is an oversimplification.
            # Here we assume a binary target for pointbiserialr by encoding.
            
            # Simple encoding for binary sensitive feature for correlation
            unique_vals = protected_col_series.unique()
            if len(unique_vals) == 2:
                encoded_protected = protected_col_series.map({unique_vals[0]: 0, unique_vals[1]: 1}).astype(int)
                from scipy.stats import pointbiserialr
                corr, p_val = pointbiserialr(encoded_protected, X_data[feat])
            else: # Handle multi-class categorical, e.g., using correlation ratio or just skipping
                corr, p_val = 0, 1 # Placeholder, complex multi-class correlation is out of scope for this simple function
        else:
            # Numeric protected attribute (e.g., age as continuous)
            from scipy.stats import pearsonr
            corr, p_val = pearsonr(protected_col_series, X_data[feat])

        proxy_scores.append({
            'feature': feat,
            'correlation_with_protected': abs(corr),
            'p_value': p_val,
        })

    proxy_df = pd.DataFrame(proxy_scores).sort_values('correlation_with_protected', ascending=False)

    # Also get SHAP importance
    # Use X_data scaled with the same scaler used for the model
    X_scaled_for_shap = scaler.transform(X_test_original[feature_cols])
    X_scaled_for_shap = pd.DataFrame(X_scaled_for_shap, columns=feature_cols)

    explainer = shap.TreeExplainer(model)
    # We only need SHAP values for a subset for performance, or for all of X_data for full importance.
    # For a comprehensive view, we compute for the full X_data.
    shap_values = explainer.shap_values(X_scaled_for_shap)
    # For multiclass, shap_values is a list of arrays. For binary, it's an array.
    if isinstance(shap_values, list): # For multi-output (e.g. 0 and 1 class), take absolute mean over outputs and then samples
        importance = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
    else: # For single output, take absolute mean over samples
        importance = np.abs(shap_values).mean(axis=0)

    importance_dict = dict(zip(feature_cols, importance))

    proxy_df['shap_importance'] = proxy_df['feature'].map(importance_dict)
    
    # Normalize SHAP importance for proxy risk score calculation
    max_shap_importance = proxy_df['shap_importance'].max()
    proxy_df['normalized_shap_importance'] = proxy_df['shap_importance'] / max_shap_importance

    # Calculate Proxy Risk Score
    proxy_df['proxy_risk_score'] = proxy_df['correlation_with_protected'] * proxy_df['normalized_shap_importance']

    # Flag high-risk proxies: correlation > threshold AND shap importance > median
    proxy_df['is_proxy'] = (
        (proxy_df['correlation_with_protected'] > threshold) &
        (proxy_df['shap_importance'] > proxy_df['shap_importance'].median())
    )

    print(f"\nPROXY VARIABLE DETECTION FOR {protected_col_series.name.upper()}:")
    print("=" * 70)
    print(f"{'Feature':<25s} {'Corr':>8s} {'SHAP':>8s} {'Risk':>8s} {'Proxy?':>8s}")
    print("-" * 70)
    for index, row in proxy_df.head(10).iterrows(): # Display top 10 by correlation
        flag = '>>> YES' if row['is_proxy'] else 'NO'
        print(f"{row['feature']:<25s} {row['correlation_with_protected']:>8.3f} {row['shap_importance']:>8.4f} {row['proxy_risk_score']:>8.4f} {flag:>8s}")

    return proxy_df

# Store proxy results for the final report
proxy_results_by_group = {}

# Detect proxy variables for each protected attribute
for attr_name, col_name in protected_attributes:
    current_proxy_df = detect_proxy_variables(model, X_test_base, X_test_fair[col_name], feature_cols, threshold=0.15)
    proxy_results_by_group[attr_name] = current_proxy_df

# Visualization: Proxy Risk Scatter Plot
def plot_proxy_risk(proxy_results_dict, protected_attributes):
    fig, axes = plt.subplots(1, len(protected_attributes), figsize=(18, 6))
    if len(protected_attributes) == 1:
        axes = [axes]

    for i, (attr_name, _) in enumerate(protected_attributes):
        ax = axes[i]
        df = proxy_results_dict[attr_name]
        
        sns.scatterplot(
            data=df,
            x='correlation_with_protected',
            y='shap_importance',
            hue='is_proxy',
            size='proxy_risk_score',
            sizes=(50, 500), # Size range of markers
            palette={True: 'red', False: 'blue'},
            alpha=0.7,
            ax=ax
        )
        ax.set_title(f'Proxy Risk for {attr_name}')
        ax.set_xlabel('Correlation with Protected Attribute (Absolute)')
        ax.set_ylabel('Model SHAP Importance (Mean Absolute)')
        ax.axhline(df['shap_importance'].median(), color='gray', linestyle=':', label='Median SHAP Importance')
        ax.axvline(0.15, color='orange', linestyle=':', label='Correlation Threshold (0.15)')
        ax.legend(title='Is Proxy?', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

plot_proxy_risk(proxy_results_by_group, protected_attributes)
```

### Markdown Cell (Explanation of Execution)

Ms. Sharma has successfully run the `detect_proxy_variables` function for each protected attribute, leveraging SHAP values to quantify model importance and statistical correlation to protected attributes. The tabular output highlights features with high proxy risk scores, indicating strong correlation with a protected attribute and significant model importance. The scatter plots visually represent this, with potential proxies highlighted in red, clearly showing features that are high on both correlation and SHAP importance axes.

This analysis allows Ms. Sharma to identify features that, while seemingly neutral (e.g., `revolving_utilization`, `employment_length`), might be indirectly driving biased outcomes. This is critical for model governance, as these features require documented business justification or consideration for mitigation strategies to ensure compliance and ethical AI deployment.

---

## 5. Conducting Counterfactual Fairness Tests

Ms. Sharma wants to move beyond group-level metrics to understand if the model treats individuals fairly. A key test for this is counterfactual fairness: would an applicant receive a different credit decision if only their protected attribute (and potentially related proxy features) were changed, while keeping all other financial characteristics identical?

### Markdown Cell (Story + Context + Real-World Relevance)

Individual fairness is a challenging concept, but counterfactual fairness provides a practical way to test it. Ms. Sharma uses this test to answer a specific question: "If a male applicant were female, or an 'Over 40' applicant were 'Under 40', would GlobalFin Bank's credit model still approve their loan, assuming all other financial factors remained exactly the same?" Significant changes in prediction outcomes under such counterfactual scenarios would indicate a lack of individual fairness, suggesting the model is sensitive to protected attributes, even if indirectly.

The process involves:
1.  **Creating Counterfactuals**: For each individual in the test set, create a hypothetical copy where only the protected attribute is flipped (e.g., Male becomes Female).
2.  **Adjusting Proxies (Optional but important)**: If proxy variables were identified, these too should be adjusted in the counterfactual to reflect how they *would* change if the protected attribute changed (e.g., if 'race_group' changes, and 'revolving_utilization' is a proxy, then 'revolving_utilization' might also be shifted to match the typical values of the counterfactual group). This accounts for indirect effects.
3.  **Comparing Predictions**: Run both the original and counterfactual individuals through the model and compare their predictions.

A significant `delta` in prediction probability or a `flipped` outcome (e.g., approved becomes denied) indicates a potential individual fairness issue. A model is considered *not* counterfactually fair if predictions change significantly upon alteration of only protected attributes.

### Code Cell (Function Definition + Execution)

```python
def counterfactual_test(model, X_data_original, y_pred_proba_original, protected_col_series, feature_cols, proxy_features=None):
    """
    For each individual, create a counterfactual: same financial profile but different protected group.
    Check if prediction changes. If predictions change significantly, the model is NOT counterfactually fair.

    Args:
        model: The trained credit model.
        X_data_original: The original test features DataFrame (unscaled, pre-demographic augmentation).
        y_pred_proba_original: Original model prediction probabilities.
        protected_col_series: Series of the protected attribute from X_test_fair.
        feature_cols: List of feature columns the model was trained on.
        proxy_features: List of identified proxy features for the current protected_col.
                        These will also be adjusted in the counterfactual.
    Returns:
        tuple: (delta, flipped) - Mean absolute prediction change and boolean series of flipped predictions.
    """
    if proxy_features is None:
        proxy_features = []

    # Prepare data for model input (scaling)
    X_original_scaled = scaler.transform(X_data_original[feature_cols])
    X_original_df = pd.DataFrame(X_original_scaled, columns=feature_cols, index=X_data_original.index)
    
    X_counterfactual_df = X_original_df.copy()

    # Determine original and counterfactual protected group values
    unique_groups = protected_col_series.unique()
    if len(unique_groups) != 2:
        print(f"Skipping counterfactual test for {protected_col_series.name} as it is not a binary protected attribute.")
        return np.nan, np.nan

    group1 = unique_groups[0]
    group2 = unique_groups[1] # The counterfactual group

    # For each individual, flip their protected attribute
    # And potentially adjust proxy features based on the typical values of the counterfactual group
    
    # Calculate group means for proxy features in the original unscaled data
    # This assumes X_data_original contains all features including demographic for reference
    # However, the proxy features are regular features.
    
    # We need to map the original protected_col_series to the index of X_data_original
    # Assuming X_data_original and protected_col_series have aligned indices.
    
    group_means_proxies = X_data_original.groupby(protected_col_series)[proxy_features].mean()

    # Iterate through individuals
    pred_original = y_pred_proba_original
    pred_counterfactual = np.zeros_like(pred_original)

    for idx in X_data_original.index:
        original_protected_group = protected_col_series.loc[idx]
        counterfactual_protected_group = group1 if original_protected_group == group2 else group2
        
        # Create counterfactual record for current individual
        individual_record_scaled = X_original_df.loc[[idx]].copy()
        
        # Adjust proxy features based on the typical values of the counterfactual group
        for proxy_feat in proxy_features:
            if proxy_feat in group_means_proxies.columns:
                # We want to shift the proxy feature for this individual towards the mean of the counterfactual group.
                # A simple way is to replace with the counterfactual group mean, but this might be too strong.
                # A more nuanced approach: shift by the difference between original group's mean and counterfactual group's mean
                # For this specific implementation, we'll try a simpler 'shift' based on group differences.
                
                # To get the shift value, we need group means of the *original unscaled* data for the proxy feature
                original_group_mean_proxy = group_means_proxies.loc[original_protected_group, proxy_feat]
                counterfactual_group_mean_proxy = group_means_proxies.loc[counterfactual_protected_group, proxy_feat]
                
                # Calculate the shift needed to move from original group's typical value to counterfactual group's typical value
                shift_val_unscaled = counterfactual_group_mean_proxy - original_group_mean_proxy
                
                # Apply this shift to the *unscaled* feature value, then re-scale for prediction
                original_val_unscaled = scaler.inverse_transform(individual_record_scaled)[[0], feature_cols.index(proxy_feat)]
                new_val_unscaled = original_val_unscaled + shift_val_unscaled
                
                # Re-scale this specific feature back into the scaled individual record
                # A bit tricky to do this in-place accurately without affecting other features' scaling
                # For simplicity, let's just replace the scaled value by transforming the new unscaled value
                # This requires re-scaling the entire feature column of X_data_original from scratch, or handling scaler on individual basis.
                # A simpler approach for the lab: directly apply the average shift observed in scaled values or replace with scaled mean.
                
                # For robustness, let's use the average scaled shift from the full X_train_scaled
                # This simulates how a proxy might change if the protected group changes.
                train_df_with_sensitive = X_train_scaled.copy()
                train_df_with_sensitive[protected_col_series.name] = y_train_true # Use y_train_true as a placeholder for sensitive feature in train set if needed for group means
                
                group_means_proxies_scaled = X_train_scaled.groupby(X_test_fair.loc[X_train_scaled.index, protected_col_series.name] if protected_col_series.name in X_test_fair.columns else y_train_true)[proxy_features].mean() # Using y_train_true as proxy for sensitive in train
                
                if not group_means_proxies_scaled.empty and proxy_feat in group_means_proxies_scaled.columns:
                    original_group_mean_proxy_scaled = group_means_proxies_scaled.loc[original_protected_group, proxy_feat]
                    counterfactual_group_mean_proxy_scaled = group_means_proxies_scaled.loc[counterfactual_protected_group, proxy_feat]
                    
                    shift_scaled_val = counterfactual_group_mean_proxy_scaled - original_group_mean_proxy_scaled
                    
                    # Apply this shift directly to the scaled feature value of the current individual
                    individual_record_scaled.loc[idx, proxy_feat] = individual_record_scaled.loc[idx, proxy_feat] + shift_scaled_val
                
        pred_counterfactual[X_data_original.index.get_loc(idx)] = model.predict_proba(individual_record_scaled)[:, 1]

    # Measure prediction changes
    delta = np.abs(pred_original - pred_counterfactual)
    
    # Predictions are 'approved' if prob > 0.5. Check if the binary decision flipped.
    flipped = ((pred_original > 0.5) != (pred_counterfactual > 0.5))

    print(f"\nCOUNTERFACTUAL FAIRNESS TEST for {protected_col_series.name.upper()}:")
    print("=" * 60)
    # Get top 3 proxy features for this protected attribute, if available
    current_proxies_df = proxy_results_by_group.get(protected_col_series.name, pd.DataFrame())
    top_proxies = current_proxies_df[current_proxies_df['is_proxy']]['feature'].tolist()
    
    if top_proxies:
        print(f"Proxy features adjusted: {', '.join(top_proxies[:3])}")
    else:
        print("No proxy features adjusted.")
        
    print(f"Mean prediction change: {delta.mean():.4f}")
    print(f"Max prediction change: {delta.max():.4f}")
    print(f"Predictions flipped: {flipped.mean():.2%}")

    if flipped.mean() > 0.05:
        print(f"\nFAIL: >{flipped.mean():.0%} of predictions change when protected attribute (and proxies) are adjusted.")
    else:
        print(f"\nPASS: <5% of predictions affected by counterfactual adjustment.")
        
    return delta, flipped

# Store counterfactual results for the final report
counterfactual_results = {}
counterfactual_deltas = {}

for attr_name, col_name in protected_attributes:
    current_proxy_df = proxy_results_by_group.get(attr_name, pd.DataFrame())
    top_proxies = current_proxy_df[current_proxy_df['is_proxy']]['feature'].tolist()

    delta, flipped = counterfactual_test(
        model, X_test_original, y_pred_proba, X_test_fair[col_name], feature_cols, proxy_features=top_proxies
    )
    if not isinstance(delta, float): # Check if the test was actually run (not skipped)
        counterfactual_results[attr_name] = {'delta_mean': delta.mean(), 'flipped_rate': flipped.mean()}
        counterfactual_deltas[attr_name] = delta
    else:
        counterfactual_results[attr_name] = {'delta_mean': np.nan, 'flipped_rate': np.nan}


# Visualization: Counterfactual Delta Histogram
def plot_counterfactual_deltas(counterfactual_deltas_dict, protected_attributes):
    fig, axes = plt.subplots(1, len(protected_attributes), figsize=(18, 6))
    if len(protected_attributes) == 1:
        axes = [axes]
    
    for i, (attr_name, _) in enumerate(protected_attributes):
        ax = axes[i]
        deltas = counterfactual_deltas_dict.get(attr_name)
        if deltas is not np.nan and deltas is not None:
            sns.histplot(deltas, bins=30, kde=True, ax=ax)
            ax.set_title(f'Counterfactual Prediction Change for {attr_name}')
            ax.set_xlabel('Absolute Change in Prediction Probability')
            ax.set_ylabel('Frequency')
        else:
            ax.set_title(f'Counterfactual Test Not Applicable for {attr_name}')
            ax.text(0.5, 0.5, 'Test Skipped (Not Binary)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

plot_counterfactual_deltas(counterfactual_deltas, protected_attributes)
```

### Markdown Cell (Explanation of Execution)

Ms. Sharma has performed counterfactual fairness tests for each protected attribute, taking into account previously identified proxy features. The output provides key metrics: the mean and max prediction changes, and most importantly, the percentage of predictions that "flipped" (i.e., changed from approval to denial or vice versa). A high flip rate indicates that the model's decisions are unacceptably sensitive to protected attributes, even after accounting for related proxy feature adjustments.

The histograms visualize the distribution of these prediction changes, allowing Ms. Sharma to see the magnitude and frequency of the model's sensitivity. If a significant number of applicants would receive a different decision solely based on their demographic group (and associated proxy shifts), it flags a serious individual fairness concern that GlobalFin Bank must address to ensure equitable treatment for all applicants.

---

## 6. Analyzing the Accuracy-Fairness Trade-off and Impossibility Theorem

Ms. Sharma now faces a critical decision point: how much model accuracy is GlobalFin Bank willing to "sacrifice" to achieve greater fairness? She needs to quantify this trade-off and understand the deeper implications of the "impossibility theorem" in fair machine learning.

### Markdown Cell (Story + Context + Real-World Relevance)

For GlobalFin Bank, maximizing model accuracy (e.g., higher AUC for predicting default) is a business imperative. However, strict adherence to fairness metrics might necessitate a slight reduction in overall accuracy. Ms. Sharma's role is to quantify this `accuracy-fairness trade-off` and communicate its economic and reputational implications to senior management and compliance.

She also needs to articulate the **Impossibility Theorem** (Chouldechova, 2017), which states that when base rates (actual default rates) differ across groups, it's generally mathematically impossible to satisfy all fairness metrics (e.g., statistical parity, equal opportunity, and predictive parity) simultaneously. This means GlobalFin Bank must make a policy decision about which fairness criterion to prioritize, guided by legal counsel and ethical considerations.

To assess the trade-off, Ms. Sharma will compare:
1.  **Unconstrained Model**: The original model, optimized solely for accuracy (e.g., AUC).
2.  **Fairness-Constrained Model**: A model retrained or adjusted to explicitly satisfy a fairness constraint (e.g., Demographic Parity). She will use `fairlearn.reductions.ExponentiatedGradient` for this, a powerful algorithm to find fair models.

The goal is to quantify the "cost of fairness" in terms of reduced AUC, and weigh it against the far greater financial and reputational costs of a fair lending violation (fines, lawsuits, reputational damage). Often, a 1-3% drop in AUC is an acceptable cost for achieving regulatory compliance and ethical standards.

### Code Cell (Function Definition + Execution)

```python
def compute_dir(y_true, y_pred_favorable, sensitive):
    """Helper to compute Disparate Impact Ratio."""
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return np.nan # DIR requires at least two groups

    # Calculate approval rates for all groups
    rates = [y_pred_favorable[sensitive == g].mean() for g in groups]
    
    # Identify advantaged and disadvantaged groups based on approval rate
    advantaged_rate = max(rates)
    disadvantaged_rate = min(rates)

    if advantaged_rate == 0:
        return np.nan # Avoid division by zero if no approvals in advantaged group
    
    dir_val = disadvantaged_rate / advantaged_rate
    return dir_val

def accuracy_fairness_tradeoff(X_train, y_train_true, X_test, y_test_true, sensitive_train, sensitive_test):
    """
    Compares an unconstrained model to a fairness-constrained model using fairlearn.
    Measures the accuracy cost of enforcing demographic parity.

    Args:
        X_train, y_train_true: Training data and true labels.
        X_test, y_test_true: Test data and true labels.
        sensitive_train, sensitive_test: Sensitive features for train and test sets.

    Returns:
        tuple: (base_auc, fair_auc, base_dir, fair_dir)
    """
    print("\nMEASURING ACCURACY-FAIRNESS TRADE-OFF:")
    print("=" * 60)

    # 1. Unconstrained model (Logistic Regression for simplicity, but could be XGBoost)
    base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42) # Use XGBoost for consistency
    base_model.fit(X_train, y_train_true)

    base_pred_proba = base_model.predict_proba(X_test)[:, 1]
    base_pred_favorable = (base_pred_proba > 0.5).astype(int) # Favorable outcome is 1 (approved)
    base_auc = roc_auc_score(y_test_true, base_pred_proba)
    base_dir = compute_dir(y_test_true, base_pred_favorable, sensitive_test)

    # 2. Fairness-constrained model (Exponentiated Gradient)
    # The estimator passed to ExponentiatedGradient should be an uncalibrated model
    # For fairlearn, y_true is 0/1 where 1 is the positive class (approved/non-default)
    # Our y_test_true is 0=default, 1=no default, which means 1 is the 'positive' class
    # For fairness, we are optimizing for approval, so we align our y_true to 1=approved
    
    # Ensure sensitive features in fairlearn have consistent indexing with X_train/X_test
    # This might require creating new Series with matching index
    sensitive_train_aligned = pd.Series(sensitive_train, index=X_train.index, name='sensitive_feature_eg')
    sensitive_test_aligned = pd.Series(sensitive_test, index=X_test.index, name='sensitive_feature_eg')
    
    fair_model = ExponentiatedGradient(
        estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        constraints=DemographicParity() # Constraint: equal approval rates across groups
    )
    
    fair_model.fit(X_train, y_train_true, sensitive_features=sensitive_train_aligned)

    fair_pred_proba = fair_model.predict_proba(X_test)[:, 1]
    fair_pred_favorable = (fair_pred_proba > 0.5).astype(int)
    fair_auc = roc_auc_score(y_test_true, fair_pred_proba)
    fair_dir = compute_dir(y_test_true, fair_pred_favorable, sensitive_test)

    print(f"{'Metric':<30s} {'Unconstrained':>15s} {'Fair Model':>15s}")
    print("-" * 55)
    print(f"{'AUC':<30s} {base_auc:>15.4f} {fair_auc:>15.4f}")
    print(f"{'Disparate Impact Ratio':<30s} {base_dir:>15.3f} {fair_dir:>15.3f}")
    print(f"{'Four-Fifths Rule':<30s} {'PASS' if base_dir >= 0.80 else 'FAIL':>15s} {'PASS' if fair_dir >= 0.80 else 'FAIL':>15s}")
    print(f"{'AUC Cost of Fairness':<30s} {'---':>15s} {base_auc - fair_auc:>+15.4f}")

    return base_auc, fair_auc, base_dir, fair_dir

# Use 'race_group' for the primary trade-off analysis as it often shows significant disparity
# We need sensitive features aligned with the train/test splits
# Let's augment X_train with sensitive features too, for a proper fairlearn train
X_train_original = scaler.inverse_transform(X_train_scaled)
X_train_original = pd.DataFrame(X_train_original, columns=feature_cols)
X_train_fair = augment_with_demographics(X_train_original.copy())

# Ensure y_train_true and y_test_true are 0/1 where 1 is the positive class (non-default/approved)
# Currently, y_train_true is 0=no default, 1=default. We want 1=approved for fairlearn target.
# So, we should pass 1-y_train_true and 1-y_test_true to fairness metrics functions if they expect favorable outcome as 1.
# However, fairlearn's ExponentiatedGradient usually expects the negative outcome as 1 for reduction. Let's re-verify.
# "For classification, y_true must be 0 or 1, with 1 representing the desired outcome."
# So, our y_train_true (0=no default, 1=default) should be inverted to 1=no default (approved), 0=default (denied)
y_train_favorable = 1 - y_train_true
y_test_favorable_tradeoff = 1 - y_test # Using y_test_true as input, ensure it's favorable for fairlearn

# Perform trade-off analysis for 'race_group'
base_auc, fair_auc, base_dir, fair_dir = accuracy_fairness_tradeoff(
    X_train_scaled, y_train_favorable, X_test_base, y_test_favorable_tradeoff,
    X_train_fair['race_group'], X_test_fair['race_group']
)

tradeoff_results = {
    'base_auc': base_auc,
    'fair_auc': fair_auc,
    'base_dir': base_dir,
    'fair_dir': fair_dir,
    'cost_of_fairness': base_auc - fair_auc
}

# Visualization: Accuracy-Fairness Pareto Curve (conceptual for this lab as we only compare two points)
def plot_accuracy_fairness_tradeoff(tradeoff_results):
    plt.figure(figsize=(8, 6))
    plt.scatter(tradeoff_results['base_dir'], tradeoff_results['base_auc'], color='blue', s=200, label='Unconstrained Model')
    plt.scatter(tradeoff_results['fair_dir'], tradeoff_results['fair_auc'], color='red', s=200, label='Fairness-Constrained Model')
    plt.plot([tradeoff_results['base_dir'], tradeoff_results['fair_dir']],
             [tradeoff_results['base_auc'], tradeoff_results['fair_auc']],
             color='gray', linestyle='--', alpha=0.7, label='Trade-off Path')

    plt.axvline(0.80, color='green', linestyle=':', label='4/5ths Rule Threshold (0.80 DIR)')

    plt.title('Accuracy-Fairness Trade-off (AUC vs. Disparate Impact Ratio)')
    plt.xlabel('Disparate Impact Ratio (DIR)')
    plt.ylabel('ROC AUC')
    plt.xlim(0, 1.05)
    plt.ylim(0.5, 1.0) # Assuming AUC is above 0.5
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

plot_accuracy_fairness_tradeoff(tradeoff_results)
```

### Markdown Cell (Explanation of Execution)

Ms. Sharma has performed a critical analysis of the accuracy-fairness trade-off using `fairlearn.reductions.ExponentiatedGradient`. The results clearly show the baseline performance of the unconstrained model (in terms of AUC and DIR) versus the fairness-constrained model. She can observe how much AUC might decrease to satisfy the four-fifths rule for Disparate Impact Ratio.

The output provides the concrete numbers needed to present to stakeholders. For example, a 0.02 drop in AUC might be a small price to pay to transform a "FAIL" on the four-fifths rule into a "PASS." The conceptual Pareto curve visually reinforces this trade-off. This quantitative understanding, coupled with the "impossibility theorem," empowers Ms. Sharma to lead informed policy discussions at GlobalFin Bank, ensuring that the bank makes deliberate, ethically sound, and legally compliant choices regarding its AI models.

---

## 7. Compiling a Regulatory-Ready Fairness Audit Report

All the analyses performed by Ms. Sharma—group fairness metrics, proxy variable detection, counterfactual tests, and trade-off analysis—must now be consolidated into a structured, regulatory-ready "Fairness Audit Report." This report will serve as the official record of the audit, informing compliance, legal, and risk management teams.

### Markdown Cell (Story + Context + Real-World Relevance)

The final step for Ms. Sharma as Head of Model Governance is to synthesize all findings into a clear, actionable report. This "AI Credit Model Fairness Audit Report" is a critical deliverable for GlobalFin Bank, demonstrating due diligence and providing a roadmap for addressing identified biases. It is designed to be easily digestible by non-technical stakeholders while containing sufficient detail for regulatory review (e.g., under ECOA or EU AI Act provisions for high-risk AI).

The report will summarize:
-   **Group Fairness Metrics**: Compliance status with the four-fifths rule for each protected attribute.
-   **Proxy Variables**: Identification of features acting as proxies, along with their risk scores.
-   **Counterfactual Fairness**: Assessment of individual-level discrimination.
-   **Accuracy-Fairness Trade-off**: Quantitative assessment of the performance impact of fairness constraints.
-   **Overall Assessment**: A concise conclusion (PASS, CONDITIONAL, FAIL) for the model's fairness.
-   **Recommended Actions**: Specific steps for GlobalFin Bank to take, such as investigating proxies, considering fairness-constrained retraining, or conducting adverse action analysis.

This report transforms complex technical analysis into strategic insights, enabling GlobalFin Bank to mitigate legal, financial, and reputational risks associated with algorithmic bias.

### Code Cell (Function Definition + Execution)

```python
def compile_fairness_report(metrics_by_group, proxy_results_by_group, counterfactual_results, tradeoff_results):
    """
    Produce an ECOA/EU AI Act aligned fairness audit report.

    Args:
        metrics_by_group (dict): Results from group fairness metrics.
        proxy_results_by_group (dict): Results from proxy detection.
        counterfactual_results (dict): Results from counterfactual tests.
        tradeoff_results (dict): Results from accuracy-fairness trade-off.

    Returns:
        dict: The structured fairness audit report.
    """
    report = {
        'title': 'AI CREDIT MODEL FAIRNESS AUDIT',
        'model': 'XGBoost Credit Default v2.1', # Placeholder, replace with actual model version
        'date': pd.Timestamp.now().isoformat(),
        'regulatory_framework': ['ECOA', 'EU AI Act (High-Risk)', 'EEOC Four-Fifths Rule'],
        'findings': [],
        'overall_assessment': None,
        'required_actions': [],
        'summary_metrics': {},
    }

    critical_failures = 0

    # 1. Assess each protected attribute for Disparate Impact (Four-Fifths Rule)
    print("\n--- FAIRNESS METRICS ASSESSMENT ---")
    for attr, metrics in metrics_by_group.items():
        report['summary_metrics'][attr] = {
            'disparate_impact_ratio': metrics['disparate_impact_ratio'],
            'four_fifths_pass': metrics['four_fifths_pass'],
            'statistical_parity_diff': metrics['statistical_parity_diff'],
            'equal_opportunity_diff': metrics['equal_opportunity_diff'],
        }
        
        if not metrics['four_fifths_pass']:
            critical_failures += 1
            report['findings'].append({
                'attribute': attr,
                'severity': 'CRITICAL',
                'finding': f"Disparate impact detected: DIR = {metrics['disparate_impact_ratio']:.3f} (below 0.80 threshold)",
                'action': 'Investigate proxy variables and consider fairness-constrained retraining'
            })
        else:
            report['findings'].append({
                'attribute': attr,
                'severity': 'PASS',
                'finding': f"Four-fifths rule satisfied: DIR = {metrics['disparate_impact_ratio']:.3f}"
            })

    # 2. Add proxy findings
    print("\n--- PROXY VARIABLE ASSESSMENT ---")
    all_proxies_found = []
    for attr, df in proxy_results_by_group.items():
        n_proxies = df['is_proxy'].sum()
        if n_proxies > 0:
            proxy_names = df[df['is_proxy']]['feature'].tolist()
            all_proxies_found.extend(proxy_names)
            report['findings'].append({
                'attribute': f'Proxy Variables ({attr})',
                'severity': 'WARNING',
                'finding': f"{n_proxies} proxy variables detected: {', '.join(proxy_names)}",
                'action': 'Review business justification for each proxy feature and consider mitigation'
            })
    if not all_proxies_found:
        report['findings'].append({
            'attribute': 'Proxy Variables',
            'severity': 'INFO',
            'finding': 'No high-risk proxy variables detected.',
            'action': 'Monitor for emerging proxy relationships'
        })


    # 3. Add counterfactual findings
    print("\n--- COUNTERFACTUAL FAIRNESS ASSESSMENT ---")
    for attr, res in counterfactual_results.items():
        if not pd.isna(res['flipped_rate']):
            if res['flipped_rate'] > 0.05: # Threshold for concern
                report['findings'].append({
                    'attribute': f'Counterfactual ({attr})',
                    'severity': 'WARNING',
                    'finding': f"{res['flipped_rate']:.2%} of predictions flipped when protected attribute (and proxies) changed.",
                    'action': 'Investigate individual-level discrimination; consider model retraining'
                })
            else:
                report['findings'].append({
                    'attribute': f'Counterfactual ({attr})',
                    'severity': 'PASS',
                    'finding': f"{res['flipped_rate']:.2%} of predictions flipped (acceptable level).",
                    'action': 'None required'
                })
        else:
            report['findings'].append({
                'attribute': f'Counterfactual ({attr})',
                'severity': 'INFO',
                'finding': f'Counterfactual test skipped for {attr} (not binary).',
                'action': 'N/A'
            })


    # 4. Overall assessment based on critical failures
    if critical_failures == 0:
        report['overall_assessment'] = 'PASS'
        report['required_actions'].append('Continue monitoring model fairness and performance.')
        if all_proxies_found:
            report['overall_assessment'] = 'CONDITIONAL'
            report['required_actions'].append('Review and document business justification for proxy variables.')
            report['required_actions'].append('Conduct adverse action analysis for declined applicants.')

    elif critical_failures <= 1:
        report['overall_assessment'] = 'CONDITIONAL'
        report['required_actions'].append('Implement fairness constraints for the flagged attribute.')
        report['required_actions'].append('Review and document business justification for proxy variables.')
        report['required_actions'].append('Conduct adverse action analysis for declined applicants.')
        report['required_actions'].append('Retest after mitigation within 90 days.')
    else:
        report['overall_assessment'] = 'FAIL'
        report['required_actions'].append('Suspend model for the affected use case.')
        report['required_actions'].append('Retrain with fairness constraints.')
        report['required_actions'].append('Full revalidation required before redeployment.')
        report['required_actions'].append('Conduct adverse action analysis for all declined applicants.')

    # 5. Add Trade-off Results
    print("\n--- ACCURACY-FAIRNESS TRADE-OFF ---")
    report['findings'].append({
        'attribute': 'Accuracy-Fairness Trade-off',
        'severity': 'INFO',
        'finding': (
            f"Unconstrained AUC: {tradeoff_results['base_auc']:.4f}, DIR: {tradeoff_results['base_dir']:.3f}. "
            f"Fairness-constrained AUC: {tradeoff_results['fair_auc']:.4f}, DIR: {tradeoff_results['fair_dir']:.3f}. "
            f"AUC Cost of Fairness: {tradeoff_results['cost_of_fairness']:.4f}."
        ),
        'action': 'Policy decision required on acceptable trade-off based on legal and ethical guidance.'
    })


    # Print report
    print("\n" + "=" * 60)
    print(f"FAIRNESS AUDIT REPORT: {report['title']}")
    print("=" * 60)
    print(f"Model: {report['model']}")
    print(f"Date: {report['date']}")
    print(f"Regulatory Framework: {', '.join(report['regulatory_framework'])}")
    print("\nOVERALL ASSESSMENT:")
    print(f"  -> {report['overall_assessment']}")

    print("\nFINDINGS:")
    for finding in report['findings']:
        print(f"\n[{finding['severity']}] {finding['attribute']}:")
        print(f"  {finding['finding']}")
        if 'action' in finding and finding['action'] != 'None required':
            print(f"  Action: {finding['action']}")

    if report['required_actions']:
        print("\nREQUIRED ACTIONS:")
        for action in report['required_actions']:
            print(f"  - {action}")

    print("\nSign-off:")
    print(f" Fair Lending Officer: __________________________ Date: ________")
    print(f" Compliance Director:  __________________________ Date: ________")
    print(f" Head of Model Governance (Anya Sharma, CFA): ____ Date: ________")

    return report

final_audit_report = compile_fairness_report(metrics_by_group, proxy_results_by_group, counterfactual_results, tradeoff_results)
```

### Markdown Cell (Explanation of Execution)

Ms. Sharma has successfully compiled the comprehensive "AI Credit Model Fairness Audit Report." The output presents a structured overview, beginning with a clear overall assessment (PASS, CONDITIONAL, or FAIL) based on the severity of identified biases. Each finding, from disparate impact to proxy variables and counterfactual sensitivity, is detailed with its severity and specific recommended actions.

This report serves as GlobalFin Bank's official record, providing transparent documentation for internal stakeholders and external regulators. For Ms. Sharma, this report is not just a summary of technical results but a strategic document that drives policy decisions, risk mitigation, and ensures GlobalFin Bank's AI systems adhere to the highest ethical and legal standards in fair lending.
