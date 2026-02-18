import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import shap
from fairlearn.metrics import (demographic_parity_difference, equalized_odds_difference, MetricFrame)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from scipy.stats import pointbiserialr, pearsonr # For proxy detection

# Set random seed for reproducibility
np.random.seed(42)
pd.set_option('display.max_columns', None)

# Helper function for Disparate Impact Ratio
def _compute_dir_ratio(y_pred_favorable, sensitive):
    """
    Disparate Impact Ratio (DIR) = min_group_selection_rate / max_group_selection_rate.
    Selection rate = P(pred=1) where 1 is favorable.
    Returns np.nan if <2 groups or if max rate is 0.
    """
    y = np.asarray(y_pred_favorable).astype(float)
    s = pd.Series(sensitive).reset_index(drop=True)

    if len(y) != len(s):
        raise ValueError(f"Length mismatch: y_pred ({len(y)}) vs sensitive ({len(s)})")

    groups = pd.unique(s.dropna())
    if len(groups) < 2:
        return np.nan

    rates = []
    for g in groups:
        mask = (s == g).to_numpy()
        if mask.sum() == 0:
            continue
        rates.append(y[mask].mean())

    if len(rates) < 2:
        return np.nan

    # Filter out NaNs if any, though with y.mean() on actual numbers, should be fine.
    valid_rates = [r for r in rates if not np.isnan(r)]
    if not valid_rates:
        return np.nan

    advantaged_rate = float(np.max(valid_rates))
    disadvantaged_rate = float(np.min(valid_rates))

    if advantaged_rate <= 0:
        return np.nan

    return disadvantaged_rate / advantaged_rate


def load_and_prepare_credit_data(n_samples=10000, test_size=0.3, random_state=42):
    """
    Simulates a credit dataset, splits it, scales features, and trains an XGBoost model.

    Returns:
        tuple: (X_test_scaled, y_test_true, y_pred, y_pred_proba, y_pred_favorable,
                y_true_favorable, model, feature_cols, X_train_scaled, y_train_true,
                scaler, X_test_original)
    """
    data = {
        'income': np.random.normal(70000, 20000, n_samples),
        'debt_ratio': np.random.beta(2, 5, n_samples) * 0.5,
        'revolving_utilization': np.random.beta(1, 10, n_samples) * 0.8,
        'employment_length': np.random.randint(0, 30, n_samples),
        'num_dependents': np.random.randint(0, 5, n_samples),
        'credit_score': np.random.normal(680, 50, n_samples),
        'loan_amount': np.random.normal(15000, 5000, n_samples),
        'loan_duration_months': np.random.randint(12, 60, n_samples)
    }
    X_base = pd.DataFrame(data)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_base), columns=X_base.columns)

    default_prob = (
        (X_scaled['debt_ratio'] * 0.4) +
        (X_scaled['revolving_utilization'] * 0.3) -
        (X_scaled['credit_score'] * 0.2) +
        (X_scaled['income'] * -0.1) +
        np.random.normal(0, 0.5, n_samples)
    )
    y_true = (default_prob > default_prob.median()).astype(int) # 1=default, 0=no default

    X_train_scaled, X_test_scaled, y_train_true, y_test_true = train_test_split(
        X_scaled, y_true, test_size=test_size, random_state=random_state, stratify=y_true
    )

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    model.fit(X_train_scaled, y_train_true)

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int) # 1=predicted default, 0=predicted no default

    # For fairness metrics, 1 typically represents the 'favorable' outcome (e.g., approval)
    y_pred_favorable = 1 - y_pred # 1 if approved, 0 if denied/default predicted
    y_true_favorable = 1 - y_test_true # 1 if actually approved, 0 if actually defaulted

    print("Initial Model Performance (on simulated test set):")
    print(f"ROC AUC: {roc_auc_score(y_test_true, y_pred_proba):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_test_true, y_pred).ravel()
    print(f"Accuracy: {(tp+tn)/(tn+fp+fn+tp):.4f}")
    print(f"Predicted Approval Rate: {y_pred_favorable.mean():.2%}")
    print(f"Actual Approval Rate: {y_true_favorable.mean():.2%}")

    feature_cols = X_scaled.columns.tolist()

    # Create X_test_original here for consistency (unscaled version of X_test_scaled)
    X_test_original = pd.DataFrame(scaler.inverse_transform(X_test_scaled), columns=feature_cols)

    return (X_test_scaled, y_test_true, y_pred, y_pred_proba, y_pred_favorable, y_true_favorable,
            model, feature_cols, X_train_scaled, y_train_true, scaler, X_test_original)


def augment_with_demographics(X: pd.DataFrame, seed=42) -> pd.DataFrame:
    """
    Adds synthetic demographic attributes (gender, race_group, age_group) to the DataFrame X.
    Returns the DataFrame with added demographic columns.
    """
    np.random.seed(seed)
    n = len(X)

    X_copy = X.copy()

    # Gender: weakly correlated with income
    income_norm = (X_copy['income'] - X_copy['income'].mean()) / X_copy['income'].std()
    gender_prob = 1 / (1 + np.exp(-0.3 * income_norm))
    X_copy['gender'] = np.random.binomial(1, gender_prob) # 1=Male, 0=Female

    # Race/ethnicity: correlated with revolving_utilization as proxy
    util_norm = (X_copy['revolving_utilization'] - X_copy['revolving_utilization'].mean()) / X_copy['revolving_utilization'].std()
    race_prob = 1 / (1 + np.exp(0.4 * util_norm))
    X_copy['race_group'] = np.where(np.random.random(n) < race_prob, 'Group_A', 'Group_B')

    # Age: correlated with employment length
    employment_length_series = X_copy.get('employment_length', pd.Series(np.random.uniform(0, 30, n), index=X_copy.index))
    X_copy['age_group'] = np.where(employment_length_series > 10, 'Over_40', 'Under_40')

    print("Demographic augmentation details:")
    print(f" Gender: {(X_copy['gender']==1).mean():.0%} Male, {(X_copy['gender']==0).mean():.0%} Female")
    print(f" Race: {(X_copy['race_group']=='Group_A').mean():.0%} Group A, {(X_copy['race_group']=='Group_B').mean():.0%} Group B")
    print(f" Age: {(X_copy['age_group']=='Over_40').mean():.0%} Over 40, {(X_copy['age_group']=='Under_40').mean():.0%} Under 40")

    return X_copy


def compute_fairness_metrics(y_true_favorable, y_pred_favorable, y_prob_favorable, sensitive_feature, feature_name):
    """
    Compute comprehensive fairness metrics for a protected attribute.
    Args:
        y_true_favorable: actual outcomes (1=favorable, 0=unfavorable)
        y_pred_favorable: predicted favorable outcome (1=approved, 0=denied)
        y_prob_favorable: predicted probability of favorable outcome (for AUC)
        sensitive_feature: protected attribute (e.g., gender, race_group)
        feature_name: name of the sensitive feature for display
    Returns:
        tuple: (fairness_report_entry, group_results, advantaged_group, disadvantaged_group)
    """
    y_true_aligned = pd.Series(y_true_favorable).reset_index(drop=True).astype(int)
    y_pred_aligned = pd.Series(y_pred_favorable).reset_index(drop=True).astype(int)
    y_prob_aligned = pd.Series(y_prob_favorable).reset_index(drop=True).astype(float)
    s_aligned = pd.Series(sensitive_feature).reset_index(drop=True)

    valid = s_aligned.notna()
    y_true_aligned = y_true_aligned[valid].reset_index(drop=True)
    y_pred_aligned = y_pred_aligned[valid].reset_index(drop=True)
    y_prob_aligned = y_prob_aligned[valid].reset_index(drop=True)
    s_aligned = s_aligned[valid].reset_index(drop=True)

    groups = pd.unique(s_aligned)
    results = {}

    for group in groups:
        mask = (s_aligned == group)
        if mask.sum() == 0:
            continue

        # confusion_matrix(y_true, y_pred, labels=[negative_class, positive_class])
        # Here, 0=unfavorable, 1=favorable
        tn, fp, fn, tp = confusion_matrix(
            y_true_aligned[mask],
            y_pred_aligned[mask],
            labels=[0, 1]
        ).ravel()

        denom_true_pos = tp + fn # Actual favorable cases
        denom_true_neg = tn + fp # Actual unfavorable cases
        denom_pred_pos = tp + fp # Predicted favorable cases
        denom_pred_neg = tn + fn # Predicted unfavorable cases

        group_fnr = fn / denom_true_pos if denom_true_pos > 0 else np.nan
        group_fpr = fp / denom_true_neg if denom_true_neg > 0 else np.nan
        group_ppv = tp / denom_pred_pos if denom_pred_pos > 0 else np.nan
        group_npv = tn / denom_pred_neg if denom_pred_neg > 0 else np.nan

        y_true_g = y_true_aligned[mask]
        y_prob_g = y_prob_aligned[mask]
        auc_g = roc_auc_score(y_true_g, y_prob_g) if y_true_g.nunique() > 1 else np.nan

        results[group] = {
            "n": int(mask.sum()),
            "base_rate": float(y_true_aligned[mask].mean()),
            "approval_rate": float(y_pred_aligned[mask].mean()),
            "fnr": float(group_fnr) if not pd.isna(group_fnr) else np.nan,
            "fpr": float(group_fpr) if not pd.isna(group_fpr) else np.nan,
            "ppv": float(group_ppv) if not pd.isna(group_ppv) else np.nan,
            "npv": float(group_npv) if not pd.isna(group_npv) else np.nan,
            "auc": float(auc_g) if not pd.isna(auc_g) else np.nan,
        }

    # Identify advantaged/disadvantaged by approval rate
    g_rates = {g: results[g]["approval_rate"] for g in groups if not pd.isna(results[g]["approval_rate"])}
    if not g_rates:
        advantaged_group, disadvantaged_group = None, None
        dir_val = np.nan
    else:
        advantaged_group = max(g_rates, key=g_rates.get)
        disadvantaged_group = min(g_rates, key=g_rates.get)
        dir_val = _compute_dir_ratio(y_pred_aligned, s_aligned)

    approval_rates = np.array([results[g]["approval_rate"] for g in groups if not pd.isna(results[g]["approval_rate"])])
    fnrs = np.array([results[g]["fnr"] for g in groups if not pd.isna(results[g]["fnr"])])
    fprs = np.array([results[g]["fpr"] for g in groups if not pd.isna(results[g]["fpr"])])
    ppvs = np.array([results[g]["ppv"] for g in groups if not pd.isna(results[g]["ppv"])])

    def max_min_gap(arr):
        return float(arr.max() - arr.min()) if arr.size > 0 else np.nan

    spd_val = max_min_gap(approval_rates)
    eod_val = max_min_gap(fnrs)
    fpr_parity_val = max_min_gap(fprs)
    ppv_parity_val = max_min_gap(ppvs)

    dp_diff = float(demographic_parity_difference(y_true_aligned, y_pred_aligned, sensitive_features=s_aligned))
    eo_diff = float(equalized_odds_difference(y_true_aligned, y_pred_aligned, sensitive_features=s_aligned))

    mf = MetricFrame(
        metrics={"selection_rate": lambda yt, yp: np.mean(yp), "base_rate": lambda yt, yp: np.mean(yt)},
        y_true=y_true_aligned, y_pred=y_pred_aligned, sensitive_features=s_aligned
    )

    print(f"\nFAIRNESS METRICS: {feature_name}")
    print("=" * 55)
    for group, stats in results.items():
        print(
            f" {group}: n={stats['n']}, "
            f"approval={stats['approval_rate']:.1%}, "
            f"FNR={stats['fnr'] if not np.isnan(stats['fnr']) else np.nan:.3f}, "
            f"FPR={stats['fpr'] if not np.isnan(stats['fpr']) else np.nan:.3f}, "
            f"PPV={stats['ppv'] if not np.isnan(stats['ppv']) else np.nan:.3f}"
        )

    print(f"\n Disparate Impact Ratio: {dir_val:.3f} ({'PASS' if (not pd.isna(dir_val) and dir_val >= 0.80) else 'FAIL'} four-fifths rule)")
    print(f" Statistical Parity Difference (gap): {spd_val:.4f}")
    print(f" Equal Opportunity Difference (FNR gap): {eod_val:.4f}")
    print(f" FPR Parity Difference (gap): {fpr_parity_val:.4f}")
    print(f" Predictive Parity Difference (PPV gap): {ppv_parity_val:.4f}")
    print(f" Fairlearn Demographic Parity Difference: {dp_diff:.4f}")
    print(f" Fairlearn Equalized Odds Difference: {eo_diff:.4f}")

    fairness_report_entry = {
        "feature": feature_name,
        "disparate_impact_ratio": round(dir_val, 3) if not pd.isna(dir_val) else np.nan,
        "four_fifths_pass": bool(dir_val >= 0.80) if not pd.isna(dir_val) else False,
        "statistical_parity_diff": round(spd_val, 4) if not np.isnan(spd_val) else np.nan,
        "equal_opportunity_diff": round(eod_val, 4) if not np.isnan(eod_val) else np.nan,
        "fpr_parity_diff": round(fpr_parity_val, 4) if not np.isnan(fpr_parity_val) else np.nan,
        "predictive_parity_diff": round(ppv_parity_val, 4) if not np.isnan(ppv_parity_val) else np.nan,
        "demographic_parity_diff_fairlearn": round(dp_diff, 4) if not pd.isna(dp_diff) else np.nan,
        "equalized_odds_diff_fairlearn": round(eo_diff, 4) if not pd.isna(eo_diff) else np.nan,
        "group_details": results,
        "advantaged_group": advantaged_group,
        "disadvantaged_group": disadvantaged_group,
        "metricframe_overview": {
            "by_group": mf.by_group.to_dict() if mf.by_group is not None else None
        },
    }

    return fairness_report_entry, results, advantaged_group, disadvantaged_group


def plot_approval_rates(metrics_by_group_dict, protected_attributes):
    """
    Generates a bar plot of approval rates by protected attribute groups.
    Returns the matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, len(protected_attributes), figsize=(18, 6), sharey=True)
    if len(protected_attributes) == 1:
        axes = [axes]

    for i, (attr_name, _) in enumerate(protected_attributes):
        ax = axes[i]
        details = metrics_by_group_dict[attr_name]['group_details']
        groups = list(details.keys())
        approval_rates = [details[g]['approval_rate'] for g in groups]

        sns.barplot(x=groups, y=approval_rates, ax=ax, palette='viridis')
        ax.set_title(f'Approval Rate by {attr_name}')
        ax.set_ylabel('Approval Rate')
        ax.set_ylim(0, max(approval_rates) * 1.2 if approval_rates else 1)

        advantaged_group = metrics_by_group_dict[attr_name].get('advantaged_group')
        if advantaged_group and advantaged_group in details:
            advantaged_rate = details[advantaged_group]['approval_rate']
            if not pd.isna(advantaged_rate):
                four_fifths_threshold = advantaged_rate * 0.80
                ax.axhline(four_fifths_threshold, color='red', linestyle='--', label='4/5ths Rule Threshold')
                ax.legend()
        dir_val = metrics_by_group_dict[attr_name].get('disparate_impact_ratio')
        if not pd.isna(dir_val):
            ax.text(0.5, 0.95, f"DIR: {dir_val:.3f}",
                    horizontalalignment='center', verticalalignment='top', transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="red", lw=1, alpha=0.6))

    plt.tight_layout()
    return fig


def detect_proxy_variables(model, X_data_scaled, protected_col_series, feature_cols, scaler_obj, threshold=0.15):
    """
    Identifies features that correlate with the protected attribute AND have high model importance
    -- potential proxies.

    Args:
        model: Trained model (e.g., XGBClassifier).
        X_data_scaled: DataFrame of SCALED features used for prediction (for SHAP).
        protected_col_series: Series of the protected attribute.
        feature_cols: List of feature columns the model was trained on.
        scaler_obj: The StandardScaler used to scale X_data_scaled.
        threshold: Correlation threshold to flag as high-risk.

    Returns:
        DataFrame: A DataFrame of proxy scores.
    """
    proxy_scores = []

    # Unscale X_data_scaled for correlation calculation with the unscaled protected attribute
    X_data_unscaled = pd.DataFrame(scaler_obj.inverse_transform(X_data_scaled), columns=feature_cols)

    for feat in feature_cols:
        # Correlation with protected attribute (handle categorical vs numeric)
        if protected_col_series.dtype == 'object' or protected_col_series.nunique() <= 2:
            unique_vals = protected_col_series.dropna().unique()
            if len(unique_vals) == 2:
                # Align series before correlation to handle missing values consistently
                aligned_protected = protected_col_series.loc[X_data_unscaled.index].fillna(unique_vals[0]) # Fill NaN for correlation
                encoded_protected = aligned_protected.map({unique_vals[0]: 0, unique_vals[1]: 1}).astype(int)
                corr, p_val = pointbiserialr(encoded_protected, X_data_unscaled[feat])
            else: # Multi-class categorical, simplified to 0 correlation for this specific function
                corr, p_val = 0, 1
        else: # Numeric protected attribute
            corr, p_val = pearsonr(protected_col_series.dropna(), X_data_unscaled[feat].loc[protected_col_series.dropna().index])

        proxy_scores.append({
            'feature': feat,
            'correlation_with_protected': abs(corr),
            'p_value': p_val
        })

    proxy_df = pd.DataFrame(proxy_scores).sort_values('correlation_with_protected', ascending=False)

    # Get SHAP importance (uses scaled data)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data_scaled)

    if isinstance(shap_values, list): # For multi-output (e.g. 0 and 1 class), take absolute mean over outputs and then samples
        importance = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
    else: # For single output, take absolute mean over samples
        importance = np.abs(shap_values).mean(axis=0)

    importance_dict = dict(zip(feature_cols, importance))

    proxy_df['shap_importance'] = proxy_df['feature'].map(importance_dict)

    max_shap_importance = proxy_df['shap_importance'].max()
    proxy_df['normalized_shap_importance'] = proxy_df['shap_importance'] / max_shap_importance if max_shap_importance > 0 else 0

    proxy_df['proxy_risk_score'] = proxy_df['correlation_with_protected'] * proxy_df['normalized_shap_importance']

    median_shap = proxy_df['shap_importance'].median()
    proxy_df['is_proxy'] = (
        (proxy_df['correlation_with_protected'] > threshold) &
        (proxy_df['shap_importance'] > median_shap)
    )

    print(f"\nPROXY VARIABLE DETECTION FOR {protected_col_series.name.upper()}:")
    print("=" * 70)
    print(f"{'Feature':<25s} {'Corr':>8s} {'SHAP':>8s} {'Risk':>8s} {'Proxy?':>8s}")
    print("-" * 70)
    for index, row in proxy_df.head(10).iterrows(): # Display top 10 by correlation
        flag = '>>> YES' if row['is_proxy'] else 'NO'
        print(f"{row['feature']:<25s} {row['correlation_with_protected']:>8.3f} {row['shap_importance']:>8.4f} {row['proxy_risk_score']:>8.4f} {flag:>8s}")

    return proxy_df


def plot_proxy_risk(proxy_results_dict, protected_attributes):
    """
    Generates a scatter plot visualizing proxy risk.
    Returns the matplotlib Figure object.
    """
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
            sizes=(50, 500),
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
    return fig


def counterfactual_test(
    model,
    X_test_original: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_pred_proba_original,
    protected_col_series: pd.Series,
    protected_col_name: str,
    feature_cols: list,
    proxy_features: list,
    scaler_obj: StandardScaler,
    X_train_scaled: pd.DataFrame,
    X_train_fair: pd.DataFrame,
    *,
    threshold: float = 0.5,
    mode: str = "shift"
):
    """
    Performs a counterfactual fairness test by adjusting proxy features for a protected group.
    Args:
        model: Trained model.
        X_test_original: Unscaled test features.
        X_test_scaled: Scaled test features (for model prediction).
        y_pred_proba_original: Original predicted probabilities on test set.
        protected_col_series: Series of the protected attribute for the test set.
        protected_col_name: Name of the protected attribute column.
        feature_cols: List of feature columns used by the model.
        proxy_features: List of features identified as proxies.
        scaler_obj: The StandardScaler object used for feature scaling.
        X_train_scaled: Scaled training features.
        X_train_fair: Unscaled training features with demographic data.
        threshold: Prediction threshold for binary outcomes.
        mode: How to adjust proxy features ("shift" or "replace").
    Returns:
        tuple: (delta, flipped) where delta is absolute change in prediction probability
               and flipped indicates if prediction changed (boolean array).
    """
    if not proxy_features:
        print(f"Skipping counterfactual test for {protected_col_name}: no proxies provided.")
        return np.nan, np.nan

    X_test_unscaled_aligned = X_test_original.reset_index(drop=True)
    s_test = protected_col_series.reset_index(drop=True)
    pred_original = pd.Series(y_pred_proba_original).reset_index(drop=True).to_numpy(dtype=float)

    groups = pd.unique(s_test.dropna())
    if len(groups) != 2:
        print(f"Skipping counterfactual test for {protected_col_name} (not binary). Groups: {list(groups)}")
        return np.nan, np.nan

    g1, g2 = groups[0], groups[1]

    proxy_features = [f for f in proxy_features if f in feature_cols]
    if not proxy_features:
        print(f"Skipping counterfactual test for {protected_col_name}: no valid proxies in feature_cols.")
        return np.nan, np.nan

    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)

    X_train_scaled_reset = X_train_scaled.reset_index(drop=True)
    s_train = X_train_fair[protected_col_name].reset_index(drop=True)

    # Filter for valid sensitive feature values in train set
    valid_train_mask = s_train.notna()
    X_train_scaled_reset = X_train_scaled_reset[valid_train_mask].reset_index(drop=True)
    s_train = s_train[valid_train_mask].reset_index(drop=True)

    group_means_scaled = X_train_scaled_reset.groupby(s_train)[proxy_features].mean()

    if (g1 not in group_means_scaled.index) or (g2 not in group_means_scaled.index):
        print(
            f"Skipping counterfactual test for {protected_col_name}: "
            f"train data missing one of the groups {g1}, {g2} for mean computation."
        )
        return np.nan, np.nan

    X_cf = X_test_scaled_df.copy()
    cf_group = s_test.map(lambda v: g1 if v == g2 else g2)

    for f in proxy_features:
        mu_orig = s_test.map(lambda v: group_means_scaled.loc[v, f]).to_numpy(dtype=float)
        mu_cf = cf_group.map(lambda v: group_means_scaled.loc[v, f]).to_numpy(dtype=float)

        if mode == "replace":
            X_cf[f] = mu_cf
        elif mode == "shift":
            X_cf[f] = X_cf[f].to_numpy(dtype=float) + (mu_cf - mu_orig)
        else:
            raise ValueError("mode must be 'shift' or 'replace'")

    pred_counterfactual = model.predict_proba(X_cf)[:, 1].astype(float)

    delta = np.abs(pred_original - pred_counterfactual)
    flipped = ((pred_original > threshold) != (pred_counterfactual > threshold))

    print(f"\nCOUNTERFACTUAL FAIRNESS TEST for {protected_col_name.upper()}:")
    print("=" * 60)
    print(f"Proxy features adjusted ({mode}): {', '.join(proxy_features[:10])}{'...' if len(proxy_features) > 10 else ''}")
    print(f"Mean prediction change: {delta.mean():.4f}")
    print(f"Max prediction change: {delta.max():.4f}")
    print(f"Predictions flipped: {flipped.mean():.2%}")

    if flipped.mean() > 0.05:
        print(f"\nFAIL: {flipped.mean():.2%} of predictions change when proxies are adjusted.")
    else:
        print(f"\nPASS: <5% of predictions affected by counterfactual adjustment.")

    return delta, flipped


def plot_counterfactual_deltas(counterfactual_deltas_dict, protected_attributes):
    """
    Generates histograms of absolute prediction changes from counterfactual testing.
    Returns the matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, len(protected_attributes), figsize=(18, 6))
    if len(protected_attributes) == 1:
        axes = [axes]

    for i, (attr_name, _) in enumerate(protected_attributes):
        ax = axes[i]
        deltas = counterfactual_deltas_dict.get(attr_name, None)

        if isinstance(deltas, np.ndarray) and len(deltas) > 0:
            sns.histplot(deltas, bins=30, kde=True, ax=ax)
            ax.set_title(f'Counterfactual Prediction Change for {attr_name}')
            ax.set_xlabel('Absolute Change in Prediction Probability')
            ax.set_ylabel('Frequency')
        else:
            ax.set_title(f'Counterfactual Test Not Applicable for {attr_name}')
            ax.text(0.5, 0.5, 'Test Skipped', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    return fig


def accuracy_fairness_tradeoff(
    X_train_scaled,
    y_train_favorable,
    X_test_scaled,
    y_test_favorable,
    sensitive_train,
    sensitive_test,
    *,
    threshold: float = 0.5,
    xgb_params=None
):
    """
    Compares an unconstrained XGBoost model against a DemographicParity-constrained model.
    Args:
        X_train_scaled: Scaled training features.
        y_train_favorable: Favorable outcomes for training (1=favorable).
        X_test_scaled: Scaled test features.
        y_test_favorable: Favorable outcomes for testing (1=favorable).
        sensitive_train: Sensitive features for the training set.
        sensitive_test: Sensitive features for the test set.
        threshold: Prediction threshold.
        xgb_params: Parameters for XGBoostClassifier.
    Returns:
        dict: A dictionary containing performance and fairness metrics for both models.
    """
    print("\nMEASURING ACCURACY-FAIRNESS TRADE-OFF:")
    print("=" * 60)

    X_train_scaled = pd.DataFrame(X_train_scaled).reset_index(drop=True)
    X_test_scaled = pd.DataFrame(X_test_scaled).reset_index(drop=True)
    y_train_favorable = pd.Series(y_train_favorable).reset_index(drop=True).astype(int)
    y_test_favorable = pd.Series(y_test_favorable).reset_index(drop=True).astype(int)

    s_train = pd.Series(sensitive_train).reset_index(drop=True)
    s_test = pd.Series(sensitive_test).reset_index(drop=True)

    if xgb_params is None:
        xgb_params = dict(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )

    # 1) Unconstrained model
    base_model = XGBClassifier(**xgb_params)
    base_model.fit(X_train_scaled, y_train_favorable)

    base_proba = base_model.predict_proba(X_test_scaled)[:, 1]
    base_pred = (base_proba >= threshold).astype(int)

    base_auc = roc_auc_score(y_test_favorable, base_proba)
    base_dir = _compute_dir_ratio(base_pred, s_test)
    base_acc = accuracy_score(y_test_favorable, base_pred)

    # 2) Fairness constrained model
    fair_model = ExponentiatedGradient(
        estimator=XGBClassifier(**xgb_params),
        constraints=DemographicParity()
    )
    fair_model.fit(X_train_scaled, y_train_favorable, sensitive_features=s_train)

    fair_pred = fair_model.predict(X_test_scaled).astype(int)
    fair_dir = _compute_dir_ratio(fair_pred, s_test)
    fair_acc = accuracy_score(y_test_favorable, fair_pred)

    print(f"{'Metric':<30s} {'Unconstrained':>15s} {'Fair Model':>15s}")
    print("-" * 60)
    print(f"{'ROC AUC':<30s} {base_auc:>15.4f} {'(n/a)':>15s}")
    print(f"{'Accuracy':<30s} {base_acc:>15.4f} {fair_acc:>15.4f}")
    print(f"{'Disparate Impact Ratio':<30s} {base_dir:>15.3f} {fair_dir:>15.3f}")
    print(f"{'Four-Fifths Rule':<30s} "
          f"{'PASS' if (not pd.isna(base_dir) and base_dir >= 0.80) else 'FAIL':>15s} "
          f"{'PASS' if (not pd.isna(fair_dir) and fair_dir >= 0.80) else 'FAIL':>15s}")
    print(f"{'Accuracy Cost of Fairness':<30s} {'---':>15s} {(base_acc - fair_acc):>+15.4f}")

    return {
        "base_auc": float(base_auc),
        "base_acc": float(base_acc),
        "fair_acc": float(fair_acc),
        "base_dir": float(base_dir) if not np.isnan(base_dir) else np.nan,
        "fair_dir": float(fair_dir) if not np.isnan(fair_dir) else np.nan,
        "acc_cost_of_fairness": float(base_acc - fair_acc),
    }


def plot_accuracy_fairness_tradeoff(tradeoff_results_dict):
    """
    Generates a scatter plot illustrating the accuracy-fairness trade-off.
    Returns the matplotlib Figure object.
    """
    fig = plt.figure(figsize=(8, 6))
    if tradeoff_results_dict.get("base_dir") is not None and tradeoff_results_dict.get("base_auc") is not None:
        plt.scatter(tradeoff_results_dict["base_dir"], tradeoff_results_dict["base_auc"], s=200, label="Unconstrained (AUC)", marker="o")
    if tradeoff_results_dict.get("fair_dir") is not None and tradeoff_results_dict.get("fair_acc") is not None:
        plt.scatter(tradeoff_results_dict["fair_dir"], tradeoff_results_dict["fair_acc"], s=200, label="Fair Model (Accuracy)", marker="s")

    plt.axvline(0.80, linestyle=":", label="4/5ths Rule (DIR=0.80)")
    plt.title("Accuracy–Fairness Trade-off")
    plt.xlabel("Disparate Impact Ratio (DIR)")
    plt.ylabel("Score (AUC for base, Accuracy for fair)")
    plt.xlim(0, 1.05)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    return fig


def compile_fairness_report(metrics_by_group, proxy_results_by_group, counterfactual_results, tradeoff_results):
    """
    Compiles a comprehensive fairness audit report based on various analysis results.
    Args:
        metrics_by_group (dict): Fairness metrics for each protected group.
        proxy_results_by_group (dict): Proxy variable detection results.
        counterfactual_results (dict): Counterfactual test results.
        tradeoff_results (dict): Accuracy-fairness trade-off results.
    Returns:
        dict: The structured fairness audit report.
    """
    report = {
        "title": "AI CREDIT MODEL FAIRNESS AUDIT",
        "model": "XGBoost Credit Default v2.1",
        "date": pd.Timestamp.now().isoformat(),
        "regulatory_framework": ["ECOA", "EU AI Act (High-Risk)", "EEOC Four-Fifths Rule"],
        "findings": [],
        "overall_assessment": None,
        "required_actions": [],
        "summary_metrics": {}
    }

    critical_failures = 0

    # ---------- 1) Group fairness metrics ----------
    print("\n--- FAIRNESS METRICS ASSESSMENT ---")
    for attr, metrics in metrics_by_group.items():
        dir_val = metrics.get("disparate_impact_ratio", np.nan)
        four_fifths_pass = bool(metrics.get("four_fifths_pass", False)) if not pd.isna(dir_val) else False

        report["summary_metrics"][attr] = {
            "disparate_impact_ratio": dir_val,
            "four_fifths_pass": four_fifths_pass,
            "statistical_parity_diff": metrics.get("statistical_parity_diff", np.nan),
            "equal_opportunity_diff": metrics.get("equal_opportunity_diff", np.nan),
        }

        if (not pd.isna(dir_val)) and (not four_fifths_pass):
            critical_failures += 1
            report["findings"].append({
                "attribute": attr,
                "severity": "CRITICAL",
                "finding": f"Disparate impact detected: DIR = {dir_val:.3f} (below 0.80 threshold)",
                "action": "Investigate proxy variables and consider fairness-constrained retraining",
            })
        elif not pd.isna(dir_val):
            report["findings"].append({
                "attribute": attr,
                "severity": "PASS",
                "finding": f"Four-fifths rule satisfied: DIR = {dir_val:.3f}",
                "action": "None required",
            })
        else:
            report["findings"].append({
                "attribute": attr,
                "severity": "INFO",
                "finding": "DIR could not be computed (insufficient groups or zero approvals).",
                "action": "Validate data slices and rerun fairness checks.",
            })

    # ---------- 2) Proxy variables ----------
    print("\n--- PROXY VARIABLE ASSESSMENT ---")
    all_proxies_found = []

    for key, df in proxy_results_by_group.items():
        if df is None or not hasattr(df, "columns") or "is_proxy" not in df.columns:
            continue

        n_proxies = int(df["is_proxy"].sum())
        if n_proxies > 0:
            proxy_names = df.loc[df["is_proxy"], "feature"].astype(str).tolist()
            all_proxies_found.extend(proxy_names)
            report["findings"].append({
                "attribute": f"Proxy Variables ({key})",
                "severity": "WARNING",
                "finding": f"{n_proxies} proxy variables detected: {', '.join(proxy_names)}",
                "action": "Review business justification for each proxy feature and consider mitigation",
            })

    if not all_proxies_found:
        report["findings"].append({
            "attribute": "Proxy Variables",
            "severity": "INFO",
            "finding": "No high-risk proxy variables detected.",
            "action": "Monitor for emerging proxy relationships",
        })

    # ---------- 3) Counterfactual fairness ----------
    print("\n--- COUNTERFACTUAL FAIRNESS ASSESSMENT ---")
    for attr, res in counterfactual_results.items():
        flipped_rate = res.get("flipped_rate", np.nan)

        if pd.isna(flipped_rate):
            report["findings"].append({
                "attribute": f"Counterfactual ({attr})",
                "severity": "INFO",
                "finding": f"Counterfactual test skipped for {attr} (not applicable or not binary).",
                "action": "N/A",
            })
        elif flipped_rate > 0.05:
            report["findings"].append({
                "attribute": f"Counterfactual ({attr})",
                "severity": "WARNING",
                "finding": f"{flipped_rate:.2%} of predictions flipped when proxies were counterfactually adjusted.",
                "action": "Investigate individual-level discrimination; consider mitigation/retraining",
            })
        else:
            report["findings"].append({
                "attribute": f"Counterfactual ({attr})",
                "severity": "PASS",
                "finding": f"{flipped_rate:.2%} of predictions flipped (acceptable level).",
                "action": "None required",
            })

    # ---------- 4) Overall assessment ----------
    if critical_failures == 0:
        report["overall_assessment"] = "PASS"
        report["required_actions"].append("Continue monitoring model fairness and performance.")
        if all_proxies_found:
            report["overall_assessment"] = "CONDITIONAL"
            report["required_actions"].append("Review and document business justification for proxy variables.")
            report["required_actions"].append("Conduct adverse action analysis for declined applicants.")
    elif critical_failures <= 1:
        report["overall_assessment"] = "CONDITIONAL"
        report["required_actions"].append("Implement fairness constraints for the flagged attribute.")
        report["required_actions"].append("Review and document business justification for proxy variables.")
        report["required_actions"].append("Conduct adverse action analysis for declined applicants.")
        report["required_actions"].append("Retest after mitigation within 90 days.")
    else:
        report["overall_assessment"] = "FAIL"
        report["required_actions"].append("Suspend model for the affected use case.")
        report["required_actions"].append("Retrain with fairness constraints.")
        report["required_actions"].append("Full revalidation required before redeployment.")
        report["required_actions"].append("Conduct adverse action analysis for all declined applicants.")

    # ---------- 5) Trade-off results ----------
    print("\n--- ACCURACY-FAIRNESS TRADE-OFF ---")

    base_auc = tradeoff_results.get("base_auc", np.nan)
    base_dir = tradeoff_results.get("base_dir", np.nan)
    fair_acc = tradeoff_results.get("fair_acc", np.nan)
    base_acc = tradeoff_results.get("base_acc", np.nan)
    acc_cost = tradeoff_results.get("acc_cost_of_fairness", np.nan)
    fair_dir = tradeoff_results.get("fair_dir", np.nan)

    finding_text = (
        f"Unconstrained AUC: {base_auc:.4f}, Accuracy: {base_acc:.4f}, DIR: {base_dir:.3f}. "
        f"Fairness-constrained Accuracy: {fair_acc:.4f}, DIR: {fair_dir:.3f}. "
        f"Accuracy Cost of Fairness: {acc_cost:+.4f}."
    )

    report["findings"].append({
        "attribute": "Accuracy-Fairness Trade-off",
        "severity": "INFO",
        "finding": finding_text,
        "action": "Policy decision required on acceptable trade-off based on legal and ethical guidance.",
    })

    # ---------- Print report ----------
    print("\n" + "=" * 60)
    print(f"FAIRNESS AUDIT REPORT: {report['title']}")
    print("=" * 60)
    print(f"Model: {report['model']}")
    print(f"Date: {report['date']}")
    print(f"Regulatory Framework: {', '.join(report['regulatory_framework'])}")
    print("\nOVERALL ASSESSMENT:")
    print(f"  -> {report['overall_assessment']}")

    print("\nFINDINGS:")
    for finding in report["findings"]:
        print(f"\n[{finding['severity']}] {finding['attribute']}:")
        print(f"  {finding['finding']}")
        if finding.get("action") and finding["action"] != "None required":
            print(f"  Action: {finding['action']}")

    if report["required_actions"]:
        print("\nREQUIRED ACTIONS:")
        for action in report["required_actions"]:
            print(f"  - {action}")

    print("\nSign-off:")
    print(" Fair Lending Officer: __________________________ Date: ________")
    print(" Compliance Director:  __________________________ Date: ________")
    print(" Head of Model Governance (Anya Sharma, CFA): ____ Date: ________")

    return report


def run_fairness_audit(n_samples=10000, test_size=0.3, random_state=42):
    """
    Main function to orchestrate the entire fairness audit process.
    Args:
        n_samples (int): Number of samples for the simulated dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation for reproducibility.
    Returns:
        dict: A dictionary containing the final audit report and generated matplotlib figures.
    """
    print("Starting fairness audit...")

    # 1. Load and Prepare Data, Train Initial Model
    (X_test_scaled, y_test_true, y_pred, y_pred_proba, y_pred_favorable, y_true_favorable,
     model, feature_cols, X_train_scaled, y_train_true, scaler, X_test_original) = \
        load_and_prepare_credit_data(n_samples=n_samples, test_size=test_size, random_state=random_state)

    # 2. Augment with Demographics (for both test and train sets)
    print("\nAugmenting test data with synthetic demographics...")
    X_test_fair = augment_with_demographics(X_test_original.copy(), seed=random_state)
    print("\nAugmenting train data with synthetic demographics (for counterfactual and trade-off tests)...")
    # Need X_train_original to augment, then combine with X_train_scaled for model training (if needed again)
    X_train_original = pd.DataFrame(scaler.inverse_transform(X_train_scaled), columns=feature_cols)
    X_train_fair = augment_with_demographics(X_train_original.copy(), seed=random_state)


    # Define protected attributes for the audit
    protected_attributes = [('Gender', 'gender'), ('Race', 'race_group'), ('Age', 'age_group')]

    # Dictionaries to store results from various audit steps
    metrics_by_group = {}
    proxy_results_by_group = {}
    counterfactual_results = {}
    counterfactual_deltas = {}

    # 3. Compute Fairness Metrics for each protected attribute
    for attr_name, col_name in protected_attributes:
        entry, _, _, _ = compute_fairness_metrics(
            y_true_favorable, y_pred_favorable, y_pred_proba, X_test_fair[col_name], attr_name
        )
        metrics_by_group[attr_name] = entry

    # 4. Plot Approval Rates
    print("\nPlotting approval rates...")
    fig_approval_rates = plot_approval_rates(metrics_by_group, protected_attributes)

    # 5. Detect Proxy Variables for each protected attribute
    for attr_name, col_name in protected_attributes:
        current_proxy_df = detect_proxy_variables(
            model, X_test_scaled, X_test_fair[col_name], feature_cols, scaler_obj=scaler, threshold=0.15
        )
        proxy_results_by_group[attr_name] = current_proxy_df

    # 6. Plot Proxy Risk
    print("\nPlotting proxy risk...")
    fig_proxy_risk = plot_proxy_risk(proxy_results_by_group, protected_attributes)

    # 7. Perform Counterfactual Testing
    for attr_name, col_name in protected_attributes:
        current_proxy_df = proxy_results_by_group.get(attr_name, pd.DataFrame())
        top_proxies = current_proxy_df[current_proxy_df["is_proxy"]]["feature"].tolist() if 'is_proxy' in current_proxy_df.columns else []

        delta, flipped = counterfactual_test(
            model=model,
            X_test_original=X_test_original,
            X_test_scaled=X_test_scaled,
            y_pred_proba_original=y_pred_proba,
            protected_col_series=X_test_fair[col_name],
            protected_col_name=col_name,
            feature_cols=feature_cols,
            proxy_features=top_proxies,
            scaler_obj=scaler,
            X_train_scaled=X_train_scaled,
            X_train_fair=X_train_fair,
            mode="shift",
            threshold=0.5
        )

        if isinstance(delta, np.ndarray):
            counterfactual_results[attr_name] = {"delta_mean": float(delta.mean()), "flipped_rate": float(flipped.mean())}
            counterfactual_deltas[attr_name] = delta
        else:
            counterfactual_results[attr_name] = {"delta_mean": np.nan, "flipped_rate": np.nan}

    # 8. Plot Counterfactual Deltas
    print("\nPlotting counterfactual deltas...")
    fig_counterfactual_deltas = plot_counterfactual_deltas(counterfactual_deltas, protected_attributes)

    # 9. Perform Accuracy-Fairness Trade-off Analysis
    # Ensure y_train_favorable and y_test_favorable are defined as 1=approved
    y_train_favorable_tradeoff = 1 - y_train_true
    y_test_favorable_tradeoff = 1 - y_test_true

    tradeoff_results = accuracy_fairness_tradeoff(
        X_train_scaled=X_train_scaled,
        y_train_favorable=y_train_favorable_tradeoff,
        X_test_scaled=X_test_scaled,
        y_test_favorable=y_test_favorable_tradeoff,
        sensitive_train=X_train_fair["race_group"],
        sensitive_test=X_test_fair["race_group"]
    )

    # 10. Plot Accuracy-Fairness Trade-off
    print("\nPlotting accuracy-fairness trade-off...")
    fig_tradeoff = plot_accuracy_fairness_tradeoff(tradeoff_results)

    # 11. Compile Final Audit Report
    final_audit_report = compile_fairness_report(metrics_by_group, proxy_results_by_group, counterfactual_results, tradeoff_results)

    print("\nFairness audit completed.")

    return {
        "report": final_audit_report,
        "figures": {
            "approval_rates": fig_approval_rates,
            "proxy_risk": fig_proxy_risk,
            "counterfactual_deltas": fig_counterfactual_deltas,
            "tradeoff": fig_tradeoff
        }
    }


if __name__ == "__main__":
    # Example usage:
    audit_output = run_fairness_audit()

    # You can now access the report and figures:
    # print(audit_output["report"])
    # audit_output["figures"]["approval_rates"].show()
    # audit_output["figures"]["proxy_risk"].show()
    # audit_output["figures"]["counterfactual_deltas"].show()
    # audit_output["figures"]["tradeoff"].show()

    # To display all figures generated by the audit:
    for fig_name, fig_obj in audit_output['figures'].items():
        if fig_obj:
            fig_obj.show()
