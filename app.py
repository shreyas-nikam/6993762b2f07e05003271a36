import streamlit as st
from source import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from fairlearn.reductions import DemographicParity
from sklearn.metrics import roc_auc_score, confusion_matrix

st.set_page_config(page_title="QuLab: Lab 40: Fairness Testing of Credit Model", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 40: Fairness Testing of Credit Model")
st.divider()

# Initialize session state variables
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Introduction & Setup'
if 'X_test_base' not in st.session_state:
    st.session_state.X_test_base = None
    st.session_state.y_test = None
    st.session_state.y_pred = None
    st.session_state.y_pred_proba = None
    st.session_state.y_pred_favorable = None
    st.session_state.y_true_favorable = None
    st.session_state.model = None
    st.session_state.feature_cols = None
    st.session_state.X_train_scaled = None
    st.session_state.y_train_true = None
    st.session_state.scaler = None
    st.session_state.X_test_original = None

if 'X_test_fair' not in st.session_state:
    st.session_state.X_test_fair = None
    st.session_state.X_train_fair = None 

if 'metrics_by_group' not in st.session_state:
    st.session_state.metrics_by_group = {}
    st.session_state.advantaged_groups = {}
    st.session_state.disadvantaged_groups = {}
    st.session_state.protected_attributes = [('Gender', 'gender'), ('Race', 'race_group'), ('Age', 'age_group')]

if 'proxy_results_by_group' not in st.session_state:
    st.session_state.proxy_results_by_group = {}

if 'counterfactual_results' not in st.session_state:
    st.session_state.counterfactual_results = {}
    st.session_state.counterfactual_deltas = {}

if 'tradeoff_results' not in st.session_state:
    st.session_state.tradeoff_results = {}

if 'final_audit_report' not in st.session_state:
    st.session_state.final_audit_report = None

# Sidebar Navigation
st.sidebar.title("AI Credit Model Fairness Audit")
st.sidebar.markdown(f"**Ms. Anya Sharma, CFA, Head of Model Governance**")

page_selection = st.sidebar.selectbox(
    "Navigate Workflow Steps",
    [
        'Introduction & Setup',
        '2. Augment Data with Synthetic Demographics',
        '3. Quantify Group Fairness & Regulatory Compliance',
        '4. Identify Potential Proxy Variables',
        '5. Conduct Counterfactual Fairness Tests',
        '6. Analyze Accuracy-Fairness Trade-off',
        '7. Compile Regulatory-Ready Fairness Audit Report'
    ],
    key='page_selector'
)
st.session_state.current_page = page_selection

# Page: Introduction & Setup
if st.session_state.current_page == 'Introduction & Setup':
    st.title("AI Credit Model Fairness Audit: A CFA Charterholder's Workflow")
    st.header("Case Study: Detecting and Mitigating Bias in Credit Decisions at GlobalFin Bank")
    st.markdown(f"")
    st.markdown(f"As Ms. Anya Sharma, CFA, Head of Model Governance at GlobalFin Bank, you are tasked with ensuring that the bank's AI-powered credit scoring model operates not only efficiently but also ethically and in full compliance with evolving regulations. Recent scrutiny, particularly from the Equal Credit Opportunity Act (ECOA) in the US and the forthcoming EU AI Act, demands robust bias testing and documentation for high-risk AI systems like credit models.")
    st.markdown(f"")
    st.markdown(f"Your primary objective is to conduct a comprehensive fairness audit of GlobalFin Bank's existing credit model. This involves identifying potential disparate impact against protected demographic groups, detecting subtle proxy variables, understanding individual-level discrimination through counterfactual analysis, and quantifying the trade-offs between model accuracy and fairness. The ultimate goal is to produce a regulatory-ready 'Fairness Audit Report' that clearly articulates findings, compliance status, and recommended actions, bridging the gap between technical data science teams and compliance/legal departments.")
    st.markdown(f"")
    st.markdown(f"This application will guide you through a step-by-step workflow that simulates a real-world fairness audit, empowering you with the tools and insights to uphold GlobalFin Bank's commitment to fair and responsible AI.")
    st.markdown(f"---")
    st.header("1. Environment Setup and Initial Model Loading")
    st.markdown(f"As a CFA Charterholder focused on model governance, the first step is to prepare your analytical environment. This includes installing necessary libraries and loading the pre-existing credit model and its associated data. For this exercise, we will simulate a credit dataset and a pre-trained model that Ms. Sharma would receive from the data science team.")
    st.markdown(f"")

    if st.button("Load Initial Model and Data"):
        X_test_base, y_test, y_pred, y_pred_proba, y_pred_favorable, y_true_favorable, model, feature_cols, X_train_scaled, y_train_true, scaler = load_and_prepare_credit_data()
        
        # Ensure we use the original (unscaled) X_test for synthetic demographic generation
        X_test_original = scaler.inverse_transform(X_test_base)
        X_test_original = pd.DataFrame(X_test_original, columns=feature_cols)

        # Store in session state
        st.session_state.X_test_base = X_test_base
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.y_pred_proba = y_pred_proba
        st.session_state.y_pred_favorable = y_pred_favorable
        st.session_state.y_true_favorable = y_true_favorable
        st.session_state.model = model
        st.session_state.feature_cols = feature_cols
        st.session_state.X_train_scaled = X_train_scaled
        st.session_state.y_train_true = y_train_true
        st.session_state.scaler = scaler
        st.session_state.X_test_original = X_test_original

        st.success("Model and data loaded successfully!")
        st.markdown(f"**Initial Model Performance (on simulated test set):**")
        st.write(f"ROC AUC: {roc_auc_score(st.session_state.y_test, st.session_state.y_pred_proba):.4f}")
        tn, fp, fn, tp = confusion_matrix(st.session_state.y_test, st.session_state.y_pred).ravel()
        st.write(f"Accuracy: {(tp+tn)/(tn+fp+fn+tp):.4f}")
        st.write(f"Predicted Approval Rate: {st.session_state.y_pred_favorable.mean():.2%}")
        st.write(f"Actual Approval Rate: {st.session_state.y_true_favorable.mean():.2%}")
        st.markdown(f"")
        st.markdown(f"Ms. Sharma has successfully set up her environment and loaded the simulated credit dataset, which includes features like income, debt ratio, and credit score. A pre-trained `XGBClassifier` model, representing the bank's production credit scoring model, has also been loaded. The model's initial performance, including ROC AUC and predicted approval rates, provides a baseline for the fairness audit. Note that for fairness analysis, we define 'approval' as the favorable outcome, which is `y_pred_favorable` (1 if approved, 0 if denied). The original `y_pred` (1 for default, 0 for no default) is inverted for this purpose.")
        st.markdown(f"")
        st.markdown(f"Sample of initial test data features:")
        st.dataframe(st.session_state.X_test_base.head())
    elif st.session_state.X_test_base is not None:
        st.success("Model and data already loaded.")
        st.markdown(f"**Initial Model Performance (on simulated test set):**")
        st.write(f"ROC AUC: {roc_auc_score(st.session_state.y_test, st.session_state.y_pred_proba):.4f}")
        tn, fp, fn, tp = confusion_matrix(st.session_state.y_test, st.session_state.y_pred).ravel()
        st.write(f"Accuracy: {(tp+tn)/(tn+fp+fn+tp):.4f}")
        st.write(f"Predicted Approval Rate: {st.session_state.y_pred_favorable.mean():.2%}")
        st.write(f"Actual Approval Rate: {st.session_state.y_true_favorable.mean():.2%}")
        st.markdown(f"")
        st.markdown(f"Ms. Sharma has successfully set up her environment and loaded the simulated credit dataset, which includes features like income, debt ratio, and credit score. A pre-trained `XGBClassifier` model, representing the bank's production credit scoring model, has also been loaded. The model's initial performance, including ROC AUC and predicted approval rates, provides a baseline for the fairness audit. Note that for fairness analysis, we define 'approval' as the favorable outcome, which is `y_pred_favorable` (1 if approved, 0 if denied). The original `y_pred` (1 for default, 0 for no default) is inverted for this purpose.")
        st.markdown(f"")
        st.markdown(f"Sample of initial test data features:")
        st.dataframe(st.session_state.X_test_base.head())

# Page: 2. Augment Data with Synthetic Demographics
elif st.session_state.current_page == '2. Augment Data with Synthetic Demographics':
    st.header("2. Augmenting Data with Synthetic Demographic Attributes")
    st.markdown(f"GlobalFin Bank, like many financial institutions, does not explicitly collect sensitive demographic data (race, gender, age) for credit decisions. However, for internal fairness auditing and methodology demonstration, Ms. Sharma needs to assess potential bias. She will augment the dataset with *synthetic* demographic attributes, ensuring a clear disclaimer that these are for analytical demonstration only and not for real production systems without legally obtained data.")
    st.markdown(f"")
    st.markdown(f"Ms. Sharma understands the sensitivity around demographic data. While production credit models at GlobalFin Bank intentionally exclude such attributes to avoid direct discrimination, regulatory bodies require robust evidence that no indirect discrimination occurs. To perform a comprehensive fairness audit, she needs to simulate how the model might behave across different demographic groups. For this lab, she will generate synthetic `gender`, `race_group`, and `age_group` attributes, carefully correlating them with existing financial features to mimic real-world proxy relationships (e.g., income with gender, revolving utilization with race, employment length with age).")
    st.markdown(f"")
    st.markdown(f"The mathematical basis for creating these correlations is rooted in simple logistic functions or direct correlations, designed to create discernible (but not perfectly deterministic) relationships for testing purposes. For example, `gender_prob` is linked to `income` using a sigmoid-like function:")
    st.markdown(r"$$ \text{gender\_prob} = \frac{1}{1 + e^{-c \times \text{income\_norm}}} $$")
    st.markdown(r"where $\text{gender\_prob}$ is the probability of a specific gender, $c$ is a constant controlling the correlation strength, and $\text{income\_norm}$ is the normalized income. Similar logic applies to `race_prob` (correlated with `revolving_utilization`) and `age_group` (correlated with `employment_length`). This approach allows Ms. Sharma to test for 'disparate impact' even when protected attributes are not direct model inputs.")
    st.markdown(f"")
    st.warning("⚠️ Disclaimer: The synthetic demographics used here are for methodological demonstration only. In production fairness testing, firms use actual demographic data from sources like HMDA (Home Mortgage Disclosure Act) data or Bayesian Improved Surname Geocoding (BISG), where legally permissible. Using synthetic demographics for real regulatory compliance would be invalid.")

    if st.button("Augment Data with Synthetic Demographics"):
        if st.session_state.X_test_original is not None:
            st.session_state.X_test_fair = augment_with_demographics(st.session_state.X_test_original.copy())
            
            # Also augment training data for consistency in later steps like tradeoff
            X_train_original_unscaled = st.session_state.scaler.inverse_transform(st.session_state.X_train_scaled)
            X_train_original_unscaled_df = pd.DataFrame(X_train_original_unscaled, columns=st.session_state.feature_cols)
            st.session_state.X_train_fair = augment_with_demographics(X_train_original_unscaled_df.copy(), seed=43) # Use different seed for train
            
            st.success("Data augmented with synthetic demographics!")
            st.markdown(f"Sample of augmented data with synthetic demographics:")
            st.dataframe(st.session_state.X_test_fair.head())
            st.markdown(f"")
            st.markdown(f"The `augment_with_demographics` function has successfully added synthetic `gender`, `race_group`, and `age_group` attributes to the credit dataset (`X_test_fair`). Ms. Sharma now has the necessary (synthetic) attributes to proceed with fairness testing, acknowledging that in a real audit, legally and ethically sourced demographic data (e.g., HMDA, BISG) would be used. This step is crucial for methodological demonstration, allowing her to detect potential biases that might otherwise remain hidden.")
        else:
            st.error("Please load the initial model and data first on the 'Introduction & Setup' page.")
    elif st.session_state.X_test_fair is not None:
        st.success("Data already augmented with synthetic demographics.")
        st.markdown(f"Sample of augmented data with synthetic demographics:")
        st.dataframe(st.session_state.X_test_fair.head())
        st.markdown(f"")
        st.markdown(f"The `augment_with_demographics` function has successfully added synthetic `gender`, `race_group`, and `age_group` attributes to the credit dataset (`X_test_fair`). Ms. Sharma now has the necessary (synthetic) attributes to proceed with fairness testing, acknowledging that in a real audit, legally and ethically sourced demographic data (e.g., HMDA, BISG) would be used. This step is crucial for methodological demonstration, allowing her to detect potential biases that might otherwise remain hidden.")

# Page: 3. Quantify Group Fairness & Regulatory Compliance
elif st.session_state.current_page == '3. Quantify Group Fairness & Regulatory Compliance':
    st.header("3. Quantifying Group Fairness and Regulatory Compliance (Four-Fifths Rule)")
    st.markdown(f"Ms. Sharma's core responsibility is to ensure the credit model complies with fair lending laws. This requires quantifying how the model's approval rates differ across various demographic groups and applying regulatory thresholds like the 'four-fifths rule' to identify disparate impact.")
    st.markdown(f"")
    st.markdown(f"As a CFA Charterholder, Ms. Sharma knows that 'fair lending is not optional—it is the law.' Disparate impact occurs when a neutral policy or model disproportionately affects a protected group, even without explicit discriminatory intent. The Equal Credit Opportunity Act (ECOA) prohibits such discrimination. The Equal Employment Opportunity Commission (EEOC) 'four-fifths rule' is a widely recognized regulatory threshold: if the selection rate for a protected group is less than 80% (or four-fifths) of the selection rate for the most favored group, it generally indicates disparate impact.")
    st.markdown(f"")
    st.markdown(f"Ms. Sharma needs to compute several key fairness metrics:")
    st.markdown(f"1. **Disparate Impact Ratio (DIR)**: Measures the ratio of the favorable outcome rate (e.g., approval rate) for the disadvantaged group to the favorable outcome rate for the advantaged group.")
    st.markdown(r"$$ \text{DIR} = \frac{P(\hat{Y} = \text{approve} \mid G = \text{disadvantaged})}{P(\hat{Y} = \text{approve} \mid G = \text{advantaged})} $$")
    st.markdown(r"where $P(\hat{Y} = \text{approve} \mid G)$ is the probability of approval given group $G$.")
    st.markdown(f"The **Four-fifths rule** states that $\text{DIR} \ge 0.80$ is required for compliance. Below 0.80 constitutes prima facie evidence of disparate impact.")
    st.markdown(f"2. **Statistical Parity Difference (SPD)**: The absolute difference in favorable outcome rates between groups. Ideally, this should be close to zero.")
    st.markdown(r"$$ \text{SPD} = |P(\hat{Y} = 1 \mid G = A) - P(\hat{Y} = 1 \mid G = B)| $$")
    st.markdown(r"Where $\hat{Y} = 1$ is the favorable outcome (approval), and $G=A$ and $G=B$ represent two different demographic groups.")
    st.markdown(f"3. **Equal Opportunity Difference (EOD)**: Measures the absolute difference in False Negative Rates (FNR) between groups. A high FNR for a disadvantaged group means truly creditworthy applicants are unfairly denied.")
    st.markdown(r"$$ \text{EOD} = |\text{FNR}_A - \text{FNR}_B| $$")
    st.markdown(r"where $\text{FNR}_A$ and $\text{FNR}_B$ are the False Negative Rates for groups A and B respectively.")
    st.markdown(f"4. **False Positive Rate (FPR) Parity**: Measures the absolute difference in False Positive Rates between groups. A high FPR for a disadvantaged group means truly uncreditworthy applicants are unfairly approved (less common concern in credit, but relevant for other domains).")
    st.markdown(r"$$ \text{FPR Parity} = |\text{FPR}_A - \text{FPR}_B| $$")
    st.markdown(r"where $\text{FPR}_A$ and $\text{FPR}_B$ are the False Positive Rates for groups A and B respectively.")
    st.markdown(f"5. **Predictive Parity**: Measures the absolute difference in Positive Predictive Values (PPV) between groups. If PPV differs, it means that among those predicted as creditworthy, the actual default rates differ across groups.")
    st.markdown(r"$$ \text{Predictive Parity} = |\text{PPV}_A - \text{PPV}_B| $$")
    st.markdown(r"where $\text{PPV}_A$ and $\text{PPV}_B$ are the Positive Predictive Values for groups A and B respectively.")
    st.markdown(f"These metrics allow Ms. Sharma to quantitatively assess the model's fairness across different dimensions.")

    if st.session_state.X_test_fair is not None:
        if st.button("Compute Fairness Metrics for All Protected Attributes"):
            for attr_name, col_name in st.session_state.protected_attributes:
                entry, group_results, adv_g, dis_g = compute_fairness_metrics(
                    st.session_state.y_true_favorable, st.session_state.y_pred_favorable, st.session_state.y_pred_proba,
                    st.session_state.X_test_fair[col_name], attr_name
                )
                st.session_state.metrics_by_group[attr_name] = entry
                st.session_state.advantaged_groups[attr_name] = adv_g
                st.session_state.disadvantaged_groups[attr_name] = dis_g
            st.success("Fairness metrics computed for all protected attributes!")
            st.markdown(f"")
            st.markdown(f"Ms. Sharma has executed the `compute_fairness_metrics` function across all three synthetic protected attributes (`gender`, `race_group`, and `age_group`). The output details the approval rates, False Negative Rates (FNR), False Positive Rates (FPR), and other key metrics for each group. Crucially, the Disparate Impact Ratio (DIR) is calculated, and its compliance with the 'four-fifths rule' is immediately assessed (PASS/FAIL).")
            st.markdown(f"")
            st.markdown(f"**Group Fairness Metrics Table:**")
            metrics_table_data = []
            for attr, data in st.session_state.metrics_by_group.items():
                metrics_table_data.append({
                    'Protected Attribute': attr,
                    'Disparate Impact Ratio': data['disparate_impact_ratio'],
                    'Four-Fifths Rule': 'PASS' if data['four_fifths_pass'] else 'FAIL',
                    'Statistical Parity Difference': data['statistical_parity_diff'],
                    'Equal Opportunity Difference': data['equal_opportunity_diff'],
                    'FPR Parity Difference': data['fpr_parity_diff'],
                    'Predictive Parity Difference': data['predictive_parity_diff'],
                    'Fairlearn Equalized Odds Difference': data['equalized_odds_diff_fairlearn']
                })
            st.dataframe(pd.DataFrame(metrics_table_data).set_index('Protected Attribute'))

            st.markdown(f"")
            st.markdown(f"**Approval Rate Bar Chart:**")
            fig, axes = plt.subplots(1, len(st.session_state.protected_attributes), figsize=(18, 6), sharey=True)
            if len(st.session_state.protected_attributes) == 1:
                axes = [axes]

            for i, (attr_name, _) in enumerate(st.session_state.protected_attributes):
                ax = axes[i]
                details = st.session_state.metrics_by_group[attr_name]['group_details']
                groups = list(details.keys())
                approval_rates = [details[g]['approval_rate'] for g in groups]

                sns.barplot(x=groups, y=approval_rates, ax=ax, palette='viridis')
                ax.set_title(f'Approval Rate by {attr_name}')
                ax.set_ylabel('Approval Rate')
                ax.set_ylim(0, max(approval_rates) * 1.2)

                # Add Four-fifths rule line
                advantaged_rate = details[st.session_state.metrics_by_group[attr_name]['advantaged_group']]['approval_rate']
                four_fifths_threshold = advantaged_rate * 0.80
                ax.axhline(four_fifths_threshold, color='red', linestyle='--', label='4/5ths Rule Threshold')
                ax.legend()
                ax.text(0.5, 0.95, f"DIR: {st.session_state.metrics_by_group[attr_name]['disparate_impact_ratio']:.3f}",
                        horizontalalignment='center', verticalalignment='top', transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="red", lw=1, alpha=0.6))
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(f"")
            st.markdown(f"The visualizations provide a clear, intuitive understanding of the approval rate disparities. Ms. Sharma can see at a glance which demographic groups have lower approval rates and whether these fall below the critical 80% threshold. For instance, if 'Group B' for race has an approval rate significantly below 80% of 'Group A's rate, it signals a potential disparate impact violation requiring further investigation. This hands-on analysis allows her to identify compliance risks directly.")
        elif st.session_state.metrics_by_group:
            st.success("Fairness metrics already computed.")
            st.markdown(f"")
            st.markdown(f"Ms. Sharma has executed the `compute_fairness_metrics` function across all three synthetic protected attributes (`gender`, `race_group`, and `age_group`). The output details the approval rates, False Negative Rates (FNR), False Positive Rates (FPR), and other key metrics for each group. Crucially, the Disparate Impact Ratio (DIR) is calculated, and its compliance with the 'four-fifths rule' is immediately assessed (PASS/FAIL).")
            st.markdown(f"")
            st.markdown(f"**Group Fairness Metrics Table:**")
            metrics_table_data = []
            for attr, data in st.session_state.metrics_by_group.items():
                metrics_table_data.append({
                    'Protected Attribute': attr,
                    'Disparate Impact Ratio': data['disparate_impact_ratio'],
                    'Four-Fifths Rule': 'PASS' if data['four_fifths_pass'] else 'FAIL',
                    'Statistical Parity Difference': data['statistical_parity_diff'],
                    'Equal Opportunity Difference': data['equal_opportunity_diff'],
                    'FPR Parity Difference': data['fpr_parity_diff'],
                    'Predictive Parity Difference': data['predictive_parity_diff'],
                    'Fairlearn Equalized Odds Difference': data['equalized_odds_diff_fairlearn']
                })
            st.dataframe(pd.DataFrame(metrics_table_data).set_index('Protected Attribute'))

            st.markdown(f"")
            st.markdown(f"**Approval Rate Bar Chart:**")
            fig, axes = plt.subplots(1, len(st.session_state.protected_attributes), figsize=(18, 6), sharey=True)
            if len(st.session_state.protected_attributes) == 1:
                axes = [axes]

            for i, (attr_name, _) in enumerate(st.session_state.protected_attributes):
                ax = axes[i]
                details = st.session_state.metrics_by_group[attr_name]['group_details']
                groups = list(details.keys())
                approval_rates = [details[g]['approval_rate'] for g in groups]

                sns.barplot(x=groups, y=approval_rates, ax=ax, palette='viridis')
                ax.set_title(f'Approval Rate by {attr_name}')
                ax.set_ylabel('Approval Rate')
                ax.set_ylim(0, max(approval_rates) * 1.2)

                advantaged_rate = details[st.session_state.metrics_by_group[attr_name]['advantaged_group']]['approval_rate']
                four_fifths_threshold = advantaged_rate * 0.80
                ax.axhline(four_fifths_threshold, color='red', linestyle='--', label='4/5ths Rule Threshold')
                ax.legend()
                ax.text(0.5, 0.95, f"DIR: {st.session_state.metrics_by_group[attr_name]['disparate_impact_ratio']:.3f}",
                        horizontalalignment='center', verticalalignment='top', transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="red", lw=1, alpha=0.6))
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown(f"")
            st.markdown(f"The visualizations provide a clear, intuitive understanding of the approval rate disparities. Ms. Sharma can see at a glance which demographic groups have lower approval rates and whether these fall below the critical 80% threshold. For instance, if 'Group B' for race has an approval rate significantly below 80% of 'Group A's rate, it signals a potential disparate impact violation requiring further investigation. This hands-on analysis allows her to identify compliance risks directly.")
    else:
        st.error("Please ensure data is augmented with synthetic demographics on the previous page ('2. Augment Data with Synthetic Demographics').")

# Page: 4. Identify Potential Proxy Variables
elif st.session_state.current_page == '4. Identify Potential Proxy Variables':
    st.header("4. Identifying Potential Proxy Variables via SHAP Analysis")
    st.markdown(f"Even if direct demographic information is excluded, other features in the model might inadvertently act as 'proxies' for protected attributes, leading to indirect discrimination. Ms. Sharma needs to identify these potential proxies, which are features that are both correlated with a protected attribute and highly important to the model's decision-making.")
    st.markdown(f"")
    st.markdown(f"The challenge of proxy variables is one of the hardest problems in fair lending. A feature like 'revolving_utilization' (how much of available credit is being used) is a legitimate business factor for creditworthiness. However, if 'revolving_utilization' is also highly correlated with a protected attribute (e.g., 'race_group' due to historical lending patterns or access to financial services in certain neighborhoods), and if the model relies heavily on this feature, it can become a proxy, perpetuating historical biases.")
    st.markdown(f"")
    st.markdown(f"Ms. Sharma must identify features that exhibit a 'dual condition':")
    st.markdown(f"1. **Significant Correlation**: The feature is statistically correlated with a protected attribute.")
    st.markdown(f"2. **High Model Importance**: The feature has a high impact on the model's predictions (e.g., measured by SHAP values).")
    st.markdown(f"")
    st.markdown(f"The **Proxy Risk Score** quantifies this dual condition:")
    st.markdown(r"$$ \text{Proxy Risk Score} = |\text{Correlation with Protected Attribute}| \times \frac{\text{SHAP Importance}}{\text{Max SHAP Importance}} $$")
    st.markdown(r"where SHAP Importance is the mean absolute SHAP value for a feature. A high proxy risk score indicates a feature that warrants close scrutiny and potential business justification or mitigation. The decision to keep or remove a proxy variable is a policy judgment requiring legal, compliance, and business input, not purely a technical one.")

    if st.session_state.X_test_fair is not None and st.session_state.model is not None:
        selected_attr_proxy = st.selectbox(
            "Select Protected Attribute for Proxy Detection",
            [attr_name for attr_name, _ in st.session_state.protected_attributes],
            key='proxy_attr_select'
        )
        threshold = st.slider(
            "Correlation Threshold to Flag as High-Risk",
            min_value=0.01, max_value=0.5, value=0.15, step=0.01
        )
        
        if st.button(f"Detect Proxy Variables for {selected_attr_proxy}"):
            col_name = next(col for name, col in st.session_state.protected_attributes if name == selected_attr_proxy)
            current_proxy_df = detect_proxy_variables(
                st.session_state.model, st.session_state.X_test_base, st.session_state.X_test_fair[col_name],
                st.session_state.feature_cols, threshold=threshold
            )
            st.session_state.proxy_results_by_group[selected_attr_proxy] = current_proxy_df
            st.success(f"Proxy variables detected for {selected_attr_proxy}!")
            st.markdown(f"")
            st.markdown(f"Ms. Sharma has successfully run the `detect_proxy_variables` function for {selected_attr_proxy}, leveraging SHAP values to quantify model importance and statistical correlation to protected attributes. The tabular output highlights features with high proxy risk scores, indicating strong correlation with a protected attribute and significant model importance.")
            st.markdown(f"")
            st.markdown(f"**Proxy Variable Detection Results for {selected_attr_proxy}:**")
            st.dataframe(current_proxy_df.sort_values('proxy_risk_score', ascending=False).head(10))

            st.markdown(f"")
            st.markdown(f"**Proxy Risk Scatter Plot for {selected_attr_proxy}:**")
            # Use matplotlib plot function directly
            fig, ax = plt.subplots(figsize=(10, 7))
            df = current_proxy_df
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
            ax.set_title(f'Proxy Risk for {selected_attr_proxy}')
            ax.set_xlabel('Correlation with Protected Attribute (Absolute)')
            ax.set_ylabel('Model SHAP Importance (Mean Absolute)')
            ax.axhline(df['shap_importance'].median(), color='gray', linestyle=':', label='Median SHAP Importance')
            ax.axvline(threshold, color='orange', linestyle=':', label=f'Correlation Threshold ({threshold:.2f})')
            ax.legend(title='Is Proxy?', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown(f"")
            st.markdown(f"The scatter plot visually represents this, with potential proxies highlighted in red, clearly showing features that are high on both correlation and SHAP importance axes.")
            st.markdown(f"")
            st.markdown(f"This analysis allows Ms. Sharma to identify features that, while seemingly neutral (e.g., `revolving_utilization`, `employment_length`), might be indirectly driving biased outcomes. This is critical for model governance, as these features require documented business justification or consideration for mitigation strategies to ensure compliance and ethical AI deployment.")
        elif selected_attr_proxy in st.session_state.proxy_results_by_group:
            st.success(f"Proxy variables already detected for {selected_attr_proxy}.")
            current_proxy_df = st.session_state.proxy_results_by_group[selected_attr_proxy]
            st.markdown(f"")
            st.markdown(f"Ms. Sharma has successfully run the `detect_proxy_variables` function for {selected_attr_proxy}, leveraging SHAP values to quantify model importance and statistical correlation to protected attributes. The tabular output highlights features with high proxy risk scores, indicating strong correlation with a protected attribute and significant model importance.")
            st.markdown(f"")
            st.markdown(f"**Proxy Variable Detection Results for {selected_attr_proxy}:**")
            st.dataframe(current_proxy_df.sort_values('proxy_risk_score', ascending=False).head(10))

            st.markdown(f"")
            st.markdown(f"**Proxy Risk Scatter Plot for {selected_attr_proxy}:**")
            fig, ax = plt.subplots(figsize=(10, 7))
            df = current_proxy_df
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
            ax.set_title(f'Proxy Risk for {selected_attr_proxy}')
            ax.set_xlabel('Correlation with Protected Attribute (Absolute)')
            ax.set_ylabel('Model SHAP Importance (Mean Absolute)')
            ax.axhline(df['shap_importance'].median(), color='gray', linestyle=':', label='Median SHAP Importance')
            ax.axvline(threshold, color='orange', linestyle=':', label=f'Correlation Threshold ({threshold:.2f})')
            ax.legend(title='Is Proxy?', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown(f"")
            st.markdown(f"The scatter plot visually represents this, with potential proxies highlighted in red, clearly showing features that are high on both correlation and SHAP importance axes.")
            st.markdown(f"")
            st.markdown(f"This analysis allows Ms. Sharma to identify features that, while seemingly neutral (e.g., `revolving_utilization`, `employment_length`), might be indirectly driving biased outcomes. This is critical for model governance, as these features require documented business justification or consideration for mitigation strategies to ensure compliance and ethical AI deployment.")
    else:
        st.error("Please ensure initial model and augmented data are loaded on previous pages.")

# Page: 5. Conduct Counterfactual Fairness Tests
elif st.session_state.current_page == '5. Conduct Counterfactual Fairness Tests':
    st.header("5. Conducting Counterfactual Fairness Tests")
    st.markdown(f"Ms. Sharma wants to move beyond group-level metrics to understand if the model treats individuals fairly. A key test for this is counterfactual fairness: would an applicant receive a different credit decision if only their protected attribute (and potentially related proxy features) were changed, while keeping all other financial characteristics identical?")
    st.markdown(f"")
    st.markdown(f"Individual fairness is a challenging concept, but counterfactual fairness provides a practical way to test it. Ms. Sharma uses this test to answer a specific question: 'If a male applicant were female, or an 'Over 40' applicant were 'Under 40', would GlobalFin Bank's credit model still approve their loan, assuming all other financial factors remained exactly the same?' Significant changes in prediction outcomes under such counterfactual scenarios would indicate a lack of individual fairness, suggesting the model is sensitive to protected attributes, even if indirectly.")
    st.markdown(f"")
    st.markdown(f"The process involves:")
    st.markdown(f"1. **Creating Counterfactuals**: For each individual in the test set, create a hypothetical copy where only the protected attribute is flipped (e.g., Male becomes Female).")
    st.markdown(f"2. **Adjusting Proxies (Optional but important)**: If proxy variables were identified, these too should be adjusted in the counterfactual to reflect how they *would* change if the protected attribute changed (e.g., if 'race_group' changes, and 'revolving_utilization' is a proxy, then 'revolving_utilization' might also be shifted to match the typical values of the counterfactual group). This accounts for indirect effects.")
    st.markdown(f"3. **Comparing Predictions**: Run both the original and counterfactual individuals through the model and compare their predictions.")
    st.markdown(f"")
    st.markdown(f"A significant `delta` in prediction probability or a `flipped` outcome (e.g., approved becomes denied) indicates a potential individual fairness issue. A model is considered *not* counterfactually fair if predictions change significantly upon alteration of only protected attributes.")

    if st.session_state.X_test_fair is not None and st.session_state.model is not None and st.session_state.proxy_results_by_group:
        selected_attr_cf = st.selectbox(
            "Select Protected Attribute for Counterfactual Test",
            [attr_name for attr_name, _ in st.session_state.protected_attributes],
            key='cf_attr_select'
        )
        
        if st.button(f"Run Counterfactual Test for {selected_attr_cf}"):
            col_name = next(col for name, col in st.session_state.protected_attributes if name == selected_attr_cf)
            
            current_proxy_df = st.session_state.proxy_results_by_group.get(selected_attr_cf, pd.DataFrame())
            top_proxies = current_proxy_df[current_proxy_df['is_proxy']]['feature'].tolist()

            delta, flipped = counterfactual_test(
                st.session_state.model, st.session_state.X_test_original, st.session_state.y_pred_proba,
                st.session_state.X_test_fair[col_name], st.session_state.feature_cols,
                proxy_features=top_proxies
            )
            
            if not isinstance(delta, float): # Check if the test was actually run (not skipped)
                st.session_state.counterfactual_results[selected_attr_cf] = {'delta_mean': delta.mean(), 'flipped_rate': flipped.mean()}
                st.session_state.counterfactual_deltas[selected_attr_cf] = delta
                st.success(f"Counterfactual test completed for {selected_attr_cf}!")

                st.markdown(f"")
                st.markdown(f"Ms. Sharma has performed counterfactual fairness tests for {selected_attr_cf}, taking into account previously identified proxy features. The output provides key metrics: the mean and max prediction changes, and most importantly, the percentage of predictions that 'flipped' (i.e., changed from approval to denial or vice versa). A high flip rate indicates that the model's decisions are unacceptably sensitive to protected attributes, even after accounting for related proxy feature adjustments.")
                st.markdown(f"")
                st.markdown(f"**Counterfactual Test Results for {selected_attr_cf}:**")
                if top_proxies:
                    st.write(f"Proxy features adjusted: {', '.join(top_proxies[:3])}")
                else:
                    st.write("No proxy features adjusted.")
                st.write(f"Mean prediction change: {delta.mean():.4f}")
                st.write(f"Max prediction change: {delta.max():.4f}")
                st.write(f"Predictions flipped: {flipped.mean():.2%}")
                
                st.markdown(f"")
                st.markdown(f"**Counterfactual Delta Histogram for {selected_attr_cf}:**")
                # Use matplotlib plot function directly
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(delta, bins=30, kde=True, ax=ax)
                ax.set_title(f'Counterfactual Prediction Change for {selected_attr_cf}')
                ax.set_xlabel('Absolute Change in Prediction Probability')
                ax.set_ylabel('Frequency')
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown(f"")
                st.markdown(f"The histograms visualize the distribution of these prediction changes, allowing Ms. Sharma to see the magnitude and frequency of the model's sensitivity. If a significant number of applicants would receive a different decision solely based on their demographic group (and associated proxy shifts), it flags a serious individual fairness concern that GlobalFin Bank must address to ensure equitable treatment for all applicants.")
            else:
                st.warning(f"Counterfactual test skipped for {selected_attr_cf} as it is not a binary protected attribute or other conditions not met.")

        elif selected_attr_cf in st.session_state.counterfactual_results:
            st.success(f"Counterfactual test already completed for {selected_attr_cf}.")
            res = st.session_state.counterfactual_results[selected_attr_cf]
            delta = st.session_state.counterfactual_deltas[selected_attr_cf]

            st.markdown(f"")
            st.markdown(f"Ms. Sharma has performed counterfactual fairness tests for {selected_attr_cf}, taking into account previously identified proxy features. The output provides key metrics: the mean and max prediction changes, and most importantly, the percentage of predictions that 'flipped' (i.e., changed from approval to denial or vice versa). A high flip rate indicates that the model's decisions are unacceptably sensitive to protected attributes, even after accounting for related proxy feature adjustments.")
            st.markdown(f"")
            st.markdown(f"**Counterfactual Test Results for {selected_attr_cf}:**")
            current_proxy_df = st.session_state.proxy_results_by_group.get(selected_attr_cf, pd.DataFrame())
            top_proxies = current_proxy_df[current_proxy_df['is_proxy']]['feature'].tolist()
            if top_proxies:
                st.write(f"Proxy features adjusted: {', '.join(top_proxies[:3])}")
            else:
                st.write("No proxy features adjusted.")
            st.write(f"Mean prediction change: {res['delta_mean']:.4f}")
            st.write(f"Predictions flipped: {res['flipped_rate']:.2%}")
            
            st.markdown(f"")
            st.markdown(f"**Counterfactual Delta Histogram for {selected_attr_cf}:**")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(delta, bins=30, kde=True, ax=ax)
            ax.set_title(f'Counterfactual Prediction Change for {selected_attr_cf}')
            ax.set_xlabel('Absolute Change in Prediction Probability')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown(f"")
            st.markdown(f"The histograms visualize the distribution of these prediction changes, allowing Ms. Sharma to see the magnitude and frequency of the model's sensitivity. If a significant number of applicants would receive a different decision solely based on their demographic group (and associated proxy shifts), it flags a serious individual fairness concern that GlobalFin Bank must address to ensure equitable treatment for all applicants.")
    else:
        st.error("Please ensure initial model, augmented data, and proxy detection results are available from previous pages.")

# Page: 6. Analyze Accuracy-Fairness Trade-off
elif st.session_state.current_page == '6. Analyze Accuracy-Fairness Trade-off':
    st.header("6. Analyzing the Accuracy-Fairness Trade-off and Impossibility Theorem")
    st.markdown(f"Ms. Sharma now faces a critical decision point: how much model accuracy is GlobalFin Bank willing to 'sacrifice' to achieve greater fairness? She needs to quantify this trade-off and understand the deeper implications of the 'impossibility theorem' in fair machine learning.")
    st.markdown(f"")
    st.markdown(f"For GlobalFin Bank, maximizing model accuracy (e.g., higher AUC for predicting default) is a business imperative. However, strict adherence to fairness metrics might necessitate a slight reduction in overall accuracy. Ms. Sharma's role is to quantify this `accuracy-fairness trade-off` and communicate its economic and reputational implications to senior management and compliance.")
    st.markdown(f"")
    st.markdown(f"She also needs to articulate the **Impossibility Theorem** (Chouldechova, 2017), which states that when base rates (actual default rates) differ across groups, it's generally mathematically impossible to satisfy all fairness metrics (e.g., statistical parity, equal opportunity, and predictive parity) simultaneously. This means GlobalFin Bank must make a policy decision about which fairness criterion to prioritize, guided by legal counsel and ethical considerations.")
    st.markdown(f"")
    st.markdown(f"To assess the trade-off, Ms. Sharma will compare:")
    st.markdown(f"1. **Unconstrained Model**: The original model, optimized solely for accuracy (e.g., AUC).")
    st.markdown(f"2. **Fairness-Constrained Model**: A model retrained or adjusted to explicitly satisfy a fairness constraint (e.g., Demographic Parity). She will use `fairlearn.reductions.ExponentiatedGradient` for this, a powerful algorithm to find fair models.")
    st.markdown(f"")
    st.markdown(f"The goal is to quantify the 'cost of fairness' in terms of reduced AUC, and weigh it against the far greater financial and reputational costs of a fair lending violation (fines, lawsuits, reputational damage). Often, a 1-3% drop in AUC is an acceptable cost for achieving regulatory compliance and ethical standards.")

    if st.session_state.X_train_scaled is not None and st.session_state.y_train_true is not None and st.session_state.X_test_base is not None and st.session_state.y_test is not None and st.session_state.X_train_fair is not None and st.session_state.X_test_fair is not None:
        selected_attr_tradeoff = st.selectbox(
            "Select Protected Attribute for Trade-off Analysis",
            [attr_name for attr_name, _ in st.session_state.protected_attributes if attr_name == 'Race'], # Often shows most significant disparity
            key='tradeoff_attr_select'
        )
        
        if st.button(f"Analyze Accuracy-Fairness Trade-off for {selected_attr_tradeoff}"):
            col_name = next(col for name, col in st.session_state.protected_attributes if name == selected_attr_tradeoff)
            
            # Ensure y_train_favorable and y_test_favorable_tradeoff are correctly derived
            y_train_favorable = 1 - st.session_state.y_train_true
            y_test_favorable_tradeoff = 1 - st.session_state.y_test

            base_auc, fair_auc, base_dir, fair_dir = accuracy_fairness_tradeoff(
                st.session_state.X_train_scaled, y_train_favorable, st.session_state.X_test_base, y_test_favorable_tradeoff,
                st.session_state.X_train_fair[col_name], st.session_state.X_test_fair[col_name]
            )
            st.session_state.tradeoff_results = {
                'base_auc': base_auc,
                'fair_auc': fair_auc,
                'base_dir': base_dir,
                'fair_dir': fair_dir,
                'cost_of_fairness': base_auc - fair_auc
            }
            st.success(f"Accuracy-Fairness Trade-off analysis completed for {selected_attr_tradeoff}!")

            st.markdown(f"")
            st.markdown(f"Ms. Sharma has performed a critical analysis of the accuracy-fairness trade-off using `fairlearn.reductions.ExponentiatedGradient`. The results clearly show the baseline performance of the unconstrained model (in terms of AUC and DIR) versus the fairness-constrained model. She can observe how much AUC might decrease to satisfy the four-fifths rule for Disparate Impact Ratio.")
            st.markdown(f"")
            st.markdown(f"**Accuracy-Fairness Trade-off Results:**")
            st.write(f"{'Metric':<30s} {'Unconstrained':>15s} {'Fair Model':>15s}")
            st.write("-" * 55)
            st.write(f"{'AUC':<30s} {base_auc:>15.4f} {fair_auc:>15.4f}")
            st.write(f"{'Disparate Impact Ratio':<30s} {base_dir:>15.3f} {fair_dir:>15.3f}")
            st.write(f"{'Four-Fifths Rule':<30s} {'PASS' if base_dir >= 0.80 else 'FAIL':>15s} {'PASS' if fair_dir >= 0.80 else 'FAIL':>15s}")
            st.write(f"{'AUC Cost of Fairness':<30s} {'---':>15s} {base_auc - fair_auc:>+15.4f}")

            st.markdown(f"")
            st.markdown(f"**Accuracy-Fairness Pareto Curve:**")
            # Use matplotlib plot function directly
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.scatter(st.session_state.tradeoff_results['base_dir'], st.session_state.tradeoff_results['base_auc'], color='blue', s=200, label='Unconstrained Model')
            plt.scatter(st.session_state.tradeoff_results['fair_dir'], st.session_state.tradeoff_results['fair_auc'], color='red', s=200, label='Fairness-Constrained Model')
            plt.plot([st.session_state.tradeoff_results['base_dir'], st.session_state.tradeoff_results['fair_dir']],
                     [st.session_state.tradeoff_results['base_auc'], st.session_state.tradeoff_results['fair_auc']],
                     color='gray', linestyle='--', alpha=0.7, label='Trade-off Path')

            plt.axvline(0.80, color='green', linestyle=':', label='4/5ths Rule Threshold (0.80 DIR)')

            plt.title('Accuracy-Fairness Trade-off (AUC vs. Disparate Impact Ratio)')
            plt.xlabel('Disparate Impact Ratio (DIR)')
            plt.ylabel('ROC AUC')
            plt.xlim(0, 1.05)
            plt.ylim(0.5, 1.0)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            st.markdown(f"")
            st.markdown(f"The output provides the concrete numbers needed to present to stakeholders. For example, a 0.02 drop in AUC might be a small price to pay to transform a 'FAIL' on the four-fifths rule into a 'PASS.' The conceptual Pareto curve visually reinforces this trade-off. This quantitative understanding, coupled with the 'impossibility theorem,' empowers Ms. Sharma to lead informed policy discussions at GlobalFin Bank, ensuring that the bank makes deliberate, ethically sound, and legally compliant choices regarding its AI models.")

        elif st.session_state.tradeoff_results:
            st.success(f"Accuracy-Fairness Trade-off analysis already completed for {selected_attr_tradeoff}.")
            st.markdown(f"")
            st.markdown(f"Ms. Sharma has performed a critical analysis of the accuracy-fairness trade-off using `fairlearn.reductions.ExponentiatedGradient`. The results clearly show the baseline performance of the unconstrained model (in terms of AUC and DIR) versus the fairness-constrained model. She can observe how much AUC might decrease to satisfy the four-fifths rule for Disparate Impact Ratio.")
            st.markdown(f"")
            st.markdown(f"**Accuracy-Fairness Trade-off Results:**")
            st.write(f"{'Metric':<30s} {'Unconstrained':>15s} {'Fair Model':>15s}")
            st.write("-" * 55)
            st.write(f"{'AUC':<30s} {st.session_state.tradeoff_results['base_auc'] :>15.4f} {st.session_state.tradeoff_results['fair_auc'] :>15.4f}")
            st.write(f"{'Disparate Impact Ratio':<30s} {st.session_state.tradeoff_results['base_dir'] :>15.3f} {st.session_state.tradeoff_results['fair_dir'] :>15.3f}")
            st.write(f"{'Four-Fifths Rule':<30s} {'PASS' if st.session_state.tradeoff_results['base_dir'] >= 0.80 else 'FAIL':>15s} {'PASS' if st.session_state.tradeoff_results['fair_dir'] >= 0.80 else 'FAIL':>15s}")
            st.write(f"{'AUC Cost of Fairness':<30s} {'--':>15s} {st.session_state.tradeoff_results['cost_of_fairness'] :>+15.4f}")

            st.markdown(f"")
            st.markdown(f"**Accuracy-Fairness Pareto Curve:**")
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.scatter(st.session_state.tradeoff_results['base_dir'], st.session_state.tradeoff_results['base_auc'], color='blue', s=200, label='Unconstrained Model')
            plt.scatter(st.session_state.tradeoff_results['fair_dir'], st.session_state.tradeoff_results['fair_auc'], color='red', s=200, label='Fairness-Constrained Model')
            plt.plot([st.session_state.tradeoff_results['base_dir'], st.session_state.tradeoff_results['fair_dir']],
                     [st.session_state.tradeoff_results['base_auc'], st.session_state.tradeoff_results['fair_auc']],
                     color='gray', linestyle='--', alpha=0.7, label='Trade-off Path')
            plt.axvline(0.80, color='green', linestyle=':', label='4/5ths Rule Threshold (0.80 DIR)')
            plt.title('Accuracy-Fairness Trade-off (AUC vs. Disparate Impact Ratio)')
            plt.xlabel('Disparate Impact Ratio (DIR)')
            plt.ylabel('ROC AUC')
            plt.xlim(0, 1.05)
            plt.ylim(0.5, 1.0)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            st.markdown(f"")
            st.markdown(f"The output provides the concrete numbers needed to present to stakeholders. For example, a 0.02 drop in AUC might be a small price to pay to transform a 'FAIL' on the four-fifths rule into a 'PASS.' The conceptual Pareto curve visually reinforces this trade-off. This quantitative understanding, coupled with the 'impossibility theorem,' empowers Ms. Sharma to lead informed policy discussions at GlobalFin Bank, ensuring that the bank makes deliberate, ethically sound, and legally compliant choices regarding its AI models.")
    else:
        st.error("Please ensure initial model, augmented data, and all previous analysis steps are completed to proceed with the trade-off analysis.")

# Page: 7. Compile Regulatory-Ready Fairness Audit Report
elif st.session_state.current_page == '7. Compile Regulatory-Ready Fairness Audit Report':
    st.header("7. Compiling a Regulatory-Ready Fairness Audit Report")
    st.markdown(f"All the analyses performed by Ms. Sharma—group fairness metrics, proxy variable detection, counterfactual tests, and trade-off analysis—must now be consolidated into a structured, regulatory-ready 'Fairness Audit Report.' This report will serve as the official record of the audit, informing compliance, legal, and risk management teams.")
    st.markdown(f"")
    st.markdown(f"The final step for Ms. Sharma as Head of Model Governance is to synthesize all findings into a clear, actionable report. This 'AI Credit Model Fairness Audit Report' is a critical deliverable for GlobalFin Bank, demonstrating due diligence and providing a roadmap for addressing identified biases. It is designed to be easily digestible by non-technical stakeholders while containing sufficient detail for regulatory review (e.g., under ECOA or EU AI Act provisions for high-risk AI).")
    st.markdown(f"")
    st.markdown(f"The report will summarize:")
    st.markdown(f"- **Group Fairness Metrics**: Compliance status with the four-fifths rule for each protected attribute.")
    st.markdown(f"- **Proxy Variables**: Identification of features acting as proxies, along with their risk scores.")
    st.markdown(f"- **Counterfactual Fairness**: Assessment of individual-level discrimination.")
    st.markdown(f"- **Accuracy-Fairness Trade-off**: Quantitative assessment of the performance impact of fairness constraints.")
    st.markdown(f"- **Overall Assessment**: A concise conclusion (PASS, CONDITIONAL, FAIL) for the model's fairness.")
    st.markdown(f"- **Recommended Actions**: Specific steps for GlobalFin Bank to take, such as investigating proxies, considering fairness-constrained retraining, or conducting adverse action analysis.")
    st.markdown(f"")
    st.markdown(f"This report transforms complex technical analysis into strategic insights, enabling GlobalFin Bank to mitigate legal, financial, and reputational risks associated with algorithmic bias.")

    if st.session_state.metrics_by_group and st.session_state.proxy_results_by_group and st.session_state.counterfactual_results and st.session_state.tradeoff_results:
        if st.button("Generate Fairness Audit Report"):
            st.session_state.final_audit_report = compile_fairness_report(
                st.session_state.metrics_by_group,
                st.session_state.proxy_results_by_group,
                st.session_state.counterfactual_results,
                st.session_state.tradeoff_results
            )
            st.success("Fairness Audit Report Generated!")
            
            # Display the report as formatted markdown
            report = st.session_state.final_audit_report
            st.markdown(f"\n" + "=" * 60)
            st.markdown(f"**FAIRNESS AUDIT REPORT: {report['title']}**")
            st.markdown("=" * 60)
            st.markdown(f"**Model:** {report['model']}")
            st.markdown(f"**Date:** {pd.Timestamp(report['date']).strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Regulatory Framework:** {', '.join(report['regulatory_framework'])}")
            st.markdown(f"\n**OVERALL ASSESSMENT:**")
            st.markdown(f"  -> {report['overall_assessment']}")

            st.markdown(f"\n**FINDINGS:**")
            for finding in report['findings']:
                st.markdown(f"\n**[{finding['severity']}] {finding['attribute']}:**")
                st.markdown(f"  {finding['finding']}")
                if 'action' in finding and finding['action'] != 'None required':
                    st.markdown(f"  Action: {finding['action']}")

            if report['required_actions']:
                st.markdown(f"\n**REQUIRED ACTIONS:**")
                for action in report['required_actions']:
                    st.markdown(f"  - {action}")

            st.markdown(f"\n**Sign-off:**")
            st.markdown(f" Fair Lending Officer: __________________________ Date: ________")
            st.markdown(f" Compliance Director:  __________________________ Date: ________")
            st.markdown(f" Head of Model Governance (Anya Sharma, CFA): ____ Date: ________")
            
            st.markdown(f"")
            st.markdown(f"Ms. Sharma has successfully compiled the comprehensive 'AI Credit Model Fairness Audit Report.' The output presents a structured overview, beginning with a clear overall assessment (PASS, CONDITIONAL, or FAIL) based on the severity of identified biases. Each finding, from disparate impact to proxy variables and counterfactual sensitivity, is detailed with its severity and specific recommended actions.")
            st.markdown(f"")
            st.markdown(f"This report serves as GlobalFin Bank's official record, providing transparent documentation for internal stakeholders and external regulators. For Ms. Sharma, this report is not just a summary of technical results but a strategic document that drives policy decisions, risk mitigation, and ensures GlobalFin Bank's AI systems adhere to the highest ethical and legal standards in fair lending.")
        
        elif st.session_state.final_audit_report is not None:
            st.success("Fairness Audit Report already generated.")
            report = st.session_state.final_audit_report
            st.markdown(f"\n" + "=" * 60)
            st.markdown(f"**FAIRNESS AUDIT REPORT: {report['title']}**")
            st.markdown("=" * 60)
            st.markdown(f"**Model:** {report['model']}")
            st.markdown(f"**Date:** {pd.Timestamp(report['date']).strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Regulatory Framework:** {', '.join(report['regulatory_framework'])}")
            st.markdown(f"\n**OVERALL ASSESSMENT:**")
            st.markdown(f"  -> {report['overall_assessment']}")

            st.markdown(f"\n**FINDINGS:**")
            for finding in report['findings']:
                st.markdown(f"\n**[{finding['severity']}] {finding['attribute']}:**")
                st.markdown(f"  {finding['finding']}")
                if 'action' in finding and finding['action'] != 'None required':
                    st.markdown(f"  Action: {finding['action']}")

            if report['required_actions']:
                st.markdown(f"\n**REQUIRED ACTIONS:**")
                for action in report['required_actions']:
                    st.markdown(f"  - {action}")

            st.markdown(f"\n**Sign-off:**")
            st.markdown(f" Fair Lending Officer: __________________________ Date: ________")
            st.markdown(f" Compliance Director:  __________________________ Date: ________")
            st.markdown(f" Head of Model Governance (Anya Sharma, CFA): ____ Date: ________")

            st.markdown(f"")
            st.markdown(f"Ms. Sharma has successfully compiled the comprehensive 'AI Credit Model Fairness Audit Report.' The output presents a structured overview, beginning with a clear overall assessment (PASS, CONDITIONAL, or FAIL) based on the severity of identified biases. Each finding, from disparate impact to proxy variables and counterfactual sensitivity, is detailed with its severity and specific recommended actions.")
            st.markdown(f"")
            st.markdown(f"This report serves as GlobalFin Bank's official record, providing transparent documentation for internal stakeholders and external regulators. For Ms. Sharma, this report is not just a summary of technical results but a strategic document that drives policy decisions, risk mitigation, and ensures GlobalFin Bank's AI systems adhere to the highest ethical and legal standards in fair lending.")

    else:
        st.error("Please ensure all previous analysis steps (Fairness Metrics, Proxy Variables, Counterfactual Tests, Trade-off Analysis) are completed to generate the final report.")


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
