Here's a comprehensive `README.md` file for your Streamlit application lab project:

---

# QuLab: Lab 40: AI Credit Model Fairness Audit

## Project Title

**QuLab: Lab 40: Fairness Testing of Credit Model**

## Project Description

This Streamlit application, "QuLab: Lab 40," simulates a comprehensive fairness audit of an AI-powered credit scoring model. Designed for a CFA Charterholder persona, Ms. Anya Sharma (Head of Model Governance at GlobalFin Bank), the application provides a step-by-step workflow to identify, quantify, and report on potential biases in a credit decisioning system.

In an era of increasing regulatory scrutiny (e.g., Equal Credit Opportunity Act in the US, EU AI Act) and ethical demands for responsible AI, this lab empowers users to:
*   Understand the nuances of fair lending laws and their implications for AI models.
*   Apply quantitative fairness metrics to detect disparate impact.
*   Uncover hidden biases through proxy variable detection using SHAP analysis.
*   Assess individual-level fairness via counterfactual testing.
*   Evaluate the trade-offs between model accuracy and fairness.
*   Compile a structured, regulatory-ready "Fairness Audit Report" to communicate findings to both technical and non-technical stakeholders.

The project emphasizes bridging the gap between technical data science and critical compliance/legal departments, providing practical tools for model governance professionals.

## Features

The application guides users through a comprehensive fairness audit workflow, structured into the following key steps:

1.  **Introduction & Setup**:
    *   Sets the context of the audit.
    *   Loads a simulated credit dataset and a pre-trained `XGBClassifier` model, representing a production credit scoring system.
    *   Displays initial model performance metrics (ROC AUC, Accuracy, Approval Rates).

2.  **Augment Data with Synthetic Demographics**:
    *   Demonstrates how to generate and integrate synthetic demographic attributes (`gender`, `race_group`, `age_group`) into the dataset for analytical purposes, highlighting the disclaimer for real-world scenarios.
    *   Explains the mathematical basis for synthetic demographic correlation.

3.  **Quantify Group Fairness & Regulatory Compliance**:
    *   Computes essential group fairness metrics (Disparate Impact Ratio, Statistical Parity Difference, Equal Opportunity Difference, FPR Parity, Predictive Parity).
    *   Applies the "Four-Fifths Rule" from the EEOC to assess regulatory compliance for disparate impact.
    *   Visualizes approval rates by demographic groups with compliance thresholds.

4.  **Identify Potential Proxy Variables**:
    *   Detects features that might indirectly act as "proxies" for protected attributes, potentially leading to indirect discrimination.
    *   Utilizes SHAP (SHapley Additive exPlanations) values to quantify feature importance.
    *   Calculates a "Proxy Risk Score" based on correlation with protected attributes and SHAP importance.
    *   Visualizes proxy risks using scatter plots.

5.  **Conduct Counterfactual Fairness Tests**:
    *   Assesses individual-level fairness by simulating how an applicant's credit decision would change if only their protected attribute (and potentially related proxy features) were altered, keeping other characteristics constant.
    *   Quantifies mean prediction change and "flipped" decision rates.
    *   Presents results through histograms of prediction changes.

6.  **Analyze Accuracy-Fairness Trade-off**:
    *   Explores the inherent tension between maximizing model accuracy and achieving strict fairness criteria.
    *   Compares the performance (AUC, Disparate Impact Ratio) of the original unconstrained model against a fairness-constrained model (using `fairlearn.reductions.ExponentiatedGradient`).
    *   Quantifies the "cost of fairness" in terms of reduced accuracy.
    *   Illustrates the trade-off and the "Impossibility Theorem" with a Pareto curve.

7.  **Compile Regulatory-Ready Fairness Audit Report**:
    *   Synthesizes all findings from the previous steps into a structured, comprehensive "Fairness Audit Report."
    *   Provides an overall assessment (PASS/CONDITIONAL/FAIL) and lists specific findings and recommended actions.
    *   Designed for clear communication to regulatory bodies, compliance teams, and senior management.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have Python installed (version 3.8+ recommended).

You will need the following Python libraries:
*   `streamlit`
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `xgboost`
*   `fairlearn`
*   `shap`

### Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```
    *(Note: Replace `<repository_url>` and `<project_directory>` with actual values if hosted in a Git repository.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file in your project root with the following contents:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    xgboost
    fairlearn
    shap
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Ensure `app.py` and `source.py` are in the same directory.** The `source.py` file contains all the backend logic and helper functions (`load_and_prepare_credit_data`, `augment_with_demographics`, `compute_fairness_metrics`, `detect_proxy_variables`, `counterfactual_test`, `accuracy_fairness_tradeoff`, `compile_fairness_report`).

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

3.  **Access the application:**
    Your default web browser should automatically open to the Streamlit application (usually at `http://localhost:8501`).

4.  **Navigate and Interact:**
    *   Use the sidebar to navigate through the different workflow steps.
    *   Click the buttons on each page to trigger the respective analyses and view the results.
    *   The application maintains state across pages, allowing you to proceed sequentially through the fairness audit.

## Project Structure

```
.
├── app.py                  # Main Streamlit application file
├── source.py               # Contains all backend logic and helper functions for fairness analysis
├── requirements.txt        # List of Python dependencies
└── README.md               # This README file
```

## Technology Stack

*   **Application Framework**: [Streamlit](https://streamlit.io/)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Machine Learning**: [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/en/stable/)
*   **Fairness Toolkits**: [Fairlearn](https://fairlearn.org/)
*   **Explainable AI (XAI)**: [SHAP](https://shap.readthedocs.io/en/latest/)
*   **Data Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
*   **Python Version**: 3.8+

## Contributing

This project is primarily intended as a lab exercise for educational purposes. Contributions are welcome for improvements, bug fixes, or additional features that enhance the learning experience. Please feel free to open issues or submit pull requests.

## License

This project is provided for educational purposes as part of the QuantUniversity QuLab series. It is not intended for commercial use without explicit permission.

## Contact

For questions or inquiries related to this lab or QuantUniversity's programs, please visit:
*   [QuantUniversity Website](https://www.quantuniversity.com/)
*   [QuLab Information](https://www.quantuniversity.com/quantic-labs/)

---