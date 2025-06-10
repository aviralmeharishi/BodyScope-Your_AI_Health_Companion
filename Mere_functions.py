

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split



def read_data(path):
    try:
        df = pd.read_csv(path)
    
        return df
    except FileNotFoundError as e:
        print('Location nahi mil rhi h file kiii')
    except Exception as e:
        print(f'Bhai kuch to Gadbad hai {e}')
        raise e
    

def x_y_split(data: pd.DataFrame, target_variable: str):
    """
    Splits the dataset into features (X) and target (y).

    Parameters:
    - data            : pd.DataFrame - The input dataset.
    - target_variable : str          - The name of the column to be used as the target.

    Returns:
    - X : pd.DataFrame - DataFrame containing all features except the target.
    - y : pd.Series    - Series containing the target variable.
    """
    try:
        x = data.drop(columns=target_variable)
        y = data[target_variable]
        return x, y
    except Exception as e:
        raise e

    


from sklearn.model_selection import train_test_split

def train_test(data: pd.DataFrame, target_variable: str, test_size=0.2, rs=42):
    """
    Splits the dataset into train and test sets.

    Parameters:
    - data (pd.DataFrame): Input DataFrame
    - target_variable (str): The name of the target column
    - test_size (float): Proportion of test set
    - rs (int): Random seed

    Returns:
    - x_train, x_test, y_train, y_test
    """
    try:
        x = data.drop(columns=[target_variable])
        y = data[target_variable]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True, random_state=rs)
        return x_train, x_test, y_train, y_test
    except Exception as e:
        print("Error during train_test_split:", e)
        raise




def save(data : pd.DataFrame,path) -> tuple:
    try:
        data.to_csv(path, index= False)
        print('Ho Gya :)')
    except Exception as e:
        raise e
    

    

def obj_to_pickle(data):
    try:
        with open(data, 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        raise e
    


def rowcols(data):
    try:
        print(f'The Numbers of Rows in this Data are {data.shape[0]}')
        print(f'The Numbers of Columns in this Data are {data.shape[1]}')
    except Exception as e:
        raise e
    


def minmax(x : pd.Series):
    try:
        min_val = x[0]
        max_val= x[0]

        for i in x:
            if i > max_val:
                max_val = i

        for j in x:
            if j < min_val:
                min_val = j


        print('The Minimum Value in this column',"is going to be",min_val)
        
        print('The Maximum Value in this column',"is going to be",max_val)

    except Exception as e:
        raise e
    

from scipy.stats import shapiro, ttest_ind, mannwhitneyu, levene

def num_num_relevance(df, num_col1, num_col2, alpha=0.05):
    data = df[[num_col1, num_col2]].dropna()
    x = data[num_col1]
    y = data[num_col2]

    # Check normality
    p_x_normal = shapiro(x)[1]
    p_y_normal = shapiro(y)[1]
    normal = p_x_normal > alpha and p_y_normal > alpha

    if normal:
        p_levene = levene(x, y)[1]
        equal_var = p_levene > alpha
        stat, p_val = ttest_ind(x, y, equal_var=equal_var)
        method = "t-test (equal var)" if equal_var else "Welch's t-test (unequal var)"
    else:
        stat, p_val = mannwhitneyu(x, y, alternative='two-sided')
        method = "Mann-Whitney U"

    return {
        "col1": num_col1,
        "col2": num_col2,
        "method": method,
        "p_value": p_val,
        "relevant": p_val < alpha,
        "reason": f"{method}, {'significant' if p_val < alpha else 'not significant'}"
    }




from scipy.stats import shapiro, levene, f_oneway, kruskal

def num_cat_relevance(df, num_col, cat_col, alpha=0.05):
    data = df[[num_col, cat_col]].dropna()
    groups = [data[num_col][data[cat_col] == cat] for cat in data[cat_col].unique()]
    valid_groups = [g for g in groups if len(g) >= 3]

    if len(valid_groups) < 2:
        return {
            "col1": num_col,
            "col2": cat_col,
            "method": "Insufficient data",
            "p_value": None,
            "relevant": False,
            "reason": "Not enough data per group for reliable test"
        }

    normality_results = [shapiro(g)[1] > alpha for g in valid_groups]
    all_normal = all(normality_results)

    if all_normal:
        p_levene = levene(*valid_groups)[1]
        equal_var = p_levene > alpha
        stat, p_val = f_oneway(*valid_groups)
        method = "ANOVA (equal var)" if equal_var else "ANOVA (unequal var — caution)"
    else:
        stat, p_val = kruskal(*valid_groups)
        method = "Kruskal-Wallis"

    return {
        "col1": num_col,
        "col2": cat_col,
        "method": method,
        "p_value": p_val,
        "relevant": p_val < alpha,
        "reason": f"{method}, {'significant' if p_val < alpha else 'not significant'}"
    }


from scipy.stats import chi2_contingency

from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

def cat_cat_relevance(df, col1, col2, alpha=0.05, max_unique_threshold=10):
    """
    Tests association between two categorical variables using Chi-Squared test.
    Converts numeric-like categories (with few unique values) to strings.
    
    Returns a one-row DataFrame with col1, col2, method, p_value, relevant, reason.
    """

    # Convert numeric-looking categoricals to strings
    if pd.api.types.is_numeric_dtype(df[col1]) and df[col1].nunique() <= max_unique_threshold:
        df[col1] = df[col1].astype(str)
    if pd.api.types.is_numeric_dtype(df[col2]) and df[col2].nunique() <= max_unique_threshold:
        df[col2] = df[col2].astype(str)

    data = df[[col1, col2]].dropna()
    contingency = pd.crosstab(data[col1], data[col2])

    # Check for sufficient data
    if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
        result = {
            "col1": col1,
            "col2": col2,
            "method": "Insufficient categories",
            "p_value": None,
            "relevant": False,
            "reason": "Not enough unique category combinations"
        }
        return pd.DataFrame([result])

    # Perform chi2 test
    chi2, p_val, _, _ = chi2_contingency(contingency)
    result = {
        "col1": col1,
        "col2": col2,
        "method": "Chi-Squared",
        "p_value": p_val,
        "relevant": p_val < alpha,
        "reason": f"Chi-Squared test, {'significant' if p_val < alpha else 'not significant'}"
    }
    return result



from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Global containers
m, acc, pre, rec, f1, kap = [], [], [], [], [], []

# Scoreline defined once
try:
    scoreline
except NameError:
    scoreline = pd.DataFrame(columns=[
        'MODEL', 'ACCURACY', 'PRECISION', 'RECALL', 'F1 SCORE', 'COHEN KAPPA SCORE'
    ])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

# --- Define these lists globally, outside the function ---
m = []
acc = []
pre = []
rec = []
f1 = []
kap = []
scoreline = pd.DataFrame() # Initialize scoreline globally as well

def multiclass_model_validation(model, xtrain, ytrain, xtest, ytest):
    """
    Trains and evaluates a multi-class classification model.
    Prompts user to save performance scores.
    """
    global scoreline # Declare global for modification
    global m, acc, pre, rec, f1, kap # Declare other lists as global

    print(f"\n🧠 Training: {model.__class__.__name__}")
    model.fit(xtrain, ytrain)

    # Handle models without predict_proba
    try:
        soft_pred = model.predict_proba(xtest)
    except AttributeError:
        soft_pred = None
        print("⚠️ This model does not support probability prediction. Skipping ROC-AUC.")

    hard_pred = model.predict(xtest)

    print('\n📊 Confusion Matrix:\n', confusion_matrix(ytest, hard_pred))
    print('🧮 Cohen Kappa Score:', round(cohen_kappa_score(ytest, hard_pred), 2))
    print('📋 Classification Report:\n', classification_report(ytest, hard_pred))

    if soft_pred is not None:
        try:
            classes = np.unique(ytest)
            ytest_bin = label_binarize(ytest, classes=classes)

            if ytest_bin.shape[1] > 1:
                roc_auc = roc_auc_score(ytest_bin, soft_pred, average='macro', multi_class='ovr')
                print(f'📈 Macro ROC-AUC Score: {roc_auc:.4f}')

                plt.figure(figsize=(8, 6))
                for i in range(len(classes)):
                    fpr, tpr, _ = roc_curve(ytest_bin[:, i], soft_pred[:, i])
                    plt.plot(fpr, tpr, label=f'Class {classes[i]}')
                plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curves (One-vs-Rest)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                print("⚠️ ROC-AUC can't be plotted for a single class.")
        except Exception as e:
            print(f"❌ ROC-AUC plot error: {e}")

    # ✅ Ask before saving
    response = input("📌 Do you want to save this model's scores? (y/n): ")
    if response.strip().lower() == 'y':
        m.append(str(model))
        acc.append(accuracy_score(ytest, hard_pred))
        pre.append(precision_score(ytest, hard_pred, average='macro', zero_division=0)) # Added zero_division
        rec.append(recall_score(ytest, hard_pred, average='macro', zero_division=0)) # Added zero_division
        f1.append(f1_score(ytest, hard_pred, average='macro', zero_division=0)) # Added zero_division
        kap.append(cohen_kappa_score(ytest, hard_pred))

        scoreline = pd.DataFrame({
            'MODEL': m,
            'ACCURACY': acc,
            'PRECISION': pre,
            'RECALL': rec,
            'F1 SCORE': f1,
            'COHEN KAPPA SCORE': kap
        })

        print("✅ Scores saved to `scoreline`.")
    else:
        print("🚫 Scores not saved.")