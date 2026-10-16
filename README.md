# Smart Complaint Classification

> **AI-assisted complaint categorization for faster support routing**  
> A college hackathon: Natural Language Processing (NLP) project built with Python, Scikit-learn, and Streamlit.

---

## 1. Problem Statement
Organizations receive a massive volume of customer feedback, queries, and complaints across diverse channels (support ticketing portals, emails, web forms). Manually reading, sorting, and routing each ticket to the responsible department causes significant turnaround delays, human routing errors, and increased operational costs.

### Why It Matters:
- **Reduces Latency:** Complaints are immediately tagged and forwarded to the right queue.
- **Minimizes Repetitive Work:** Support agents spend time resolving issues instead of sorting tickets.
- **Operational Scalability:** Enables customer service desks to handle spikes in ticket volume during outages or product launches.

### Intended Users:
- Customer Support Representatives
- Operations & Service Desk Managers
- IT Helpdesks and Escalation Teams

---

## 2. Project Objective
Build a lightweight, reliable, demo-ready, and interpretable complaint classification system using classical machine learning on a verified 1,600-record dataset spanning 8 distinct categories.

---

## 3. End-to-End Pipeline Architecture

```
Raw Customer Complaint
          │
          ▼
   Text Preprocessing
(Lowercasing, URL removal, punctuation cleaning, whitespace normalization)
          │
          ▼
 Stratified 80/20 Train/Test Split
  (Strictly preventing data leakage)
          │
          ▼
 TF-IDF Feature Extraction
  (N-grams: 1-2, Sublinear TF, English Stop Words)
          │
     ┌────┴────────────────────────┐
     ▼                             ▼
Model 1: Logistic Regression   Model 2: Linear SVM
(max_iter=2000, random_state=42) (random_state=42)
     │                             │
     └────┬────────────────────────┘
          ▼
Comparative Evaluation (Accuracy, Precision, Recall, F1-Score)
          │
          ▼
Dynamic Final Model Selection (Highest Test F1-Score)
          │
          ▼
Serialized Artifacts (`models/` + `model_metadata.json`)
          │
          ▼
Real-Time Inference (`src/predict.py`) & Interactive Dashboard (`app.py`)
```

---

## 4. Dataset Overview
- **Source File:** `data/complaint_classification_dataset_1600.csv`
- **Total Records:** 1,600
- **Total Columns:** 3 (`id`, `complaint`, `category`)
- **Missing Values:** 0 (clean dataset)
- **Duplicate Records:** 0
- **Class Balance:** **Balanced** — exactly 200 records per category (12.5% each).


## 5. Complaint Categories & Business Scopes
1. **Billing & Payment:** Issues involving charges, unexpected fees, payment failures, invoices, and billing discrepancies.
2. **Delivery:** Issues involving shipment delays, courier tracking, incorrect delivery address, or missing packages.
3. **Product Issue:** Issues involving broken, defective, substandard, or mismatched physical/digital goods.
4. **Account & Login:** Issues involving password resets, two-factor authentication, account lockouts, and credential recovery.
5. **Technical Issue:** Issues involving application crashes, server errors, bugs, screen freezes, and file upload failures.
6. **Refund:** Issues involving pending refunds, delayed reimbursements, or incorrect refund sums.
7. **Cancellation:** Requests or issues involving cancelling recurring subscriptions, active orders, or bookings.
8. **General Inquiry:** Informational questions regarding product features, pricing plans, service availability, and general policy.

---

## 6. Data Preprocessing Pipeline (`src/preprocess.py`)
Text normalization is handled deterministically without heavy black-box dependencies:
1. **Null Safety Check:** Converts non-string or NaN values into safe empty strings.
2. **Lowercasing:** Converts all characters to lowercase so identical words in different cases map to the same vocabulary index.
3. **URL Removal:** Strips links (`http://`, `https://`, `www.`) which contain noisy domain-specific URLs.
4. **Punctuation & Special Character Removal:** Replaces punctuation marks with whitespace, preserving compound boundaries (e.g., `order#123` becomes `order 123`).
5. **Whitespace Collapse:** Normalizes tabs, multiple spaces, and trailing whitespace into single space delimiters.

---

## 7. TF-IDF Feature Extraction Explained
**TF-IDF** stands for **Term Frequency - Inverse Document Frequency**.

### Concept (B.Tech Data Science Level):
TF-IDF converts unstructured text into a numerical matrix by balancing two factors:
1. **Term Frequency ($\text{TF}$):** How often a word occurs in a specific complaint. If "refund" appears repeatedly in a complaint, its TF increases.
2. **Inverse Document Frequency ($\text{IDF}$):** How common or rare a word is across *all* 1,600 complaints. Common words across every category (e.g., "customer", "please", "help") receive low IDF weights, while domain-specific terms (e.g., "courier", "invoice", "password") receive high IDF weights.

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### Configured Hyperparameters:
```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True
)
```
- **`ngram_range=(1, 2)`:** Extracts both individual words ("payment", "deducted") and two-word phrases ("charged twice", "cannot login").
- **`sublinear_tf=True`:** Uses logarithmic term frequency ($1 + \log(\text{TF})$) to prevent words appearing 10 times from having 10x the mathematical weight of words appearing once.
- **Strict Leakage Prevention:** The vectorizer is fitted **only on the 80% training data** (`X_train`) and merely transformed on the 20% test data (`X_test`).

---

## 8. Why Classical Machine Learning Instead of Deep Learning?
> *"For a small structured text dataset, TF-IDF with linear classification provides a lightweight, fast and interpretable baseline that is suitable for real-time complaint classification."*

1. **Dataset Scale:** 1,600 records is ideal for linear convex optimizers. Complex Transformer models (BERT, RoBERTa) risk overfitting, require GPU compute, and introduce multi-second latency for single-sentence inference.
2. **Speed & Resource Efficiency:** Inference takes less than 2 milliseconds on CPU.
3. **Interpretability:** TF-IDF feature coefficients directly tell us which tokens triggered a category prediction.
4. **Demo Reliability:** Zero external API rate limits, zero GPU memory crashes, and 100% deterministic reproducibility.

---

## 9. Models Evaluated

### Model 1: Logistic Regression
- **Algorithm:** Multinomial Logistic Regression (`max_iter=2000`, `random_state=42`).
- **Mechanism:** Models the log-odds of each category as a linear combination of TF-IDF feature weights, applying the softmax function to output true probability distributions.

### Model 2: Linear Support Vector Machine (LinearSVC)
- **Algorithm:** Linear Support Vector Classifier (`random_state=42`).
- **Mechanism:** Finds maximum-margin hyperplanes that separate the high-dimensional TF-IDF vectors between classes with minimal hinge loss.

Both models were trained on identical feature representations to ensure a fair comparison.

---

## 10. Measured Model Evaluation Results

Tested on an independent **320-sample holdout test set** (40 samples per category):

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Linear SVM** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

*Both models achieved perfect classification on the test set due to well-curated lexical boundaries across the 8 categories.*

### Automated Selection:
The pipeline automatically selected **Logistic Regression** as the final deployed model (`models/final_model.pkl`) because it provides direct calibrated probability distributions (`predict_proba()`), enabling transparent confidence scores for customer service agents.

---

## 11. Distinctive TF-IDF Terms by Category
By computing the mean TF-IDF weights for each category, our system extracts the top distinctive keywords:

- **Billing & Payment:** `bill`, `charged`, `fee`, `deducted`, `payment`, `amount`, `unexpected`, `invoice`
- **Delivery:** `delivery`, `package`, `arrived`, `courier`, `tracking`, `shipment`, `address`, `order`
- **Product Issue:** `product`, `damaged`, `broken`, `parts`, `defective`, `item`, `missing`, `working`
- **Account & Login:** `account`, `password`, `login`, `email`, `verification`, `access`, `profile`, `reset`
- **Technical Issue:** `crashing`, `error`, `freezes`, `application`, `app`, `upload`, `screen`, `bug`
- **Refund:** `refund`, `received`, `promised`, `days`, `money`, `pending`, `returned`, `credit`
- **Cancellation:** `cancel`, `subscription`, `booking`, `stop`, `recurring`, `membership`, `order`
- **General Inquiry:** `inquiry`, `services`, `plans`, `features`, `information`, `help`, `pricing`, `details`

---

## 12. Project Structure

```
smart-complaint-classification/
│
├── data/
│   └── complaint_classification_dataset_1600.csv     # 1600 complaint records
│
├── models/
│   ├── tfidf_vectorizer.pkl                           # Fitted TF-IDF vectorizer
│   ├── logistic_regression.pkl                        # Trained Logistic Regression model
│   ├── linear_svm.pkl                                 # Trained Linear SVM model
│   ├── final_model.pkl                                # Winning deployed model
│   └── model_metadata.json                            # Metrics, categories, keywords
│
├── outputs/
│   ├── category_distribution.png                      # Category balance bar chart
│   ├── confusion_matrix_lr.png                        # Confusion matrix for LR
│   ├── confusion_matrix_svm.png                       # Confusion matrix for Linear SVM
│   ├── model_comparison.png                           # LR vs SVM metrics comparison
│   └── common_words.png                               # Top TF-IDF words per category
│
├── src/
│   ├── preprocess.py                                  # Data loader & text cleaner
│   ├── train.py                                       # Training, evaluation & export
│   ├── evaluate.py                                    # Metrics & plotting helpers
│   └── predict.py                                     # Standalone prediction pipeline
│
├── app.py                                             # 4-Tab Streamlit web app
├── requirements.txt                                   # Python dependencies
├── README.md                                          # Complete documentation
└── .gitignore                                         # Git ignore rules
```

---

## 13. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.

### Step 1: Clone or Navigate to the Project Folder
```bash
cd d:\projects\smart-complaint-classification
```

### Step 2: Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

---

## 14. How to Train the Models
To re-run the training pipeline, regenerate confusion matrices, and update serialized model files:
```bash
python src/train.py
```

The script will:
1. Load and validate `complaint_classification_dataset_1600.csv`.
2. Clean complaint text and verify balance.
3. Perform an 80/20 stratified split.
4. Fit the TF-IDF vectorizer and extract category keywords.
5. Train both Logistic Regression and Linear SVM.
6. Evaluate and display metrics tables.
7. Save models to `models/` and charts to `outputs/`.

---

## 15. How to Run Real-Time CLI Predictions
You can run standalone predictions from the command line without launching the UI:
```bash
python src/predict.py
```

Or import the predictor into your own Python code:
```python
from src.predict import predict_complaint

result = predict_complaint("My payment was deducted twice for the same order.")
print("Predicted Category:", result["predicted_category"])
print("Confidence:", result["score_percentage"])
```

---

## 16. How to Run the Streamlit Dashboard
Launch the interactive web application:
```bash
streamlit run app.py
```

The application opens at `http://localhost:8501` (or next available port) and provides four tabs:
- **🔍 Tab 1: Classify Complaint:** Live input text area, one-click demo prefill buttons, category badge, honest confidence percentage, category scope description, and keyword tags.
- **📊 Tab 2: Dataset Insights:** Dataset KPIs, category distribution bar chart, balance verification indicator, and searchable raw records viewer.
- **🤖 Tab 3: Model Performance:** Model comparison metrics table, performance bar chart, and test set confusion matrices.
- **📝 Tab 4: Category Keywords:** Interactive dropdown to inspect the top 10 TF-IDF distinctive terms per category with bar charts.

---

## 17. Demonstration Examples & Verification

Here are the 8 standard test examples evaluated by our system:

| # | Complaint Input | True Category | Predicted Category | Confidence / Score |
| :-: | :--- | :---: | :---: | :---: |
| 1 | *"My payment was deducted twice for the same order."* | **Billing & Payment** | **Billing & Payment** | 77.3% |
| 2 | *"My package shows delivered but I never received it."* | **Delivery** | **Delivery** | 91.8% |
| 3 | *"The product arrived broken and has missing parts."* | **Product Issue** | **Product Issue** | 91.3% |
| 4 | *"I forgot my password and cannot log into my account."* | **Account & Login** | **Account & Login** | 95.0% |
| 5 | *"The application crashes whenever I try to open it."* | **Technical Issue** | **Technical Issue** | 66.3% |
| 6 | *"I requested a refund several days ago but haven't received it."* | **Refund** | **Refund** | 90.4% |
| 7 | *"I want to cancel my subscription."* | **Cancellation** | **Cancellation** | 95.4% |
| 8 | *"What plans and services do you offer?"* | **General Inquiry** | **General Inquiry** | 45.6% |

---

## 18. Future Improvements
1. **Active Learning Feedback Loop:** Add a button in the UI for support agents to correct misclassified complaints and log them for retraining.
2. **Multi-Label Classification:** Extend the system to complaints spanning multiple departments (e.g., "Cancel my order and give me a refund" -> Cancellation + Refund).
3. **Sentiment & Urgency Scoring:** Incorporate sentiment polarity to flag angry or high-priority tickets for immediate supervisor review.