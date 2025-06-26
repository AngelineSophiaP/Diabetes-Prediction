
## 🩺 Diabetes Prediction using Logistic Regression

### 📌 Project Overview

This project aims to **predict whether a person has diabetes or not** based on medical features such 
as glucose level, blood pressure, BMI, age, and more.

It uses the **Pima Indians Diabetes Dataset**, a widely used dataset for binary classification tasks.

---

### 🎯 Problem Statement

Can we predict if a patient is diabetic based on diagnostic measurements?

This is a **binary classification problem**:

* `1` → Diabetic
* `0` → Not Diabetic

---

### 🧠 Why Logistic Regression?

**Logistic Regression** was used because:

* ✅ It's well-suited for **binary classification**
* ✅ It's simple and interpretable
* ✅ It provides **probability scores**, not just hard predictions
* ✅ It performs well when the relationship between features and output is approximately **linear**

---

### 🧪 Dataset Information

* 📄 Source: Pima Indians Diabetes Dataset
* 🔢 Rows: 768
* 🧬 Features:

  * Pregnancies
  * Glucose
  * Blood Pressure
  * Skin Thickness
  * Insulin
  * BMI
  * Diabetes Pedigree Function
  * Age
* 🎯 Target: `Outcome` (0 or 1)

---

### 📊 Workflow Followed

```
Load → Clean → Split → Train → Predict → Evaluate
```

#### 1. **Load**

* Data loaded using **pandas**
* Custom column names added

#### 2. **Clean**

* Removed duplicates
* Checked for missing/null values

#### 3. **Split**

* Used `train_test_split()` from **scikit-learn**
* 80% training, 20% testing

#### 4. **Train**

* Trained a `LogisticRegression` model (`max_iter=1000`) on training data

#### 5. **Predict**

* Model predicts diabetes status for test data

#### 6. **Evaluate**

* Evaluated using:

  * **Accuracy Score**
  * **Confusion Matrix**
  * **Classification Report** (Precision, Recall, F1-Score)

---

### 💡 Learnings

* Logistic Regression is powerful for binary tasks.
* Cleaning and understanding the data is just as important as training the model.
* Model evaluation must include more than just accuracy (especially for imbalanced datasets).

---

### 📁 Libraries Used

* `pandas` — data handling
* `numpy` — numerical operations
* `scikit-learn` — model training and evaluation
* `matplotlib / seaborn` *(optional)* — for visualization

---

