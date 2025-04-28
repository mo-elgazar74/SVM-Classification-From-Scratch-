
# SVM Classification Project (From Scratch)

This project demonstrates binary classification using **Support Vector Machines (SVM)** in Python. A custom modular class has been created to manage training, evaluation, and visualization. The notebook includes detailed data visualizations, model comparisons, and decision boundary plots.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Notebook Structure (Important Cells)](#notebook-structure-important-cells)
- [Custom VSM Class](#custom-vsm-class)
- [Data Visualization](#data-visualization)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Project Overview

- Implement SVM for binary classification.
- Compare different kernel functions (**linear**, **polynomial**, **rbf**).
- Visualize datasets and decision boundaries.
- Use a **custom class** to organize the workflow (training, evaluation, plotting).

---

## Technologies Used

- Python 3.x
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Notebook Structure (Important Cells)

| Cell Stage | Description | Importance |
|:-----------|:------------|:-----------|
| **Importing Libraries** | Import necessary libraries like `numpy`, `pandas`, `matplotlib`, `sklearn`. | Essential for enabling ML, visualization, and data handling. |
| **Data Loading** | Load datasets, typically synthetic (`make_moons`, `make_circles`, etc.) or real datasets. | Provides data for training and visualization. |
| **Data Visualization (Before Modeling)** | Plot the raw input data to understand its distribution and potential separability. | Helps see if linear/non-linear kernels will work better. |
| **Data Preprocessing** | Standardize features using `StandardScaler`. Split into training and test sets. | Ensures fair model training and testing. |
| **VSM Class Definition** | Define the custom `VSM` (Visual SVM) class. | Modularizes SVM logic: training, predicting, scoring, and plotting. |
| **Model Training (Linear Kernel)** | Train the SVM with a linear kernel using the `VSM` class. | First baseline model to compare. |
| **Visualization (Linear Kernel)** | Plot decision boundaries for linear SVM. | Check if data is linearly separable. |
| **Model Training (RBF Kernel)** | Train SVM with the RBF kernel. | Non-linear decision surface. |
| **Visualization (RBF Kernel)** | Plot decision boundaries for RBF SVM. | Shows improved fitting for curved boundaries. |
| **Model Training (Polynomial Kernel)** | Train SVM with polynomial kernel (typically degree 3). | Captures more complex non-linear patterns. |
| **Visualization (Polynomial Kernel)** | Plot decision boundaries for polynomial SVM. | Helps understand overfitting/underfitting. |
| **Comparison Summary** | Compare model accuracies and boundary plots. | Final evaluation of the best model. |

---

## Custom VSM Class

The notebook defines a class named **`VSM`** to encapsulate the entire SVM workflow:

### Methods in the `VSM` Class:
- `__init__` → Initialize SVM model with specified kernel and hyperparameters.
- `train(X_train, y_train)` → Train the SVM model.
- `predict(X_test)` → Make predictions.
- `evaluate(X_test, y_test)` → Calculate accuracy and confusion matrix.
- `plot_decision_boundary(X, y)` → Visualize decision regions and true labels.

✅ **Benefits**:
- Code reusability
- Cleaner notebook
- Easier experimentation with different models

---

## 📊 Data Visualization

Visualization is a key strength of this notebook:

### 1. **Initial Dataset Visualization**
- Before any model is trained, the dataset is plotted using **Matplotlib scatter plots**.
- Different classes are colored differently.
- Purpose: To understand the complexity of the classification task.

### 2. **Decision Boundary Visualization**
- After model training, the decision boundary is plotted.
- The boundary indicates regions classified as each class.
- Background color represents the prediction class, while scatter points show true labels.
- Clear differences between Linear, RBF, and Polynomial kernels are visible:
  - **Linear Kernel**: Straight-line boundaries.
  - **RBF Kernel**: Smooth, curved boundaries adapting to data shape.
  - **Polynomial Kernel**: Complex curves, depending on degree.

---

## How to Run

1. Clone the repository or download the notebook.
2. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
3. Open and run:
   ```bash
   jupyter notebook SVM-Classification.ipynb
   ```
4. Execute all cells sequentially to see full workflow.

---

## Results

- **Linear SVM** performed well for nearly linearly separable data but failed for complex curves.
- **RBF Kernel** provided the best performance for non-linear datasets.
- **Polynomial Kernel** showed good results but risked overfitting with high degrees.
- Visualization allowed clear identification of model performance visually.

---

## 🛠️ Future Improvements

- Apply **GridSearchCV** for automated hyperparameter tuning.
- Test on real-world datasets (e.g., breast cancer, digits).
- Extend the `SVM` class to support multi-class classification (One-vs-One, One-vs-Rest).

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
