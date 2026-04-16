# Data Science Foundations — NumPy, Pandas & Matplotlib

A structured set of Jupyter notebooks covering the three most essential Python libraries
for data science and machine learning — NumPy for numerical computation, Pandas for data
manipulation, and Matplotlib for data visualization.

---

## Notebooks Overview

### 1. NumPy — `numpy_basics.ipynb`

NumPy is the foundation of all numerical computing in Python. This notebook covers
everything needed to work efficiently with arrays and mathematical operations.

**Topics Covered:**
- Creating arrays — `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`
- Array shapes, reshaping, and transposing
- Indexing, slicing, and boolean masking
- Vectorized arithmetic and universal functions (ufuncs)
- Broadcasting rules and how NumPy handles mismatched shapes
- Statistical operations — `mean`, `std`, `var`, `min`, `max`, `sum`
- Linear algebra — dot product, matrix multiplication, `np.linalg`
- Random module — `np.random.seed()`, `np.random.randn()`, `np.random.choice()`

**Key Concepts Demonstrated:**

```python
import numpy as np

# Array creation
arr = np.array([1, 2, 3, 4, 5])
matrix = np.zeros((3, 3))

# Vectorized operation (no loops needed)
result = arr ** 2 + np.sqrt(arr)

# Matrix multiplication
A = np.random.randn(3, 4)
B = np.random.randn(4, 2)
C = A @ B  # shape: (3, 2)
```

**Why it matters for ML:** NumPy is the backbone of scikit-learn, TensorFlow, and PyTorch.
Every model weight, gradient, and prediction is ultimately a NumPy array or tensor.

---

### 2. Pandas DataFrames — `dataframe_intro.ipynb`

Introduction to Pandas DataFrames — the most widely used data structure for loading,
exploring, and preparing datasets before feeding them into machine learning models.

**Topics Covered:**
- Creating DataFrames from dictionaries, lists, and CSV files
- Exploring data — `.head()`, `.tail()`, `.info()`, `.describe()`, `.shape`
- Column selection, row selection with `.loc[]` and `.iloc[]`
- Boolean filtering and conditional selection
- Handling missing values — `.isnull()`, `.dropna()`, `.fillna()`
- Adding, renaming, and dropping columns
- Sorting — `.sort_values()`, `.sort_index()`
- Data types and type casting — `.astype()`

**Key Concepts Demonstrated:**

```python
import pandas as pd

# Load data
df = pd.read_csv("data/sample.csv")

# Explore
print(df.shape)       # (rows, columns)
print(df.info())      # dtypes and non-null counts
print(df.describe())  # summary statistics

# Filter rows
senior_devs = df[(df["experience"] > 5) & (df["role"] == "Engineer")]

# Handle missing values
df["salary"].fillna(df["salary"].median(), inplace=True)
```

**Why it matters for ML:** Real-world datasets are messy. This notebook builds the skills
to clean, inspect, and understand any dataset before modeling.

---

### 3. Pandas Operations — `dataframe_operations.ipynb`

Advanced DataFrame operations for transforming, aggregating, and combining datasets —
the skills that separate a beginner from a proficient data practitioner.

**Topics Covered:**
- GroupBy operations — `.groupby()`, `.agg()`, `.transform()`
- Pivot tables and crosstabs
- Merging and joining DataFrames — `pd.merge()`, `.join()`
- Concatenating DataFrames — `pd.concat()`
- Applying custom functions — `.apply()`, `.map()`
- String operations with `.str` accessor
- DateTime operations with `.dt` accessor
- Encoding categorical variables — `pd.get_dummies()`, label encoding

**Key Concepts Demonstrated:**

```python
import pandas as pd

# GroupBy with multiple aggregations
summary = df.groupby("department").agg(
    avg_salary=("salary", "mean"),
    total_employees=("name", "count"),
    max_experience=("experience", "max")
).reset_index()

# Merging two DataFrames
merged = pd.merge(employees, departments, on="dept_id", how="left")

# One-hot encoding for ML
df_encoded = pd.get_dummies(df, columns=["city", "role"], drop_first=True)
```

**Why it matters for ML:** Feature engineering lives here. GroupBy, merge, and encode
are daily operations in any real ML pipeline.

---

### 4. Matplotlib — `matplotlib_visuals.ipynb`

Data visualization is how we communicate insights. This notebook covers Matplotlib from
basic plots to publication-quality figures.

**Topics Covered:**
- Line plots, bar charts, histograms, scatter plots, pie charts
- Figure and axes — `plt.figure()`, `fig, ax = plt.subplots()`
- Customization — titles, labels, legends, colors, line styles, markers
- Subplots and grid layouts — `plt.subplot()`, `gridspec`
- Annotations and text on plots
- Saving figures — `plt.savefig()` with DPI control
- Styling — `plt.style.use()`, custom color palettes
- Plotting directly from Pandas — `df.plot()`

**Key Concepts Demonstrated:**

```python
import matplotlib.pyplot as plt
import numpy as np

# Multi-subplot figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Line plot
x = np.linspace(0, 10, 100)
axes[0].plot(x, np.sin(x), label="sin(x)", color="steelblue", linewidth=2)
axes[0].set_title("Sine Wave")
axes[0].legend()

# Histogram
data = np.random.randn(1000)
axes[1].hist(data, bins=30, color="coral", edgecolor="black", alpha=0.7)
axes[1].set_title("Normal Distribution")

plt.tight_layout()
plt.savefig("reports/figures/sample_plots.png", dpi=150)
plt.show()
```

**Why it matters for ML:** Visualizing distributions, correlations, model predictions,
and training curves is essential for debugging and communicating results.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

## Tools and Libraries

| Library    | Version | Purpose                             |
|------------|---------|-------------------------------------|
| NumPy      | 1.24+   | Numerical arrays and linear algebra |
| Pandas     | 2.0+    | Data manipulation and analysis      |
| Matplotlib | 3.7+    | Data visualization                  |
| Jupyter    | 7.0+    | Interactive notebook environment    |

---

## Key Learnings

- NumPy vectorized operations are 10–100x faster than Python loops for numerical tasks
- Pandas `.loc[]` vs `.iloc[]` — label-based vs position-based indexing are two different ways to access data
- `groupby().agg()` can replace dozens of lines of manual loop-based aggregation
- Matplotlib's object-oriented API (`fig, ax`) gives more control than `plt.plot()` for complex figures
- Always explore data before modeling — `.describe()`, `.value_counts()`, and histograms reveal hidden issues

---

## References

- NumPy Official Documentation — https://numpy.org/doc/
- Pandas Official Documentation — https://pandas.pydata.org/docs/
- Matplotlib Official Documentation — https://matplotlib.org/stable/contents.html
- Python Data Science Handbook — https://jakevdp.github.io/PythonDataScienceHandbook/

---

## Author

**DB Moktan**
Aspiring Machine Learning Engineer
Kathmandu, Nepal

GitHub — https://github.com/DBMoktan
LinkedIn — https://www.linkedin.com/in/db-moktan/

---

*Part of my ML Engineering learning journey — building strong fundamentals before
diving into deep learning and production systems.*
