# 📊 Data Mining Application — Ultra‑Detailed Task Plan

**Author:** Ramy  
**Version:** 3.0  
**Goal:** Build a complete academic data mining application that strictly follows course material, with every task broken down into explicit, executable steps, similar in detail to a professional software development roadmap.

---

## ⚠️ GLOBAL RULE (NON‑NEGOTIABLE)
> If a feature, algorithm, metric, or technique is **not explicitly taught in class → DO NOT ADD IT**.

---

## 🧩 PHASE 1 – PLANNING & PROJECT INITIALIZATION
**Objective:** Create a clean, well‑organized, and traceable project foundation before writing any logic.

- [ ] **Create a new project workspace** → Initialize a root folder named `data-mining-app`.
- [ ] **Define global folder structure** → Create folders such as `app/`, `preprocessing/`, `unsupervised/`, `supervised/`, `metrics/`, `visualization/`, and `utils/`.
- [ ] **Define file naming conventions** → Decide naming rules for Python files, functions, and variables (snake_case only).
- [ ] **Create an initial README.md** → Describe project goal, academic constraints, technologies used, and global structure.
- [ ] **Define phase‑by‑phase milestones** → Clearly specify when a phase is considered completed.
- [ ] **Initialize Git repository** → Create repository, initial commit, and define commit message rules.

---

## 🐳 PHASE 2 – ENVIRONMENT & DOCKER SETUP
**Objective:** Ensure reproducibility and isolation of the application.

- [ ] **Create `requirements.txt`** → Add only allowed libraries (streamlit, pandas, numpy, sklearn, plotly).
- [ ] **Fix library versions** → Lock versions exactly as specified by the course.
- [ ] **Create Dockerfile** → Define Python base image, copy files, install dependencies, expose Streamlit port.
- [ ] **Configure Streamlit entry point** → Set `app.py` as the main application file.
- [ ] **Build Docker image** → Verify build completes without errors.
- [ ] **Run Docker container** → Ensure app runs correctly in browser.

---

## 📂 PHASE 3 – DATA LOADING & DATASET UNDERSTANDING
**Objective:** Load and inspect the dataset before any modification.

- [x] **Implement CSV uploader** → Allow user to upload a CSV file only.
- [x] **Validate file type** → Reject non‑CSV files with a clear error message.
- [x] **Load dataset into Pandas DataFrame** → Store dataset in memory.
- [x] **Display full dataset** → Render the entire DataFrame with scrolling support.
- [x] **Display dataset dimensions** → Show number of rows and columns.
- [x] **Display column names** → List all attributes clearly.
- [x] **Display data types** → Show dtype for each column.

---

## 🧹 PHASE 4 – DATA CLEANING & QUALITY CONTROL
**Objective:** Fix inconsistencies and missing information using simple, user‑controlled methods.

### Duplicate Handling
- [x] **Detect duplicate rows** → Identify duplicated records.
- [x] **Display duplicate count** → Show number of duplicates found.
- [x] **Remove duplicates** → Delete duplicates only after user confirmation.

### Missing Values Handling
- [x] **Detect missing values per column** → Count nulls for each attribute. (Now includes custom indicators and type mismatches with detailed summary and improved UI)
- [x] **Display missing value table** → Summarize missing data visually. (Now shows standard NaNs and "messing" values with new UI and dynamic updates)
- [x] **Drop rows with missing values** → Remove affected rows.
- [x] **Drop selected columns** → Allow column‑level deletion.
- [x] **Replace missing values with 0** → Apply numeric replacement.
- [x] **Replace missing values with mean** → Use column mean.
- [x] **Forward fill missing values** → Propagate previous value.
- [x] **Backward fill missing values** → Propagate next value.
- [x] **Replace missing values with median** -> Use column median.
- [x] **Implement Auto Clean Data button** → Converts all detected non-standard values/type mismatches to NaN.
- [x] **Implement Apply Custom Indicators to NaN button** → Converts user-defined indicators to NaN.

---

## 📉 PHASE 5 – OUTLIER DETECTION & HANDLING
**Objective:** Identify extreme values visually using basic techniques.

- [x] **Generate boxplot per numeric column** → Visualize data spread.
- [x] **Identify outliers visually** → Detect extreme values.
- [x] **Remove outliers** → Delete selected outliers.
- [x] **Keep outliers** → Allow user to ignore outliers.

---

## 📊 PHASE 6 – DATA EXPLORATION & DESCRIPTIVE ANALYSIS
**Objective:** Understand dataset distribution and relationships.

- [x] **Compute minimum values**
- [x] **Compute maximum values**
- [x] **Compute mean**
- [x] **Compute median**
- [x] **Compute mode**
- [x] **Compute first quartile (Q1)**
- [x] **Compute third quartile (Q3)**
- [x] **Generate scatter plots** → Visualize relationships between two attributes.

---

## 🧠 PHASE 7 – DATA TYPING & OPTIMIZATION
**Objective:** Improve data representation without changing meaning.

- [x] **Detect float columns representing integers**
- [x] **Convert floats to integers when valid**
- [x] **Validate conversion integrity**

---

## ⚙️ PHASE 8 – DATA NORMALIZATION
**Objective:** Prepare numerical data for learning algorithms.

- [x] **Allow user to choose normalization strategy**
- [x] **Apply Min‑Max normalization**
- [x] **Apply Z‑score standardization**
- [x] **Skip normalization option**
- [x] **Normalize numerical columns only**

---

## 🔍 PHASE 9 – UNSUPERVISED LEARNING
**Objective:** Discover hidden clusters using algorithms implemented from scratch.

- [x] **Create algorithm selection menu** → Dynamic loading from algorithms/unsupervised/ folder
- [x] **Create parameter input controls** → Dynamic parameters based on selected algorithm
- [x] **Add execute button** → "Run Clustering" button with proper state management
- [x] **Add "Select all" option** → Run and compare all algorithms at once

### Interface Features

- [x] **Display cleaned data table** → Show current data before clustering
- [x] **Feature selection for 2D visualization** → X-axis and Y-axis dropdowns with Apply button
- [x] **2D scatter plot preview** → Display selected features before running algorithm
- [x] **2-column layout** → Algorithm selection | Parameter configuration
- [x] **Session state management** → Persist selected features and results across reruns
- [x] **Dynamic algorithm loading** → Use importlib to scan and load algorithm modules

### Algorithms

- [x] **Implement K‑Means from scratch** → Include centroid update, convergence condition, manual implementation
- [x] **Implement K‑Medoids from scratch** → PAM algorithm with medoid updates
- [x] **Implement AGNES** → Single, complete, average, ward linkage options
- [x] **Implement DIANA** → Recursive divisive clustering
- [x] **Implement DBSCAN** → eps, MinPts, noise detection, core/border/noise points

### Visualizations

- [x] **Add 2D cluster visualization** → Scatter plots with cluster colors
- [x] **Add 3D cluster visualization** → 3D scatter plots when sufficient numeric columns available
- [x] **Display centroid/medoid positions** → Markers on 2D and 3D plots for K-Means/K-Medoids
- [x] **Display elbow method graph** → Automatic k-range testing (k=2 to k=10) for K-Means & K-Medoids
- [x] **Detect and display optimal k** → Derivative-based elbow detection with recommendation
- [x] **Display dendrograms** → Hierarchical tree visualization for AGNES & DIANA
- [x] **Add cluster label column to dataset** → Cluster assignments added to dataframe
- [x] **Cluster distribution histograms** → Show count of points per cluster

### Metrics & Evaluation

- [x] **Compute Silhouette Score** → Measure cluster separation quality
- [x] **Compute Davies-Bouldin Index** → Evaluate cluster compactness and separation
- [x] **Compute Calinski-Harabasz Index** → Variance ratio criterion
- [x] **Compute Inertia/WCSS** → Within-cluster sum of squares for K-Means/K-Medoids
- [x] **Handle DBSCAN noise points** → Separate metrics for clustered points (excluding noise)
- [x] **Display metrics in card format** → Clean presentation with st.metric()

### Algorithm Comparison (Extra Feature)

- [x] **Run all algorithms with optimal parameters** → Automatic execution with k=3 default
- [x] **Collect and display histograms** → 3-column grid showing all cluster distributions
- [x] **Create comparison table** → Side-by-side metrics for all algorithms
- [x] **Compute composite score** → Weighted combination of all metrics
- [x] **Algorithm recommendation** → Automatic best algorithm selection based on scores
- [x] **Score ranking display** → Show all algorithms ranked by performance
- [x] **Metrics explanation** → Info expander with detailed metric descriptions

---

## 📈 PHASE 10 – UNSUPERVISED COMPARISON
**Objective:** Identify the best clustering algorithm.

- [x] **Run all algorithms automatically** → "Select all" option in algorithm dropdown
- [x] **Collect metrics for each algorithm** → Silhouette, Davies-Bouldin, Calinski-Harabasz
- [x] **Create comparison table** → Algorithm, all metrics side-by-side with formatting
- [x] **Create histogram grid** → Display cluster distributions for all algorithms
- [x] **Compute composite score** → Normalized weighted scoring system
- [x] **Display best algorithm recommendation** → Automatic selection with reasoning
- [x] **Show ranking of all algorithms** → Sorted by composite score

---

## 🎯 PHASE 11 – SUPERVISED LEARNING
**Objective:** Train classifiers on labeled data.

- [ ] **Select target column**
- [ ] **Split dataset (80% train / 20% test)**

### k‑NN
- [ ] **Train k‑NN for k=1..10**
- [ ] **Compute confusion matrix**
- [ ] **Compute precision, recall, F‑measure**
- [ ] **Plot precision vs k**

### Naive Bayes
- [ ] **Train and test model**
- [ ] **Compute evaluation metrics**

### C4.5
- [ ] **Train decision tree**
- [ ] **Test model and compute metrics**

---

## ✅ PHASE 12 – VALIDATION & FINAL DELIVERY
**Objective:** Ensure correctness, usability, and academic compliance.

- [ ] **Verify all UI controls**
- [ ] **Verify all visualizations**
- [ ] **Test complete workflow**
- [ ] **Validate Docker execution on clean system**

---

✨ **Final Outcome:** A fully compliant, academically rigorous data mining application with clear, traceable implementation steps.

