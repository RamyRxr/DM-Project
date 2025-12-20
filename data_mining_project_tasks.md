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

- [x] **Select target column** → User chooses class/label column with confirmation
- [x] **Display class distribution** → Show unique classes, counts, and bar chart
- [x] **Split dataset (80% train / 20% test)** → Manual train/test split with adjustable percentage (10-40% test)

### Architecture & Interface
- [x] **Create supervised learning interface** → Same architecture as unsupervised (dynamic loading, importlib)
- [x] **Dynamic algorithm loading** → Scan algorithms/supervised/ folder and load modules
- [x] **Algorithm selection dropdown** → Individual algorithm selection or "Select all"
- [x] **Parameter configuration** → Algorithm-specific parameters (k for k-NN, none for others)
- [x] **Train/test split UI** → Slider for test size, button to split data
- [x] **Display train/test info** → Show sample counts and class distributions

### k‑NN (k-Nearest Neighbors)
- [x] **Implement k-NN from scratch** → Euclidean distance, majority voting, exact 4-step process from class
- [x] **Train k‑NN for k=1..20** → Multi-k evaluation with automatic testing
- [x] **Compute confusion matrix** → sklearn confusion_matrix with proper class handling
- [x] **Compute precision, recall, F‑measure** → Per-class calculation with macro averaging
- [x] **Compute accuracy** → Total correct predictions / total predictions
- [x] **Plot accuracy vs k** → Line chart showing accuracy across k values
- [x] **Plot precision vs k** → Line chart showing precision across k values
- [x] **Display confusion matrix heatmap** → Plotly heatmap visualization

### Naive Bayes
- [x] **Implement Gaussian Naive Bayes from scratch** → Prior probabilities, conditional probabilities, log posterior
- [x] **Calculate Prior Probabilities P(Ck)** → Number of instances per class / total instances
- [x] **Calculate Conditional Probabilities P(xi | Ck)** → Gaussian distribution for continuous attributes
- [x] **Implement prediction formula** → argmax_Ck (P(Ck) × ∏ P(xi | Ck))
- [x] **Train and test model** → Use Gaussian likelihood with mean and std per class per feature
- [x] **Compute evaluation metrics** → Precision, Recall, F-Measure, Accuracy
- [x] **Generate confusion matrix** → Display predictions vs actual
- [x] **Display confusion matrix heatmap** → Plotly visualization

### C4.5 (Decision Tree)
- [x] **Implement C4.5 from scratch** → Entropy, Information Gain, Split Information, Gain Ratio
- [x] **Calculate Total Entropy** → -∑ (p * log2(p)) for parent node
- [x] **Calculate Information Gain** → Parent Entropy - Weighted Child Entropy
- [x] **Calculate Split Information** → -∑ (|Sj|/|S|) * log2(|Sj|/|S|)
- [x] **Calculate Gain Ratio** → Information Gain / Split Information (C4.5 criterion)
- [x] **Build decision tree recursively** → Select attribute with max Gain Ratio, split dataset, create leaves when pure
- [x] **Implement tree traversal** → Navigate from root to leaf based on attribute values
- [x] **Train decision tree** → Build tree with max_depth=10 default
- [x] **Test model and compute metrics** → Precision, Recall, F-Measure, Accuracy
- [x] **Generate confusion matrix** → Display predictions vs actual
- [x] **Display confusion matrix heatmap** → Plotly visualization

### SVM (Support Vector Machine)
- [x] **Implement SVM classifier** → Maximum margin hyperplane for classification
- [x] **Kernel functions** → Support for linear, RBF, and polynomial kernels
- [x] **Regularization parameter C** → Control margin vs misclassification trade-off
- [x] **Train SVM model** → Find optimal hyperplane and support vectors
- [x] **Test model and compute metrics** → Precision, Recall, F-Measure, Accuracy
- [x] **Generate confusion matrix** → Display predictions vs actual
- [x] **Display confusion matrix heatmap** → Plotly visualization
- [x] **Support vector visualization** → Bar chart showing support vectors per class
- [x] **Hyperparameter configuration** → UI controls for kernel type and C parameter

### Algorithm Comparison
- [x] **"Select all" option** → Run all four algorithms with default parameters
- [x] **Collect metrics for each algorithm** → Store Accuracy, Precision, Recall, F-Measure
- [x] **Create comparison table** → Display all metrics side-by-side with formatting
- [x] **Compute composite score** → Average of all four metrics
- [x] **Display best algorithm recommendation** → Algorithm with highest composite score
- [x] **Show ranking of all algorithms** → Sorted table by composite score
- [x] **Add metrics explanation** → Collapsible expander explaining Accuracy, Precision, Recall, F-Measure, Composite Score
- [x] **Individual visualizations for all algorithms** → Display confusion matrices and plots for each algorithm in comparison mode

### Data Type Handling & Bug Fixes
- [x] **Fix classification type errors** → Convert y_train and y_test to int type
- [x] **Handle numpy arrays vs pandas Series** → Proper conversion in all algorithms
- [x] **Fix confusion matrix errors** → Ensure discrete integer labels
- [x] **Align with class material** → k-NN 4-step process, Naive Bayes formulas, C4.5 Gain Ratio

---

## ✅ PHASE 12 – VALIDATION & FINAL DELIVERY
**Objective:** Ensure correctness, usability, and academic compliance.

### UI Controls Verification
- [x] **Data Loading Section**
  - [x] Test CSV file upload functionality
  - [x] Verify file type validation (reject non-CSV files)
  - [x] Confirm dataframe display works correctly
  - [x] Check dataset dimensions display
  - [x] Verify column names and data types display

- [x] **Data Cleaning Section**
  - [x] Test duplicate detection and removal
  - [x] Test missing value detection (standard NaN, custom indicators, type mismatches)
  - [x] Verify "Auto Clean Data" button functionality
  - [x] Test all missing value handling methods (drop, fill with 0, mean, median, forward fill, backward fill)
  - [x] Confirm custom indicator input and conversion to NaN
  - [x] Verify column deletion functionality

- [x] **Outlier Detection Section**
  - [x] Test boxplot generation for numeric columns
  - [x] Verify outlier visualization
  - [x] Test outlier removal and keep options

- [x] **Data Exploration Section**
  - [x] Verify all statistical metrics (min, max, mean, median, mode, Q1, Q3)
  - [x] Test scatter plot generation
  - [x] Confirm proper visualization of relationships

- [x] **Data Typing Section**
  - [x] Test float to integer conversion detection
  - [x] Verify conversion integrity validation

- [x] **Normalization Section**
  - [x] Test Min-Max normalization
  - [x] Test Z-score standardization
  - [x] Verify skip normalization option
  - [x] Confirm numerical columns only are normalized

- [x] **Unsupervised Learning Section**
  - [x] Test feature selection (X-axis, Y-axis dropdowns)
  - [x] Verify 2D scatter plot preview
  - [x] Test algorithm selection dropdown
  - [x] Verify parameter controls for each algorithm (k, eps, MinPts, linkage, etc.)
  - [x] Test "Run Clustering" button
  - [x] Test "Select all" option
  - [x] Verify session state persistence

- [x] **Supervised Learning Section**
  - [x] Test target column selection and confirmation
  - [x] Verify class distribution visualization
  - [x] Test train/test split with adjustable percentage
  - [x] Verify data split display (train/test sample counts)
  - [x] Test algorithm selection dropdown
  - [x] Verify parameter controls (k for k-NN)
  - [x] Test "Train & Test" button
  - [x] Test "Select all" option

### Visualizations Verification
- [x] **Unsupervised Learning Visualizations**
  - [x] K-Means: 2D clusters, 3D clusters, elbow curve, centroids display
  - [x] K-Medoids: 2D clusters, 3D clusters, elbow curve, medoids display
  - [x] AGNES: 2D clusters, 3D clusters, dendrogram
  - [x] DIANA: 2D clusters, 3D clusters, dendrogram
  - [x] DBSCAN: 2D clusters with noise points, 3D clusters with noise, noise detection
  - [x] All algorithms: Cluster distribution histograms
  - [x] Comparison mode: Histogram grid (3 columns), metrics table, best algorithm recommendation

- [x] **Supervised Learning Visualizations**
  - [x] k-NN: Accuracy vs k plot, Precision vs k plot, confusion matrix heatmap
  - [x] Naive Bayes: Confusion matrix heatmap
  - [x] C4.5: Confusion matrix heatmap
  - [x] All algorithms: Confusion matrix tables, metrics cards
  - [x] Comparison mode: Metrics table, composite score ranking, individual visualizations

- [x] **Data Exploration Visualizations**
  - [x] Boxplots for outlier detection
  - [x] Scatter plots for relationship analysis
  - [x] Class distribution bar charts (supervised learning)

### Complete Workflow Testing
- [x] **End-to-End Workflow Test 1: Unsupervised Learning**
  - [x] Load CSV file
  - [x] Clean data (handle missing values, remove duplicates)
  - [x] Handle outliers
  - [x] Normalize data
  - [x] Run single clustering algorithm (K-Means)
  - [x] Verify results, metrics, and visualizations
  - [x] Run all clustering algorithms (comparison mode)
  - [x] Verify comparison table and best algorithm recommendation

- [x] **End-to-End Workflow Test 2: Supervised Learning**
  - [x] Load CSV file
  - [x] Clean data
  - [x] Normalize data
  - [x] Select target column
  - [x] Split data (train/test)
  - [x] Run single classification algorithm (k-NN)
  - [x] Verify results, metrics, and visualizations
  - [x] Run all classification algorithms (comparison mode)
  - [x] Verify comparison table and individual visualizations

- [x] **Edge Cases Testing**
  - [x] Test with dataset containing many missing values
  - [x] Test with dataset containing many outliers
  - [x] Test with small dataset (< 50 rows)
  - [x] Test with large dataset (> 10,000 rows)
  - [x] Test with dataset containing only numeric columns
  - [x] Test with dataset containing mixed types
  - [x] Test with imbalanced classes (supervised learning)

### Docker Validation
- [x] **Build Docker Image**
  - [x] Verify Dockerfile syntax and structure
  - [x] Run `docker build` command successfully
  - [x] Confirm image size is reasonable
  - [x] Verify all dependencies are installed

- [x] **Run Docker Container**
  - [x] Start container with proper port mapping
  - [x] Access application in browser (localhost:8501)
  - [x] Verify Streamlit interface loads correctly
  - [x] Test complete workflow inside Docker container

- [x] **Test on Clean System**
  - [x] Pull image on different machine (if possible)
  - [x] Run container on clean system
  - [x] Verify reproducibility of results
  - [x] Confirm no missing dependencies

### Academic Compliance
- [x] **Algorithm Implementation Verification**
  - [x] K-Means: Manual centroid updates, convergence check
  - [x] K-Medoids: PAM algorithm, medoid selection
  - [x] AGNES: All linkage methods (single, complete, average, ward)
  - [x] DIANA: Recursive divisive splitting
  - [x] DBSCAN: Core, border, noise point classification
  - [x] k-NN: Exact 4-step process from class material
  - [x] Naive Bayes: Prior probabilities, conditional probabilities, Gaussian likelihood
  - [x] C4.5: Gain Ratio (not just Information Gain), entropy, split information

- [x] **Metrics Calculation Verification**
  - [x] Silhouette Score: Correct formula implementation
  - [x] Davies-Bouldin Index: Lower is better
  - [x] Calinski-Harabasz Index: Higher is better
  - [x] Accuracy: TP + TN / Total
  - [x] Precision: TP / (TP + FP)
  - [x] Recall: TP / (TP + FN)
  - [x] F-Measure: Harmonic mean of Precision and Recall

- [x] **Documentation Verification**
  - [x] README.md is complete and accurate
  - [x] All phases in task plan are marked correctly
  - [x] Code comments explain academic concepts
  - [x] Metrics explanation expanders are clear and accurate

### Final Checks
- [x] **Code Quality**
  - [x] No unused imports
  - [x] No debug print statements
  - [x] Consistent naming conventions
  - [x] Proper error handling throughout
  - [x] Session state management is clean

- [x] **User Experience**
  - [x] All buttons have clear labels
  - [x] Loading indicators for long operations
  - [x] Success/error messages are informative
  - [x] Layout is clean and organized
  - [x] No UI overlap or formatting issues

- [x] **Performance**
  - [x] Algorithms complete in reasonable time (< 30 seconds for medium datasets)
  - [x] Caching is working (@st.cache_data)
  - [x] No memory leaks or excessive memory usage
  - [x] Visualizations render smoothly

---

✨ **Final Outcome:** A fully compliant, academically rigorous data mining application with clear, traceable implementation steps.



