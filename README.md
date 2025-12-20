# 📊 Data Mining Application

A comprehensive, academically rigorous data mining application built with Streamlit that implements clustering and classification algorithms from scratch, strictly following academic course material.

---

## 🚀 Quick Start Guide

### Prerequisites

Before installing, make sure you have one of the following:

- **Option 1:** Docker installed on your system
- **Option 2:** Python 3.9+ installed

---

## 📥 Installation & Usage

### Step 0: Clone the Repository (Required for Both Options)

First, clone the repository to your local machine:

```bash
git clone <repository-url>
cd DM-Project
```

---

### Option 1: Using Docker (Recommended)

Docker ensures the application runs consistently across all systems without dependency issues.

#### Step 1: Build the Docker Image

```bash
docker build -t data-mining-app .
```

#### Step 2: Run the Container

```bash
docker run -p 8501:8501 data-mining-app
```

#### Step 3: Access the Application

Open your browser and navigate to:
```
http://localhost:8501
```

---

### Option 2: Using Python Directly

If you prefer running without Docker:

#### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 2: Run the Application

```bash
streamlit run app.py
```

#### Step 3: Access the Application

Your browser should automatically open to:
```
http://localhost:8501
```

---

## 📖 How to Use the Application

### 1. **Data Loading**
- Click **"Browse files"** in the sidebar
- Upload a **CSV file** containing your dataset
- The application will display your data with dimensions and column information

### 2. **Data Cleaning** (Phase 4)
Navigate through the data cleaning workflow:

- **Duplicates:** Detect and remove duplicate rows
- **Missing Values:** 
  - View missing data summary
  - Use "Auto Clean Data" to convert non-standard values to NaN
  - Add custom missing indicators (e.g., "?", "N/A", "missing")
  - Choose handling method: Drop rows, Fill with 0/Mean/Median, Forward/Backward fill
- **Column Management:** Delete unnecessary columns

### 3. **Outlier Detection** (Phase 5)
- View boxplots for each numeric column
- Identify extreme values visually
- Choose to remove or keep outliers

### 4. **Data Exploration** (Phase 6)
- View descriptive statistics: Min, Max, Mean, Median, Mode, Q1, Q3
- Generate scatter plots to visualize relationships between attributes

### 5. **Data Typing** (Phase 7)
- Automatically detect float columns that represent integers
- Convert them to proper integer types

### 6. **Normalization** (Phase 8)
Choose your normalization strategy:
- **Min-Max Normalization:** Scales values to [0, 1]
- **Z-Score Standardization:** Centers data with mean=0, std=1
- **Skip:** Proceed without normalization

### 7. **Unsupervised Learning** (Phase 9)
Discover hidden patterns in your data:

#### Select Features
- Choose X-axis and Y-axis features for 2D visualization
- Preview data distribution before clustering

#### Choose Algorithm
- **K-Means:** Centroid-based clustering with elbow method
- **K-Medoids (PAM):** Medoid-based clustering, more robust to outliers
- **AGNES:** Hierarchical agglomerative clustering with dendrogram
- **DIANA:** Hierarchical divisive clustering
- **DBSCAN:** Density-based clustering with noise detection

#### Configure Parameters
- K-Means/K-Medoids: Number of clusters (k)
- AGNES: Linkage method (single, complete, average, ward)
- DBSCAN: eps (neighborhood radius), MinPts (minimum points)

#### Run & Compare
- **Single Algorithm:** Run one algorithm with specific parameters
- **Select All:** Compare all algorithms automatically with optimal parameters

#### View Results
- 2D and 3D cluster visualizations
- Cluster distribution histograms
- Evaluation metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
- Elbow curves (K-Means, K-Medoids)
- Dendrograms (AGNES, DIANA)
- Best algorithm recommendation based on composite scores

### 8. **Supervised Learning** (Phase 11)
Train classifiers to predict target classes:

#### Select Target Column
- Choose the column you want to predict (class/label)
- View class distribution and balance

#### Split Data
- Adjust train/test split ratio (default: 80/20)
- View sample counts per class in both sets

#### Choose Algorithm
- **k-NN (k-Nearest Neighbors):** Distance-based classification
  - Configure k (number of neighbors, 1-20)
  - View accuracy and precision across different k values
- **Naive Bayes:** Probabilistic classifier using Bayes' theorem
  - Gaussian distribution for continuous attributes
- **C4.5 Decision Tree:** Tree-based classifier using Gain Ratio
  - Automatically builds tree with optimal splits

#### Run & Compare
- **Single Algorithm:** Train one classifier with specific parameters
- **Select All:** Compare all three classifiers automatically

#### View Results
- Evaluation metrics: Accuracy, Precision, Recall, F-Measure
- Confusion matrices (table and heatmap)
- Performance plots (k-NN: Accuracy vs k, Precision vs k)
- Best algorithm recommendation
- Individual visualizations for each algorithm in comparison mode

---

## 🏗️ Project Structure

```
DM-Project/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker configuration
├── docker-compose.yml                  # Docker Compose setup
├── data_mining_project_tasks.md        # Detailed project roadmap
├── README.md                           # This file
│
└── algorithms/
    ├── unsupervised/                   # Clustering algorithms
    │   ├── interface.py                # Unsupervised learning UI
    │   ├── kmeans.py                   # K-Means implementation
    │   ├── kmedoids.py                 # K-Medoids (PAM) implementation
    │   ├── agnes.py                    # AGNES hierarchical clustering
    │   ├── diana.py                    # DIANA divisive clustering
    │   └── dbscan.py                   # DBSCAN density-based clustering
    │
    └── supervised/                     # Classification algorithms
        ├── interface.py                # Supervised learning UI
        ├── knn.py                      # k-Nearest Neighbors
        ├── naive_bayes.py              # Gaussian Naive Bayes
        └── c45.py                      # C4.5 Decision Tree
```

---

## 🧠 Algorithms Implemented

### Unsupervised Learning (Clustering)

#### 1. **K-Means**
- Manual centroid initialization and updates
- Iterative convergence checking
- Elbow method for optimal k detection
- **Metrics:** Inertia (WCSS), Silhouette Score, Davies-Bouldin, Calinski-Harabasz

#### 2. **K-Medoids (PAM)**
- Medoid-based clustering (more robust to outliers)
- PAM algorithm implementation
- Elbow method visualization
- **Metrics:** Same as K-Means

#### 3. **AGNES (Agglomerative Hierarchical)**
- Bottom-up hierarchical clustering
- Multiple linkage methods: Single, Complete, Average, Ward
- Dendrogram visualization
- **Metrics:** Silhouette Score, Davies-Bouldin, Calinski-Harabasz

#### 4. **DIANA (Divisive Hierarchical)**
- Top-down hierarchical clustering
- Recursive splitting strategy
- Dendrogram visualization
- **Metrics:** Same as AGNES

#### 5. **DBSCAN**
- Density-based clustering
- Automatic noise detection
- Core, Border, and Noise point classification
- **Parameters:** eps (neighborhood radius), MinPts (minimum points)
- **Metrics:** Separate metrics for clustered points and all points

### Supervised Learning (Classification)

#### 1. **k-Nearest Neighbors (k-NN)**
- **4-Step Process (from class material):**
  1. Compute Euclidean distances to all training points
  2. Select k nearest neighbors
  3. Identify classes of k neighbors
  4. Majority vote for final prediction
- Multi-k evaluation (k=1 to k=20)
- **Visualizations:** Accuracy vs k, Precision vs k, Confusion matrix
- **Metrics:** Accuracy, Precision, Recall, F-Measure

#### 2. **Naive Bayes (Gaussian)**
- **Training Phase:**
  - Prior probabilities: P(Ck) = Count(Ck) / Total
  - Conditional probabilities: P(xi | Ck) using Gaussian distribution
- **Prediction Phase:**
  - Posterior = P(Ck) × ∏ P(xi | Ck)
  - Predicted class = argmax(Posterior)
- **Key Assumption:** Conditional independence between attributes
- **Metrics:** Accuracy, Precision, Recall, F-Measure, Confusion matrix

#### 3. **C4.5 Decision Tree**
- **Training Phase:**
  - Calculate entropy for node purity
  - Compute Information Gain for potential splits
  - Calculate Split Information
  - **Gain Ratio = Information Gain / Split Information** (key C4.5 improvement)
  - Select attribute with maximum Gain Ratio
  - Recursive tree building with stopping criteria
- **Prediction Phase:**
  - Traverse tree from root to leaf
  - Follow branches based on attribute values
- **Metrics:** Accuracy, Precision, Recall, F-Measure, Confusion matrix

---

## 📊 Evaluation Metrics

### Clustering Metrics

- **Silhouette Score (0 to 1, higher is better)**
  - Measures how similar points are to their own cluster vs. other clusters
  - Values near 1 indicate well-separated clusters

- **Davies-Bouldin Index (0 to ∞, lower is better)**
  - Measures average similarity between clusters
  - Lower values indicate better separation

- **Calinski-Harabasz Index (0 to ∞, higher is better)**
  - Ratio of between-cluster to within-cluster dispersion
  - Higher values indicate denser, well-separated clusters

- **Composite Score**
  - Normalized weighted combination of all metrics
  - Used to recommend the best algorithm

### Classification Metrics

- **Accuracy (0 to 1, higher is better)**
  - Proportion of correct predictions
  - Formula: (TP + TN) / Total

- **Precision (0 to 1, higher is better)**
  - Of all positive predictions, how many were correct
  - Formula: TP / (TP + FP)

- **Recall (0 to 1, higher is better)**
  - Of all actual positives, how many were identified
  - Formula: TP / (TP + FN)

- **F-Measure (0 to 1, higher is better)**
  - Harmonic mean of Precision and Recall
  - Formula: 2 × (Precision × Recall) / (Precision + Recall)

- **Composite Score**
  - Average of Accuracy, Precision, Recall, and F-Measure
  - Used to recommend the best algorithm

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **Streamlit** - Interactive web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations
- **scikit-learn** - Only for confusion matrix and validation metrics
- **Docker** - Containerization for reproducibility

---

## 🎯 Academic Compliance

This application strictly follows academic course material:

### ✅ All Algorithms Implemented from Scratch
- No use of sklearn's clustering or classification algorithms
- Manual implementation of distance calculations, centroid/medoid updates, tree building, etc.
- Only sklearn used for: confusion matrix, train_test_split validation, and metric calculations

### ✅ Exact Formulas from Class Material
- **k-NN:** 4-step process with Euclidean distance
- **Naive Bayes:** Prior probabilities, Gaussian likelihood, posterior calculation
- **C4.5:** Entropy, Information Gain, Split Information, **Gain Ratio** (not just Information Gain)
- **Clustering:** All distance metrics, linkage methods, and evaluation metrics match course definitions

### ✅ Comprehensive Documentation
- Code comments explain academic concepts
- UI includes expandable metric explanations
- All formulas are documented

---

## 🐛 Troubleshooting

### Application won't start
- **Docker:** Ensure Docker Desktop is running
- **Python:** Verify Python version with `python --version` (must be 3.9+)
- **Dependencies:** Run `pip install -r requirements.txt` again

### Port already in use
- **Change port:** `streamlit run app.py --server.port 8502`
- **Docker:** `docker run -p 8502:8501 data-mining-app`

### Out of memory errors
- Large datasets may require more RAM
- Try reducing dataset size or using Docker with increased memory allocation

### Visualizations not appearing
- Ensure Plotly is installed: `pip install plotly`
- Clear browser cache and refresh

---

## 📝 License

This project is developed for academic purposes as part of a Data Mining course.

---

## 👨‍💻 Author

**Ramy**  
Data Mining Course Project  
Version 3.0  
December 2025

---

## 🙏 Acknowledgments

- Course instructors for providing detailed algorithm specifications
- Academic material that guided the implementation
- Streamlit community for excellent documentation

---

## 📧 Support

For issues or questions about the application:
1. Check the troubleshooting section above
2. Review the detailed task plan in `data_mining_project_tasks.md`
3. Verify your setup matches the prerequisites

---

**✨ Enjoy exploring your data with rigorous, academically-compliant algorithms!**
