# DDoS Attack Detection System

## Project Summary

Our AI system aims to solve the problem of identifying network attacks using DDoS network attack logs. Our model will take network packet features, such as packet IDs, types, size, rate, and duration, etc as the input, and it will output a binary variable of whether it is safe or unsafe. We will build a machine learning classifier using supervised learning techniques, such as decision trees, KNN, and neural networks, as our main model.

## External Tools & Libraries

### Machine Learning Frameworks
- Scikit-learn
- PyTorch

### Data Processing Libraries
- Pandas
- NumPy
- SciPy

## Dataset

**DDoS Attack Network Logs Dataset**, JacobVS (Kaggle, 2022)  
https://www.kaggle.com/datasets/jacobvs/ddos-attack-network-logs

The balanced dataset used for preprocessing and training is stored in `sample-data.csv` on our GitHub.

## Required Software & Versions

- Python 3.11
- scikit-learn
- SciPy
- PyTorch
- NumPy
- pandas
- matplotlib
- seaborn

## Installation & Setup Instructions

### Step 1: Install Dependencies
```bash
pip install scikit-learn torch pandas numpy scipy matplotlib seaborn
```

### Step 2: File Structure Setup
Create the following directory structure:
```
project/
├── data/
│   └── sample-data.csv
├── script.py
├── decision_tree.py
├── knn.py
└── neural_network.py
```

### Step 3: Execution Order

1. **Run preprocessing first:** Execute `python script.py` to generate `X_processed.npy` and `y_processed.npy` files

2. **Run individual models:** Execute any of the following:
   - `python decision_tree.py` for decision tree model
   - `python knn.py` for K-nearest neighbors model
   - `python neural_network.py` for neural network model

## Expected Outputs

- **Preprocessing generates:** `X_processed.npy`, `y_processed.npy`
- **Models output:** Performance metrics, accuracy scores, and feature importance analysis

### Runtime
- KNN: ~25 minutes
- Decision Tree: ~5 minutes  
- Neural Network: ~10 minutes

## Open-Source Code Attribution

- [Machine Learning/Label Encoding Across Multiple Columns in Scikit-Learn](https://www.geeksforgeeks.org/machine-learning/label-encoding-across-multiple-columns-in-scikit-learn/)
- [How to Automate Data Cleaning in Python? - GeeksforGeeks](https://www.geeksforgeeks.org/how-to-automate-data-cleaning-in-python/)

## Platform Dependencies

The system requires Python 3.7+ with standard scientific computing packages. Installation dependencies include:

- **scikit-learn** for machine learning algorithms
- **torch** for neural network implementation
- **pandas** and **numpy** for data processing
- **scipy** for extended file format support