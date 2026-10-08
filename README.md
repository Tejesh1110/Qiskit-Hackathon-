# Qiskit Fall Fest 2026 — Industry Track I3: Fraud Detector

**Repository:** [https://github.com/Tejesh1110/Qiskit-Hackathon-](https://github.com/Tejesh1110/Qiskit-Hackathon-)

A hybrid Quantum-Classical Machine Learning study evaluating a **Quantum Kernel Classifier (QSVC)** against a **Classical Support Vector Machine (SVM)** baseline for financial transaction fraud detection using **Qiskit Machine Learning**.

---

## 1. Project Architecture

The pipeline strictly adheres to the requested challenge architecture:

```text
                 FRAUD DETECTOR
                       │
              Transaction Dataset
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
   Data Preprocessing        Feature Selection
          │                         │
          └────────────┬────────────┘
                       ↓
                Train/Test Split
                       │
              ┌────────┴────────┐
              ↓                 ↓
        Classical SVM          QSVC
              │                 │
              ↓                 ↓
        Predictions        Predictions
              │                 │
              └────────┬────────┘
                       ↓
            Precision / Recall / F1
                       │
                       ↓
               Fair Comparison
```

---

## 2. Methodology & Experimental Design

### A. Synthetic Dataset (`data/transactions.csv`)
- **Size**: 125 reproducible synthetic transactions (`random_state=42`).
- **Features Included**:
  - `transaction_id`: Unique identifier (e.g., `TXN_10001`).
  - `amount`: Monetary value in USD.
  - `distance_from_home`: Distance from cardholder's home base (km).
  - `time_diff_hours`: Elapsed time since the previous transaction.
  - `is_online_order`: Digital checkout flag (0 or 1).
  - `is_fraud`: Ground-truth binary target (`0 = Legitimate`, `1 = Fraud`).
- **Class Imbalance**: 100 Legitimate (80.0%) vs. 25 Fraudulent (20.0%).
- **Privacy**: No personal or confidential data is used.

### B. Preprocessing & Feature Selection
- Selected **exactly 2 features** to map onto a compact 2-qubit quantum register:
  - `amount` (Monetary scale)
  - `distance_from_home` (Geographical anomaly indicator)
- Normalized using `MinMaxScaler` into $[0, \pi]$ for quantum angle encoding in parameterized gates.

### C. Models Compared
1. **Classical SVM Baseline**:
   - `sklearn.svm.SVC` with Radial Basis Function (`rbf`) kernel, $C = 1.0$, `random_state=42`.
2. **Quantum QSVC**:
   - Built using **Qiskit Machine Learning** (`qiskit-machine-learning`).
   - Feature Map: 2-qubit `ZZFeatureMap(feature_dimension=2, reps=1, entanglement='linear')`.
   - Quantum Kernel: `FidelityQuantumKernel` using statevector overlap fidelity $|\langle \psi(x_i) | \psi(x_j) \rangle|^2$.
   - Classifier: `QSVC(quantum_kernel=quantum_kernel)`.

---

## 3. Experimental Results

### Experiment 1: Single Stratified 80/20 Split (Initial Demonstration)
- **Training Set**: 100 samples (80 legitimate, 20 fraud)
- **Test Set**: 25 samples (20 legitimate, 5 fraud)

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Classical SVM (RBF)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Quantum QSVC (ZZFeatureMap)** | **0.8800** | **1.0000** | **0.4000** | **0.5714** |

#### Why a Single Split is Fragile:
In the single 80/20 test set, there are **only 5 fraudulent transactions**. A single false negative causes a **20.0% swing** in Recall ($\Delta = 1/5$). A single partition is therefore vulnerable to random sampling luck.

---

### Experiment 2: Robust Stratified 5-Fold Cross-Validation (Primary Evaluation)
To ensure experimental fairness and statistical rigor, both models were evaluated across **identical folds** using 5-Fold Stratified Cross-Validation.

To make 5-fold cross-validation computationally efficient on the quantum statevector simulator, the full $125 \times 125$ quantum Gram matrix $K(X, X)$ was evaluated once, and exact sub-matrices $K_{train, train}$ and $K_{test, train}$ were passed to each fold without data leakage.

#### Fold-by-Fold Breakdown:
| Fold | SVM Acc | SVM Prec | SVM Rec | SVM F1 | QSVC Acc | QSVC Prec | QSVC Rec | QSVC F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.9600 | 1.0000 | 0.8000 | 0.8889 | 0.8400 | 1.0000 | 0.2000 | 0.3333 |
| **2** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8400 | 1.0000 | 0.2000 | 0.3333 |
| **3** | 0.9600 | 1.0000 | 0.8000 | 0.8889 | 0.8400 | 1.0000 | 0.2000 | 0.3333 |
| **4** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8400 | 0.6000 | 0.6000 | 0.6000 |
| **5** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8800 | 1.0000 | 0.4000 | 0.5714 |

#### Summary (Mean ± Standard Deviation):
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Classical SVM (RBF)** | **0.9840 ± 0.0196** | **1.0000 ± 0.0000** | **0.9200 ± 0.0980** | **0.9556 ± 0.0544** |
| **Quantum QSVC (ZZFeatureMap)** | **0.8480 ± 0.0160** | **0.9200 ± 0.1600** | **0.3200 ± 0.1600** | **0.4343 ± 0.1240** |

---

## 4. Key Scientific Insights

1. **Precision Stability**: Both Classical SVM ($1.0000$) and Quantum QSVC ($0.9200 \pm 0.1600$) achieved high precision, demonstrating that quantum kernel decision boundaries rarely generate false alarms.
2. **Recall Discrepancy**: The 2-qubit QSVC is conservative in detecting fraud (Recall: $0.3200 \pm 0.1600$), whereas Classical SVM achieves $0.9200 \pm 0.0980$. In the $ZZFeatureMap$, quantum state overlaps oscillate periodically with input features ($\cos(x_i - x_j)$), which creates tight, localized decision boundaries around fraud clusters rather than broad monotonic decision regions.
3. **Statistical Validity**: 5-fold cross-validation tested all 25 fraud cases across all folds, eliminating single-split bias and revealing the true standard deviations.

---

## 5. Limitations & Strict Disclaimer

> [!IMPORTANT]
> **No Quantum Advantage**: We strictly do **not** claim quantum advantage.

- **Classical Superiority on Tabular Data**: Classical SVM with an RBF kernel outperforms the 2-qubit QSVC across all metrics, with significantly lower computational overhead.
- **Simulator vs. Physical QPUs**: The QSVC was simulated on an ideal, noiseless statevector simulator. Physical NISQ QPUs would introduce decoherence, gate infidelity, and shot noise.
- **Dimensionality Scaling**: Scaling quantum kernels to higher feature dimensions and large datasets incurs an $O(N^2)$ circuit evaluation cost.

---

## 6. Project Files

```text
Qiskit Hackathon/
├── data/
│   └── transactions.csv       # Synthetic transaction dataset (125 rows)
├── fraud_detector.ipynb       # Fully executed notebook (Sections A-K, with 5-Fold CV)
├── requirements.txt           # Project dependencies
└── README.md                  # Complete documentation and evaluation results
```

---

## 7. How to Run

```bash
pip install -r requirements.txt
python -m notebook fraud_detector.ipynb
```
