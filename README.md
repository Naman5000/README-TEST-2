# 🌟 MediAlert: AI-Powered Anomaly Detection for Smarter Healthcare Insights 🌟  

**🎓 Internship Program:** Infosys Springboard  
**📊 Dataset:** [Healthcare Providers Data](https://www.kaggle.com/datasets/tamilsel/healthcare-providers-data)  

---

## 🎯 Objective  
Leverage machine learning and deep learning techniques to effectively identify and analyze fraudulent healthcare transactions.  

---

## 🏥 Overview  
Healthcare fraud is a significant challenge, diverting resources from essential medical services. This project aims to detect billing-related anomalies using domain knowledge and cutting-edge AI techniques. We employ:  
- **🌲 Isolation Forest** for unsupervised anomaly detection.  
- **🧠 Autoencoder models** for deep learning-based anomaly detection.  

---

## 🛠️ Workflow  

The project is divided into multiple modules:  

1. **🔍 Data Understanding and Exploratory Data Analysis (EDA):**  
   - Performed univariate and bivariate analyses to explore and visualize key features.  
   - Identified patterns and trends using domain-specific insights.  

2. **🧹 Data Preprocessing:**  
   - Treated missing values and standardized the data using `StandardScaler`.  
   - Encoded categorical values as required and split the data into training and validation sets.  
   - Created normalized datasets suitable for both traditional machine learning models and deep learning approaches.  

3. **🚨 Anomaly Detection:**  
   - **🌲 Isolation Forest:**  
     - Trained an Isolation Forest model to isolate and detect approximately 5% anomalous transactions.  
     - Utilized domain knowledge to fine-tune hyperparameters and validate results.  
   - **🧠 Autoencoder:**  
     - Developed an Autoencoder deep learning model with mixed precision training for optimized GPU utilization.  
     - Designed and trained the model using TensorFlow, incorporating advanced callbacks like `ReduceLROnPlateau`, `EarlyStopping`, and `ModelCheckpoint`.  
     - Detected anomalies based on reconstruction errors, defining an anomaly threshold at the 95th percentile.  

4. **📊 Visualization:**  
   - Plotted training and validation losses to evaluate Autoencoder performance.  
   - Visualized the reconstruction error distribution, comparing normal and anomalous transactions.  

---

## ✨ Key Implementation Highlights  

### 🌲 Isolation Forest  
- **Algorithm:** Isolation Forest  
- **Purpose:** Anomaly detection by isolating data points that deviate significantly from the norm.  
- **Outcome:** Successfully detected ~5% of transactions as anomalies.  

### 🧠 Autoencoder Model  
- **Architecture:**  
  - Input Layer: Equal to the number of features.  
  - Encoders: Three layers with dimensional reductions.  
  - Decoders: Three layers for reconstructing inputs.  
- **Loss Function:** Mean Squared Error (MSE).  
- **Training Strategy:**  
  - Mixed precision training enabled for faster computation.  
  - Optimized data pipelines using TensorFlow's `tf.data` API.  
- **Callbacks:** Early stopping, learning rate adjustment, and model checkpointing.  
- **Outcome:** Detected anomalies based on reconstruction error, with results validated using domain expertise.  

---

### 📊 Visualizations  

#### 📉 Training and Validation Loss  
![Training Loss](https://github.com/user-attachments/assets/d986c973-1875-4490-8be5-6644b054ba4d)  

#### 📈 Log-Transformed Distribution of Charges  
![Distribution](![Modelling Plot 2](https://github.com/user-attachments/assets/e26f96ab-a282-4585-bdc4-9556d6c9e255))  

---

## 🌟 Results and Insights  

- ✅ Successfully isolated ~5% of healthcare transactions as anomalies using both Isolation Forest and Autoencoder models.  
- ✅ Deep learning-based Autoencoder demonstrated improved sensitivity in capturing subtle anomalies compared to traditional approaches.  
- ✅ The analysis highlights the effectiveness of AI-driven anomaly detection in combating healthcare fraud.  

---

## 🛠️ Tools and Technologies  

- **Programming Language:** Python 🐍  
- **Libraries:** TensorFlow, Scikit-learn, Pandas, Matplotlib, NumPy 📚  
- **Models:** Isolation Forest 🌲, Autoencoder 🧠  
- **Hardware:** Mixed precision training on GPU ⚡  

---

## 🚀 Future Scope  

- 🔗 Integrate results with a real-time anomaly detection pipeline for healthcare providers.  
- 🔍 Explore additional algorithms like Variational Autoencoders and Generative Adversarial Networks for anomaly detection.  
- 🌍 Expand the dataset for enhanced generalizability across healthcare domains.  

---
