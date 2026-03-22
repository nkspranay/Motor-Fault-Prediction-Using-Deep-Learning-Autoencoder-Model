# 🔧 Motor Fault Prediction Using Deep Learning Autoencoder Model

A deep learning-based system for detecting motor faults using an autoencoder. The model learns normal behavior from sensor data and identifies anomalies based on reconstruction error.


## 🚀 Features
- Autoencoder-based anomaly detection
- No labeled fault data required
- Real-time prediction via API
- Lightweight and scalable design


## 🧠 Tech Stack
- Python  
- PyTorch  
- FastAPI / Flask  
- NumPy, Pandas  


## 📂 Project Structure

├── api/ # Backend API
├── model/ # Model files (excluded from Git)
├── data/ # Dataset (optional)
├── notebooks/ # Experiments / training
├── requirements.txt # Dependencies
└── README.md




## ⚙️ Installation & Setup

### 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/Motor-Fault-Prediction-Using-Deep-Learning-Autoencoder-Model.git
cd Motor-Fault-Prediction-Using-Deep-Learning-Autoencoder-Model
2. Create virtual environment
python -m venv venv
3. Activate environment

Windows:

venv\Scripts\activate

Mac/Linux:

source venv/bin/activate
4. Install dependencies
pip install -r requirements.txt
▶️ Running the Project
Start the API server

If using FastAPI:

uvicorn api.main:app --reload

If using Flask:

python api/app.py
🔍 How Prediction Works
Input sensor data is sent to the API
Data is passed through the autoencoder
Reconstruction error is calculated
If error > threshold → Fault detected
⚠️ Notes
model.pth is not included due to size
You can:
Train your own model
OR use dummy predictions for testing
💡 Use Cases
Predictive maintenance
Industrial motor monitoring
IoT-based fault detection systems