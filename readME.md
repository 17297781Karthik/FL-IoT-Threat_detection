# FL-IoT-Threat Detection

A **Federated Learning-based IoT Threat Detection System** that uses distributed machine learning to detect malware and network attacks in IoT devices while preserving data privacy.

## 🔍 Overview

This project implements a federated learning framework for IoT threat detection using the N-BaIoT dataset. The system trains neural network models across multiple IoT devices without centralizing sensitive data, enabling collaborative threat detection while maintaining privacy.

## 🏗️ Architecture

```
FL-IoT-Threat Detection/
├── 📁 Core Components
│   ├── server.py                    # Federated learning server (Flower)
│   ├── client1.py                   # FL client for device 1
│   ├── client2.py                   # FL client for device 2
│   └── model.py                     # Neural network architecture
├── 📁 Real-time Detection
│   └── RealTimeService/
│       ├── realtime_pipeline.py     # Real-time threat detection pipeline
│       ├── threat_predictor.py      # ML-based threat prediction
│       ├── feature_extractor.py     # Network feature extraction
│       └── monitor.py               # Continuous monitoring service
├── 📁 Analysis & Training
│   ├── analysis.ipynb               # Data analysis and exploration
│   ├── reduucedTrain.ipynb          # Training with reduced dataset
│   ├── devicewise_analysis.ipynb    # Device-specific analysis
│   └── federated_learning.py       # Custom FL implementation
├── 📁 Data Simulation
│   └── samplePackets/
│       └── packet_simulator.py      # IoT traffic simulation
└── 📁 Results & Models
    ├── SavedGlobalModel/            # Trained federated models
    ├── Results/                     # Performance metrics & plots
    └── Logs/                        # Training and evaluation logs
```

## ✨ Features

### 🔐 **Federated Learning**
- **Privacy-Preserving**: Train models without sharing raw data
- **Distributed Training**: Multiple IoT devices collaborate in learning
- **Flower Framework**: Production-ready federated learning implementation

### 🛡️ **Threat Detection**
- **Multi-class Classification**: Detects various IoT malware types
- **Real-time Monitoring**: Continuous network traffic analysis
- **Attack Types**: Gafgyt, Mirai, and other IoT botnets

### 📊 **Advanced Analytics**
- **Network Feature Extraction**: 115+ statistical network features
- **Performance Visualization**: Confusion matrices, ROC curves
- **Device-wise Analysis**: Individual device behavior patterns

### 🔧 **Real-time Pipeline**
- **PCAP Processing**: Live network packet analysis
- **Feature Engineering**: Automated feature extraction from traffic
- **Threat Classification**: ML-based real-time threat scoring

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### 1. **Federated Learning Setup**

**Start the FL Server:**
```bash
python server.py
```

**Run FL Clients (in separate terminals):**
```bash
# Terminal 1
python client1.py

# Terminal 2  
python client2.py
```

### 2. **Real-time Threat Detection**

**Setup the pipeline:**
```bash
cd RealTimeService
python setup.py
```

**Start monitoring:**
```bash
python monitor.py --pcap-dir /path/to/pcap/files
```

### 3. **Data Analysis**

**Explore the Jupyter notebooks:**
```bash
jupyter notebook
# Open: analysis.ipynb, reduucedTrain.ipynb, devicewise_analysis.ipynb
```

## 📊 Dataset

The project uses the **N-BaIoT dataset** containing network traffic from 9 IoT devices under various attack scenarios:

### **Attack Types:**
- **Mirai**: ack, scan, syn, udp, udpplain
- **Gafgyt**: combo, junk, scan, tcp, udp
- **Benign**: Normal IoT device traffic

### **IoT Devices:**
- Danmini Doorbell, Ecobee Thermostat, Ennio Doorbell
- Philips Baby Monitor, Provision Security Camera
- Samsung SNH Camera, SimpleHome Security Camera
- And more...

## 🔧 Configuration

### **Model Configuration (`model.py`)**
```python
# Neural Network Architecture
- Input Layer: 115 features (network statistics)
- Hidden Layer 1: 128 neurons + ReLU
- Hidden Layer 2: 64 neurons + ReLU  
- Output Layer: 10 classes (attack types)
```

### **FL Server Configuration (`server.py`)**
```python
# Federated Learning Parameters
- Rounds: 4 (configurable)
- Min Clients: 2
- Strategy: FedAvg (Federated Averaging)
- Evaluation: Accuracy, Loss metrics
```

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification performance
- **Precision/Recall**: Per-class detection quality
- **F1-Score**: Balanced performance measure
- **Confusion Matrix**: Detailed classification breakdown
- **ROC Curves**: True/False positive trade-offs

## 🛠️ Advanced Usage

### **Custom Data Pipeline**
```bash
# Generate synthetic IoT traffic
python test_simulator.py

# Extract features from PCAP files
cd RealTimeService
python feature_extractor.py --input traffic.pcap --output features.csv
```

### **Model Evaluation**
```bash
# Comprehensive model testing
python test.py

# Generate evaluation reports
# Results saved to Results/ directory
```

### **Real-time Integration**
```python
# Example: Integrate with existing security systems
from RealTimeService.threat_predictor import IoTThreatPredictor

predictor = IoTThreatPredictor("SavedGlobalModel/final_model.pth")
threat_score = predictor.predict_pcap("network_traffic.pcap")
```

## 🔬 Research Applications

This project supports research in:

- **Federated Learning for IoT Security**
- **Privacy-Preserving Machine Learning**
- **Real-time Network Threat Detection**
- **IoT Botnet Analysis**
- **Distributed AI for Edge Computing**

## 📁 Key Files Description

| File/Directory | Description |
|----------------|-------------|
| `server.py` | Flower-based federated learning server |
| `client1.py`, `client2.py` | FL clients for different IoT devices |
| `model.py` | Neural network architecture definition |
| `federated_learning.py` | Custom FL implementation with detailed logging |
| `RealTimeService/` | Complete real-time threat detection pipeline |
| `analysis.ipynb` | Comprehensive data analysis and visualization |
| `reduucedTrain.ipynb` | Training with reduced dataset for faster development |
| `test.py` | Model evaluation and performance testing |
| `requirements.txt` | Python dependencies |

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Commit changes**: `git commit -am 'Add new feature'`
4. **Push to branch**: `git push origin feature/new-feature`
5. **Submit a Pull Request**



## 🔗 References

- **N-BaIoT Dataset**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Detection_of_IoT_botnet_attacks_N_BaIoT)
- **Flower Framework**: [https://flower.dev/](https://flower.dev/)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)

## 📧 Contact

For questions, issues, or collaborations, please open an issue or contact the project maintainer.

---

**⭐ Star this repository if you find it useful for your IoT security research!**
