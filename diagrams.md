# Federated Learning IoT Malware Detection System - UML Diagrams

## Table of Contents
- [Use Case Diagram](#use-case-diagram)
- [Class Diagram](#class-diagram)
- [Sequence Diagram](#sequence-diagram)
- [Component Diagram](#component-diagram)
- [Activity Diagram](#activity-diagram)
- [Deployment Diagram](#deployment-diagram)

---

## Use Case Diagram

<div style="background-color: #f8f9fa; border: 2px solid #e9ecef; border-radius: 8px; padding: 20px; margin: 10px 0; overflow-x: auto; font-family: monospace; white-space: pre;">

### Complete FL IoT Malware Detection System with Real-Time Prediction

```
                 FL IoT Malware Detection System
    
    ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
    │                             STAKEHOLDERS & USE CASES                                       │
    │                                                                                             │
    │  IoT Device Owner     System Administrator    Security Analyst      Network Admin          │
    │        │                       │                      │                      │             │
    │        │                       │                      │                      │             │
    │        ▼                       ▼                      ▼                      ▼             │
    │   ┌─────────────┐       ┌─────────────────┐    ┌─────────────┐      ┌─────────────────┐   │
    │   │   Monitor   │       │  Configure FL   │    │   Monitor   │      │   Configure     │   │
    │   │   Device    │       │    Server       │    │   Security  │      │   Network       │   │
    │   │   Status    │       │                 │    │   Alerts    │      │   Capture       │   │
    │   └─────────────┘       └─────────────────┘    └─────────────┘      └─────────────────┘   │
    │        │                       │                      │                      │             │
    │        │                       │                      │                      │             │
    │        ▼                       ▼                      ▼                      ▼             │
    │   ┌─────────────┐       ┌─────────────────┐    ┌─────────────┐      ┌─────────────────┐   │
    │   │ Participate │       │   Aggregate     │    │ Investigate │      │   Deploy        │   │
    │   │  in FL      │◄─────►│    Models       │    │   Threats   │      │   Real-Time     │   │
    │   │  Training   │       │   (FedAvg)      │    │             │      │   Detection     │   │
    │   └─────────────┘       └─────────────────┘    └─────────────┘      └─────────────────┘   │
    │        │                       │                      │                      │             │
    │        │                       │                      │                      │             │
    │        ▼                       ▼                      ▼                      ▼             │
    │   ┌─────────────┐       ┌─────────────────┐    ┌─────────────┐      ┌─────────────────┐   │
    │   │   Train     │       │  Save Global    │    │   Respond   │      │   Extract       │   │
    │   │   Local     │       │     Model       │    │  to Attacks │      │   Features      │   │
    │   │   Model     │       │                 │    │             │      │  from Traffic   │   │
    │   └─────────────┘       └─────────────────┘    └─────────────┘      └─────────────────┘   │
    │        │                       │                      │                      │             │
    │        │                       │                      │                      │             │
    │        ▼                       ▼                      ▼                      ▼             │
    │   ┌─────────────┐       ┌─────────────────┐    ┌─────────────┐      ┌─────────────────┐   │
    │   │   Detect    │       │   Monitor FL    │    │    Block    │      │   Real-Time     │   │
    │   │   Local     │◄─────►│   Performance   │    │  Malicious  │◄────►│   Malware       │   │
    │   │  Malware    │       │                 │    │   Traffic   │      │  Prediction     │   │
    │   └─────────────┘       └─────────────────┘    └─────────────┘      └─────────────────┘   │
    │        │                       │                      │                      │             │
    │        │                       │                      │                      │             │
    │        ▼                       ▼                      ▼                      ▼             │
    │   ┌─────────────┐       ┌─────────────────┐    ┌─────────────┐      ┌─────────────────┐   │
    │   │  Generate   │       │  Update Model   │    │  Generate   │      │  Send Alerts    │   │
    │   │   Local     │       │   Versions      │    │   Security  │      │   & Reports     │   │
    │   │   Reports   │       │                 │    │   Reports   │      │                 │   │
    │   └─────────────┘       └─────────────────┘    └─────────────┘      └─────────────────┘   │
    │                                                                                             │
    └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## Class Diagram

<div style="background-color: #f8f9fa; border: 2px solid #e9ecef; border-radius: 8px; padding: 20px; margin: 10px 0; overflow-x: auto; font-family: monospace; white-space: pre;">

### Complete Class Diagram (Training + Prediction System)

```
    ┌─────────────────────────────┐      ┌─────────────────────────────┐      ┌─────────────────────────────┐
    │      NeuralNetwork          │      │    NetworkCapture           │      │  RealTimeFeatureExtractor   │
    ├─────────────────────────────┤      ├─────────────────────────────┤      ├─────────────────────────────┤
    │ - fc1: Linear               │      │ - interface: str            │      │ - window_size: int          │
    │ - fc2: Linear               │      │ - packet_buffer: Queue      │      │ - packet_windows: dict      │
    │ - fc3: Linear               │      │ - is_capturing: bool        │      │ - feature_cache: dict       │
    │ - relu: ReLU                │      ├─────────────────────────────┤      ├─────────────────────────────┤
    ├─────────────────────────────┤      │ + start_capture()           │      │ + extract_features()        │
    │ + __init__(input_size, classes)│   │ + stop_capture()            │      │ + calculate_statistics()    │
    │ + forward(x): Tensor        │      │ + get_packets(batch_size)   │      │ + _sliding_window()         │
    └─────────────────────────────┘      │ + packet_handler(packet)    │      └─────────────────────────────┘
                    △                    └─────────────────────────────┘                      △
                    │                                    │                                    │
                    │ uses                               │ uses                               │ uses
                    │                                    │                                    │
    ┌─────────────────────────────┐      ┌─────────────────────────────┐      ┌─────────────────────────────┐
    │      FlowerClient           │      │ RealTimePredictionService   │      │       AlertSystem           │
    ├─────────────────────────────┤      ├─────────────────────────────┤      ├─────────────────────────────┤
    │ - net: NeuralNetwork        │      │ - model: NeuralNetwork      │      │ - email_config: dict        │
    │ - train_loader: DataLoader  │      │ - feature_extractor: obj    │      │ - webhook_url: str          │
    │ - test_loader: DataLoader   │      │ - network_capture: obj      │      │ - siem_endpoint: str        │
    │ - criterion: Loss           │      │ - threshold: float          │      ├─────────────────────────────┤
    │ - optimizer: Optimizer      │      │ - attack_types: dict        │      │ + send_email_alert()        │
    ├─────────────────────────────┤      │ - is_running: bool          │      │ + send_webhook_alert()      │
    │ + fit(parameters, config)   │      ├─────────────────────────────┤      │ + log_to_siem()             │
    │ + evaluate(parameters, config)│     │ + _load_model(path)         │      └─────────────────────────────┘
    │ + get_parameters()          │      │ + predict_malware(features) │                      △
    │ + set_parameters(params)    │      │ + start_real_time_detection()│                     │ uses
    └─────────────────────────────┘      │ + _handle_prediction_result()│                     │
                    △                    │ + _trigger_response()       │                      │
                    │ coordinates        │ + stop_detection()          │                      │
                    │                    └─────────────────────────────┘                      │
    ┌─────────────────────────────┐                      │                                    │
    │    SaveModelStrategy        │                      │ triggers                           │
    ├─────────────────────────────┤                      │                                    │
    │ - model: NeuralNetwork      │                      ▼                                    │
    ├─────────────────────────────┤      ┌─────────────────────────────┐                      │
    │ + aggregate_fit()           │      │     TrafficSimulator        │                      │
    │ + aggregate_evaluate()      │      ├─────────────────────────────┤                      │
    │ + save_model()              │      │ - attack_patterns: dict     │                      │
    │ + load_model()              │      │ - normal_patterns: dict     │                      │
    └─────────────────────────────┘      ├─────────────────────────────┤                      │
                    △                    │ + generate_normal_traffic() │                      │
                    │ extends            │ + generate_attack_traffic() │                      │
                    │                    │ + inject_mirai_pattern()    │                      │
    ┌─────────────────────────────┐      │ + save_to_pcap()            │                      │
    │         FedAvg              │      └─────────────────────────────┘                      │
    ├─────────────────────────────┤                      │                                    │
    │ + aggregate_fit()           │                      │ generates                          │
    │ + aggregate_evaluate()      │                      ▼                                    │
    └─────────────────────────────┘      ┌─────────────────────────────┐                      │
                                         │      DataPipeline           │◄─────────────────────┘
                                         ├─────────────────────────────┤
                                         │ - input_path: str           │
                                         │ - output_path: str          │
                                         │ - quality_checks: list      │
                                         ├─────────────────────────────┤
                                         │ + ingest_data()             │
                                         │ + preprocess_data()         │
                                         │ + validate_quality()        │
                                         │ + distribute_to_clients()   │
                                         └─────────────────────────────┘
```

</div>

---

## Sequence Diagram

<div style="background-color: #f8f9fa; border: 2px solid #e9ecef; border-radius: 8px; padding: 20px; margin: 10px 0; overflow-x: auto; font-family: monospace; white-space: pre;">

### Complete Sequence Diagram (FL Training + Real-Time Prediction)

```
FL Server    FL Client    Network     Feature      Prediction   Alert      Security
             (Device)     Capture     Extractor    Service      System     Analyst
    │           │           │            │            │           │           │
    │ ┌─────────────────────────────────────────────────────────────────────────────┐
    │ │                    PHASE 1: FL Training                                    │
    │ └─────────────────────────────────────────────────────────────────────────────┘
    │           │           │            │            │           │           │
    │───────────►           │            │            │           │           │ Send Global Model
    │           │───────────►            │            │           │           │ Start Local Training
    │           │           │            │            │           │           │
    │           │ ┌─────────┐            │            │           │           │
    │           │ │ Local   │            │            │           │           │
    │           │ │Training │            │            │           │           │
    │           │ └─────────┘            │            │           │           │
    │           │           │            │            │           │           │
    │◄──────────            │            │            │           │           │ Send Updates
    │           │           │            │            │           │           │
    │ ┌─────────┐           │            │            │           │           │
    │ │Aggregate│           │            │            │           │           │
    │ │& Update │           │            │            │           │           │
    │ └─────────┘           │            │            │           │           │
    │           │           │            │            │           │           │
    │───────────►           │            │            │           │           │ Send Updated Model
    │           │           │            │            │           │           │
    │ ┌─────────────────────────────────────────────────────────────────────────────┐
    │ │                 PHASE 2: Real-Time Detection                               │
    │ └─────────────────────────────────────────────────────────────────────────────┘
    │           │           │            │            │           │           │
    │───────────►           │            │            │           │           │ Deploy Trained Model
    │           │───────────►            │            │           │           │ Start Real-Time Capture
    │           │           │────────────►            │           │           │ Capture Network Traffic
    │           │           │            │────────────►           │           │ Extract Features
    │           │           │            │            │───────────►           │ Predict Malware
    │           │           │            │            │           │           │
    │           │           │            │            │ ┌─────────┐           │
    │           │           │            │            │ │ Malware │           │
    │           │           │            │            │ │Detected!│           │
    │           │           │            │            │ └─────────┘           │
    │           │           │            │            │           │           │
    │           │           │            │            │───────────►           │ Send Alert
    │           │           │            │            │           │───────────► Alert: Mirai Detected
    │           │           │            │            │           │           │
    │           │◄──────────────────────────────────────────────────────────────────── Block Malicious IP
    │           │           │            │            │           │           │
    │◄──────────            │            │            │           │           │ Log Incident
    │           │           │            │            │           │           │
    │ ┌─────────────────────────────────────────────────────────────────────────────┐
    │ │              PHASE 3: Continuous Learning                                  │
    │ └─────────────────────────────────────────────────────────────────────────────┘
    │           │           │            │            │           │           │
    │───────────►           │            │            │           │           │ Request New Training
    │           │───────────►            │            │           │           │ Collect New Attack Data
    │           │           │            │            │           │           │
    │           ... (Repeat FL Training with Updated Attack Patterns) ...      │
```

</div>

---

## Component Diagram

<div style="background-color: #f8f9fa; border: 2px solid #e9ecef; border-radius: 8px; padding: 20px; margin: 10px 0; overflow-x: auto; font-family: monospace; white-space: pre;">

### Complete Component Diagram (Training + Prediction System)

```
    ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
    │                      Complete FL IoT Malware Detection System                              │
    │                                                                                             │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
    │  │                         FL Training Subsystem                                      │   │
    │  │                                                                                     │   │
    │  │  ┌─────────────────┐              ┌─────────────────────────────────────────────┐   │   │
    │  │  │   FL Server     │              │              FL Clients                     │   │   │
    │  │  │                 │              │                                             │   │   │
    │  │  │ ┌─────────────┐ │              │ ┌─────────────┐ ┌─────────────────────────┐ │   │   │
    │  │  │ │   FedAvg    │ │              │ │   Local     │ │    Data Processing      │ │   │   │
    │  │  │ │  Strategy   │ │              │ │  Training   │ │       Component         │ │   │   │
    │  │  │ └─────────────┘ │              │ │  Component  │ └─────────────────────────┘ │   │   │
    │  │  │                 │              │ └─────────────┘                             │   │   │
    │  │  │ ┌─────────────┐ │              │ ┌─────────────────────────────────────────┐ │   │   │
    │  │  │ │   Model     │ │              │ │         Communication Handler          │ │   │   │
    │  │  │ │  Manager    │ │              │ └─────────────────────────────────────────┘ │   │   │
    │  │  │ └─────────────┘ │              └─────────────────────────────────────────────┘   │   │
    │  │  └─────────────────┘                                                              │   │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                             │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
    │  │                      Real-Time Prediction Subsystem                                │   │
    │  │                                                                                     │   │
    │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │   │
    │  │  │   Network       │  │    Feature      │  │   Prediction    │  │     Alert       │ │   │
    │  │  │   Traffic       │  │   Extraction    │  │    Service      │  │    System       │ │   │
    │  │  │   Capture       │  │    Engine       │  │                 │  │                 │ │   │
    │  │  │                 │  │                 │  │                 │  │                 │ │   │
    │  │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │ │   │
    │  │  │ │  Packet     │ │  │ │ Statistical │ │  │ │   Neural    │ │  │ │   Email     │ │ │   │
    │  │  │ │  Sniffer    │ │  │ │  Feature    │ │  │ │   Network   │ │  │ │   Alerts    │ │ │   │
    │  │  │ └─────────────┘ │  │ │ Calculator  │ │  │ │   Inference │ │  │ └─────────────┘ │ │   │
    │  │  │                 │  │ └─────────────┘ │  │ └─────────────┘ │  │                 │ │   │
    │  │  │ ┌─────────────┐ │  │                 │  │                 │  │ ┌─────────────┐ │ │   │
    │  │  │ │   Buffer    │ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ │  Webhook    │ │ │   │
    │  │  │ │ Management  │ │  │ │   Window    │ │  │ │ Threshold   │ │  │ │ Notifications│ │ │   │
    │  │  │ └─────────────┘ │  │ │ Management  │ │  │ │ Manager     │ │  │ └─────────────┘ │ │   │
    │  │  └─────────────────┘  │ └─────────────┘ │  │ └─────────────┘ │  └─────────────────┘ │   │
    │  │           │           └─────────────────┘           │        └─────────────────┘   │   │
    │  │           └───────────────────┬─────────────────────┘                              │   │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                             │
    │  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
    │  │                         Support Components                                         │   │
    │  │                                                                                     │   │
    │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐ ┌─────────────┐ │   │
    │  │  │   Neural    │ │   Logging   │ │   Security  │ │      Data       │ │   Traffic   │ │   │
    │  │  │   Network   │ │   System    │ │   Module    │ │    Pipeline     │ │  Simulator  │ │   │
    │  │  │   Model     │ │             │ │ (SSL/TLS)   │ │                 │ │             │ │   │
    │  │  │             │ │ ┌─────────┐ │ │             │ │ ┌─────────────┐ │ │ ┌─────────┐ │ │   │
    │  │  │ ┌─────────┐ │ │ │  File   │ │ │ ┌─────────┐ │ │ │   Quality   │ │ │ │ Attack  │ │ │   │
    │  │  │ │ 3-Layer │ │ │ │ Logging │ │ │ │  Cert   │ │ │ │   Checks    │ │ │ │Pattern  │ │ │   │
    │  │  │ │   MLP   │ │ │ └─────────┘ │ │ │ Manager │ │ │ └─────────────┘ │ │ │Generator│ │ │   │
    │  │  │ └─────────┘ │ │             │ │ └─────────┘ │ │                 │ │ └─────────┘ │ │   │
    │  │  └─────────────┘ │ ┌─────────┐ │ │             │ │ ┌─────────────┐ │ │             │ │   │
    │  │                  │ │Console  │ │ │ ┌─────────┐ │ │ │Preprocessing│ │ │ ┌─────────┐ │ │   │
    │  │                  │ │ Output  │ │ │ │Encrypted│ │ │ │   Engine    │ │ │ │  Pcap   │ │ │   │
    │  │                  │ └─────────┘ │ │ │  Comm   │ │ │ └─────────────┘ │ │ │Generator│ │ │   │
    │  │                  └─────────────┘ │ └─────────┘ │ └─────────────────┘ │ └─────────┘ │ │   │
    │  │                                  └─────────────┘                     └─────────────┘ │   │
    │  └─────────────────────────────────────────────────────────────────────────────────────┘   │
    │                                                                                             │
    └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## Activity Diagram

<div style="background-color: #f8f9fa; border: 2px solid #e9ecef; border-radius: 8px; padding: 20px; margin: 10px 0; overflow-x: auto; font-family: monospace; white-space: pre;">

### Complete Activity Diagram (Training + Real-Time Prediction)

```
    Start System
         │
         ▼
    ┌─────────────────┐
    │ Initialize FL   │
    │    Server       │
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │   Wait for      │
    │   Clients       │
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐      No     ┌─────────────────┐
    │ Min Clients     │─────────────►│    Wait for     │
    │ Available?      │              │  More Clients   │
    └─────────────────┘              └─────────────────┘
         │ Yes                              │
         ▼                                  │
    ┌─────────────────┐                     │
    │ Start FL        │                     │
    │ Training        │◄────────────────────┘
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │ Send Global     │
    │ Parameters      │
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │ Clients Train   │
    │ Locally         │
    │ (Parallel)      │
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │ Aggregate       │
    │ Model Updates   │
    │ (FedAvg)        │
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐    No    ┌─────────────────┐
    │ Max Rounds      │─────────►│ Increment       │
    │ Reached?        │          │ Round Counter   │
    └─────────────────┘          └─────────────────┘
         │ Yes                          │
         ▼                              │
    ┌─────────────────┐                 │
    │ Save Final      │                 │
    │ Global Model    │                 │
    └─────────────────┘                 │
         │                              │
         ▼              ◄───────────────┘
    ┌─────────────────┐
    │ Deploy Model    │
    │ for Real-Time   │
    │ Prediction      │
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │ Start Network   │
    │ Traffic Capture │
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │ Extract Features│
    │ from Packets    │
    │ (58 features)   │
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │ Run Malware     │
    │ Prediction      │
    │ (Neural Net)    │
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐    No     ┌─────────────────┐
    │ Malware         │──────────►│ Log Normal      │
    │ Detected?       │           │ Traffic         │
    └─────────────────┘           └─────────────────┘
         │ Yes                           │
         ▼                               │
    ┌─────────────────┐                  │
    │ Trigger Alert   │                  │
    │ System          │                  │
    └─────────────────┘                  │
         │                               │
         ▼                               │
    ┌─────────────────┐                  │
    │ Send Security   │                  │
    │ Notifications   │                  │
    └─────────────────┘                  │
         │                               │
         ▼                               │
    ┌─────────────────┐                  │
    │ Block Malicious │                  │
    │ IP/Update       │                  │
    │ Firewall        │                  │
    └─────────────────┘                  │
         │                               │
         ▼                               │
    ┌─────────────────┐                  │
    │ Log Attack      │                  │
    │ Incident        │                  │
    └─────────────────┘                  │
         │                               │
         ▼                               │
    ┌─────────────────┐                  │
    │ Update Training │                  │
    │ Data with New   │                  │
    │ Attack Pattern  │                  │
    └─────────────────┘                  │
         │                               │
         ▼                               │
    ┌─────────────────┐    No     ┌─────────────────┐
    │ Retrain Model   │──────────►│ Continue        │
    │ Needed?         │           │ Monitoring      │
    └─────────────────┘           └─────────────────┘
         │ Yes                           │
         ▼                               │
    ┌─────────────────┐                  │
    │ Trigger New     │                  │
    │ FL Training     │                  │
    │ Round           │                  │
    └─────────────────┘                  │
         │                               │
         └───────────────────────────────┘
```

</div>

---

## Deployment Diagram

<div style="background-color: #f8f9fa; border: 2px solid #e9ecef; border-radius: 8px; padding: 20px; margin: 10px 0; overflow-x: auto; font-family: monospace; white-space: pre;">

### Complete Deployment Diagram (Training + Production)

```
                    Complete FL IoT Malware Detection Deployment
    
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                              EDGE DEVICES LAYER                                             │
    │                                                                                              │
    │  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐              │
    │  │   IoT Device 1      │    │   IoT Device 2      │    │   IoT Device N      │              │
    │  │                     │    │                     │    │                     │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │              │
    │  │ │ FL Client       │ │    │ │ FL Client       │ │    │ │ FL Client       │ │              │
    │  │ │ Application     │ │    │ │ Application     │ │    │ │ Application     │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │              │
    │  │                     │    │                     │    │                     │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │              │
    │  │ │ Real-Time       │ │    │ │ Real-Time       │ │    │ │ Real-Time       │ │              │
    │  │ │ Prediction      │ │    │ │ Prediction      │ │    │ │ Prediction      │ │              │
    │  │ │ Service         │ │    │ │ Service         │ │    │ │ Service         │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │              │
    │  │                     │    │                     │    │                     │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │              │
    │  │ │ Network Traffic │ │    │ │ Network Traffic │ │    │ │ Network Traffic │ │              │
    │  │ │ Capture Module  │ │    │ │ Capture Module  │ │    │ │ Capture Module  │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │              │
    │  │                     │    │                     │    │                     │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │              │
    │  │ │ Feature         │ │    │ │ Feature         │ │    │ │ Feature         │ │              │
    │  │ │ Extraction      │ │    │ │ Extraction      │ │    │ │ Extraction      │ │              │
    │  │ │ Engine          │ │    │ │ Engine          │ │    │ │ Engine          │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │              │
    │  │                     │    │                     │    │                     │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │              │
    │  │ │ Local Data      │ │    │ │ Local Data      │ │    │ │ Local Data      │ │              │
    │  │ │ Storage         │ │    │ │ Storage         │ │    │ │ Storage         │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │              │
    │  │                     │    │                     │    │                     │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │              │
    │  │ │ Edge Computing  │ │    │ │ Edge Computing  │ │    │ │ Edge Computing  │ │              │
    │  │ │ Resources       │ │    │ │ Resources       │ │    │ │ Resources       │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │              │
    │  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘              │
    │            │                          │                          │                          │
    └────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┘
                 │                          │                          │
                 │           Secure Communication (SSL/TLS)            │
                 │                          │                          │
                 └──────────────────────────┼──────────────────────────┘
                                            │
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                              CENTRAL SERVER LAYER                                           │
    │                                                                                              │
    │                                ┌─────────────▼──────────────┐                               │
    │                                │      Cloud/Server          │                               │
    │                                │                            │                               │
    │                                │ ┌────────────────────────┐ │                               │
    │                                │ │  FL Server             │ │                               │
    │                                │ │  (Federated Learning   │ │                               │
    │                                │ │   Coordinator)         │ │                               │
    │                                │ └────────────────────────┘ │                               │
    │                                │                            │                               │
    │                                │ ┌────────────────────────┐ │                               │
    │                                │ │  Global Model          │ │                               │
    │                                │ │  Repository            │ │                               │
    │                                │ └────────────────────────┘ │                               │
    │                                │                            │                               │
    │                                │ ┌────────────────────────┐ │                               │
    │                                │ │  Traffic Simulator     │ │                               │
    │                                │ │  (Synthetic Data       │ │                               │
    │                                │ │   Generator)           │ │                               │
    │                                │ └────────────────────────┘ │                               │
    │                                │                            │                               │
    │                                │ ┌────────────────────────┐ │                               │
    │                                │ │  Data Pipeline         │ │                               │
    │                                │ │  (Ingestion &          │ │                               │
    │                                │ │   Preprocessing)       │ │                               │
    │                                │ └────────────────────────┘ │                               │
    │                                └────────────────────────────┘                               │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                   │
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                           MONITORING & SECURITY LAYER                                       │
    │                                                                                              │
    │  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐              │
    │  │   Security          │    │   Monitoring        │    │   Alert System      │              │
    │  │   Operations        │    │   Dashboard         │    │                     │              │
    │  │   Center (SOC)      │    │                     │    │ ┌─────────────────┐ │              │
    │  │                     │    │ ┌─────────────────┐ │    │ │ Email Alerts    │ │              │
    │  │ ┌─────────────────┐ │    │ │ Real-Time       │ │    │ └─────────────────┘ │              │
    │  │ │ Incident        │ │    │ │ Metrics         │ │    │                     │              │
    │  │ │ Response        │ │    │ │ (Grafana)       │ │    │ ┌─────────────────┐ │              │
    │  │ │ System          │ │    │ └─────────────────┘ │    │ │ Slack/Teams     │ │              │
    │  │ └─────────────────┘ │    │                     │    │ │ Webhooks        │ │              │
    │  │                     │    │ ┌─────────────────┐ │    │ └─────────────────┘ │              │
    │  │ ┌─────────────────┐ │    │ │ Performance     │ │    │                     │              │
    │  │ │ Firewall        │ │    │ │ Analytics       │ │    │ ┌─────────────────┐ │              │
    │  │ │ Management      │ │    │ │ (Prometheus)    │ │    │ │ SIEM            │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ │ Integration     │ │              │
    │  │                     │    │                     │    │ └─────────────────┘ │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    └─────────────────────┘              │
    │  │ │ Log             │ │    │ │ Training        │ │                                          │
    │  │ │ Management      │ │    │ │ Progress        │ │                                          │
    │  │ │ (ELK Stack)     │ │    │ │ Tracking        │ │                                          │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │                                          │
    │  └─────────────────────┘    └─────────────────────┘                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## Summary

This comprehensive set of UML diagrams provides a complete view of the Federated Learning IoT Malware Detection System, covering both the training phase and the real-time prediction deployment. The diagrams show the system architecture, component interactions, data flow, and deployment infrastructure needed for a production-ready federated learning solution.

### Key System Features Illustrated:

1. **Decentralized Training**: FL clients train locally while sharing only model parameters
2. **Real-Time Detection**: Continuous network monitoring and malware prediction
3. **Security Components**: SSL/TLS encryption, alert systems, and incident response
4. **Scalable Architecture**: Edge computing, cloud coordination, and monitoring infrastructure
5. **Data Pipeline**: Synthetic data generation, feature extraction, and quality assurance

These diagrams can be used for system documentation, stakeholder presentations, and implementation guidance.
    └─────────────────┘                  │
         │                               │
         ▼                               │
    ┌─────────────────┐                  │
    │ Block Malicious │                  │
    │ IP/Update       │                  │
    │ Firewall        │                  │
    └─────────────────┘                  │
         │                               │
         ▼                               │
    ┌─────────────────┐                  │
    │ Log Attack      │                  │
    │ Incident        │                  │
    └─────────────────┘                  │
         │                               │
         ▼                               │
    ┌─────────────────┐                  │
    │ Update Training │                  │
    │ Data with New   │                  │
    │ Attack Pattern  │                  │
    └─────────────────┘                  │
         │                               │
         ▼                               │
    ┌─────────────────┐    No     ┌─────────────────┐
    │ Retrain Model   │──────────►│ Continue        │
    │ Needed?         │           │ Monitoring      │
    └─────────────────┘           └─────────────────┘
         │ Yes                           │
         ▼                               │
    ┌─────────────────┐                  │
    │ Trigger New     │                  │
    │ FL Training     │                  │
    │ Round           │                  │
    └─────────────────┘                  │
         │                               │
         └───────────────────────────────┘
```

## 6. Complete Deployment Diagram (Training + Production)

```
                    Complete FL IoT Malware Detection Deployment
    
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                              EDGE DEVICES LAYER                                             │
    │                                                                                              │
    │  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐              │
    │  │   IoT Device 1      │    │   IoT Device 2      │    │   IoT Device N      │              │
    │  │                     │    │                     │    │                     │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │              │
    │  │ │ FL Client       │ │    │ │ FL Client       │ │    │ │ FL Client       │ │              │
    │  │ │ Application     │ │    │ │ Application     │ │    │ │ Application     │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │              │
    │  │                     │    │                     │    │                     │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │              │
    │  │ │ Real-Time       │ │    │ │ Real-Time       │ │    │ │ Real-Time       │ │              │
    │  │ │ Prediction      │ │    │ │ Prediction      │ │    │ │ Prediction      │ │              │
    │  │ │ Service         │ │    │ │ Service         │ │    │ │ Service         │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │              │
    │  │                     │    │                     │    │                     │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │              │
    │  │ │ Network Traffic │ │    │ │ Network Traffic │ │    │ │ Network Traffic │ │              │
    │  │ │ Capture Module  │ │    │ │ Capture Module  │ │    │ │ Capture Module  │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │              │
    │  │                     │    │                     │    │                     │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │              │
    │  │ │ Feature         │ │    │ │ Feature         │ │    │ │ Feature         │ │              │
    │  │ │ Extraction      │ │    │ │ Extraction      │ │    │ │ Extraction      │ │              │
    │  │ │ Engine          │ │    │ │ Engine          │ │    │ │ Engine          │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │              │
    │  │                     │    │                     │    │                     │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │              │
    │  │ │ Local Data      │ │    │ │ Local Data      │ │    │ │ Local Data      │ │              │
    │  │ │ Storage         │ │    │ │ Storage         │ │    │ │ Storage         │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │              │
    │  │                     │    │                     │    │                     │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │              │
    │  │ │ Edge Computing  │ │    │ │ Edge Computing  │ │    │ │ Edge Computing  │ │              │
    │  │ │ Resources       │ │    │ │ Resources       │ │    │ │ Resources       │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │              │
    │  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘              │
    │            │                          │                          │                          │
    └────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┘
                 │                          │                          │
                 │           Secure Communication (SSL/TLS)            │
                 │                          │                          │
                 └──────────────────────────┼──────────────────────────┘
                                            │
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                              CENTRAL SERVER LAYER                                           │
    │                                                                                              │
    │                                ┌─────────────▼──────────────┐                               │
    │                                │      Cloud/Server          │                               │
    │                                │                            │                               │
    │                                │ ┌────────────────────────┐ │                               │
    │                                │ │  FL Server             │ │                               │
    │                                │ │  (Federated Learning   │ │                               │
    │                                │ │   Coordinator)         │ │                               │
    │                                │ └────────────────────────┘ │                               │
    │                                │                            │                               │
    │                                │ ┌────────────────────────┐ │                               │
    │                                │ │  Global Model          │ │                               │
    │                                │ │  Repository            │ │                               │
    │                                │ └────────────────────────┘ │                               │
    │                                │                            │                               │
    │                                │ ┌────────────────────────┐ │                               │
    │                                │ │  Traffic Simulator     │ │                               │
    │                                │ │  (Synthetic Data       │ │                               │
    │                                │ │   Generator)           │ │                               │
    │                                │ └────────────────────────┘ │                               │
    │                                │                            │                               │
    │                                │ ┌────────────────────────┐ │                               │
    │                                │ │  Data Pipeline         │ │                               │
    │                                │ │  (Ingestion &          │ │                               │
    │                                │ │   Preprocessing)       │ │                               │
    │                                │ └────────────────────────┘ │                               │
    │                                └────────────────────────────┘                               │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                   │
    ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    │                           MONITORING & SECURITY LAYER                                       │
    │                                                                                              │
    │  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐              │
    │  │   Security          │    │   Monitoring        │    │   Alert System      │              │
    │  │   Operations        │    │   Dashboard         │    │                     │              │
    │  │   Center (SOC)      │    │                     │    │ ┌─────────────────┐ │              │
    │  │                     │    │ ┌─────────────────┐ │    │ │ Email Alerts    │ │              │
    │  │ ┌─────────────────┐ │    │ │ Real-Time       │ │    │ └─────────────────┘ │              │
    │  │ │ Incident        │ │    │ │ Metrics         │ │    │                     │              │
    │  │ │ Response        │ │    │ │ (Grafana)       │ │    │ ┌─────────────────┐ │              │
    │  │ │ System          │ │    │ └─────────────────┘ │    │ │ Slack/Teams     │ │              │
    │  │ └─────────────────┘ │    │                     │    │ │ Webhooks        │ │              │
    │  │                     │    │ ┌─────────────────┐ │    │ └─────────────────┘ │              │
    │  │ ┌─────────────────┐ │    │ │ Performance     │ │    │                     │              │
    │  │ │ Firewall        │ │    │ │ Analytics       │ │    │ ┌─────────────────┐ │              │
    │  │ │ Management      │ │    │ │ (Prometheus)    │ │    │ │ SIEM            │ │              │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │    │ │ Integration     │ │              │
    │  │                     │    │                     │    │ └─────────────────┘ │              │
    │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │    └─────────────────────┘              │
    │  │ │ Log             │ │    │ │ Training        │ │                                          │
    │  │ │ Management      │ │    │ │ Progress        │ │                                          │
    │  │ │ (ELK Stack)     │ │    │ │ Tracking        │ │                                          │
    │  │ └─────────────────┘ │    │ └─────────────────┘ │                                          │
    │  └─────────────────────┘    └─────────────────────┘                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

These comprehensive diagrams now include both the federated learning training system AND the complete real-time prediction pipeline, showing how your project works end-to-end from training to production deployment!

Similar code found with 8 license types