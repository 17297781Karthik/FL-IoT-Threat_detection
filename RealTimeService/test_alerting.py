import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alerting import ThreatEmailAlerter

def test_gafgyt_alert():
    alerter = ThreatEmailAlerter()
    
    detection_data = {
        'attack_type': 'Gafgyt Combo',
        'confidence': '95.5%',
        'file': 'test_traffic.pcap',
        'total_flows': 1000,
        'malicious_flows': 850,
        'timestamp': '2025-11-06 10:30:45'
    }
    
    success = alerter.send_alert(detection_data)
    print(f"Gafgyt alert sent: {success}")

def test_mirai_alert():
    alerter = ThreatEmailAlerter()
    
    detection_data = {
        'attack_type': 'Mirai SYN',
        'confidence': '98.2%',
        'file': 'suspicious_traffic.pcap',
        'total_flows': 2000,
        'malicious_flows': 1950,
        'timestamp': '2025-11-06 10:35:22'
    }
    
    success = alerter.send_alert(detection_data)
    print(f"Mirai alert sent: {success}")

def test_batch_alert():
    alerter = ThreatEmailAlerter()
    
    detections = [
        {
            'attack_type': 'Gafgyt TCP',
            'confidence': '92.1%',
            'file': 'traffic_001.pcap'
        },
        {
            'attack_type': 'Mirai UDP',
            'confidence': '96.8%',
            'file': 'traffic_002.pcap'
        }
    ]
    
    success = alerter.send_batch_alert(detections)
    print(f"Batch alert sent: {success}")

if __name__ == '__main__':
    print("Testing Email Alerting System")
    print("-" * 50)
    
    try:
        test_gafgyt_alert()
        test_mirai_alert()
        test_batch_alert()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease configure your .env file with email credentials.")
    except Exception as e:
        print(f"Error: {e}")
