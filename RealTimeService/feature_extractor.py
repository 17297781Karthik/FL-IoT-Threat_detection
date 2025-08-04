#!/usr/bin/env python3
"""
Feature Extraction Pipeline for IoT Network Traffic
Extracts network flow features from PCAP files for threat detection
"""

import pandas as pd
import numpy as np
from scapy.all import *
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether
import os
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkFeatureExtractor:
    """
    Extracts network flow features from PCAP files for IoT threat detection
    """
    
    def __init__(self):
        self.features = [
            'HH_jit_L1_variance', 'HH_jit_L0.01_variance', 'HH_jit_L3_variance',
            'HH_jit_L0.1_variance', 'HH_jit_L0.1_mean', 'H_L0.01_variance',
            'HH_jit_L5_variance', 'MI_dir_L0.01_variance', 'MI_dir_L0.1_mean',
            'MI_dir_L0.01_weight', 'HH_L0.01_std', 'MI_dir_L0.1_weight',
            'H_L0.01_mean', 'H_L0.01_weight', 'HH_jit_L5_mean',
            'MI_dir_L1_weight', 'H_L0.1_weight'
        ]
        
        # Time windows for lambda calculations (in seconds)
        self.lambdas = [0.01, 0.1, 1, 3, 5]
        
    def read_pcap(self, pcap_file: str) -> List[Dict]:
        """Read PCAP file and extract basic packet information"""
        logger.info(f"Reading PCAP file: {pcap_file}")
        
        try:
            packets = rdpcap(pcap_file)
            packet_data = []
            
            for i, pkt in enumerate(packets):
                pkt_info = {
                    'timestamp': float(pkt.time),
                    'packet_id': i,
                    'size': len(pkt)
                }
                
                # Extract IP layer information
                if IP in pkt:
                    pkt_info.update({
                        'src_ip': pkt[IP].src,
                        'dst_ip': pkt[IP].dst,
                        'protocol': pkt[IP].proto,
                        'ttl': pkt[IP].ttl,
                        'ip_len': pkt[IP].len
                    })
                    
                    # Extract TCP information
                    if TCP in pkt:
                        pkt_info.update({
                            'src_port': pkt[TCP].sport,
                            'dst_port': pkt[TCP].dport,
                            'tcp_flags': pkt[TCP].flags,
                            'seq': pkt[TCP].seq,
                            'ack': pkt[TCP].ack,
                            'window': pkt[TCP].window
                        })
                        
                    # Extract UDP information
                    elif UDP in pkt:
                        pkt_info.update({
                            'src_port': pkt[UDP].sport,
                            'dst_port': pkt[UDP].dport,
                            'udp_len': pkt[UDP].len
                        })
                
                packet_data.append(pkt_info)
                
            logger.info(f"Extracted {len(packet_data)} packets")
            return packet_data
            
        except Exception as e:
            logger.error(f"Error reading PCAP file {pcap_file}: {e}")
            return []
    
    def create_flows(self, packets: List[Dict]) -> Dict[str, List[Dict]]:
        """Group packets into bidirectional flows"""
        flows = defaultdict(list)
        
        for pkt in packets:
            if 'src_ip' in pkt and 'dst_ip' in pkt:
                # Create flow key (bidirectional)
                src_ip, dst_ip = pkt['src_ip'], pkt['dst_ip']
                src_port = pkt.get('src_port', 0)
                dst_port = pkt.get('dst_port', 0)
                protocol = pkt.get('protocol', 0)
                
                # Normalize flow key for bidirectional flows
                if (src_ip, src_port) < (dst_ip, dst_port):
                    flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
                    pkt['direction'] = 'forward'
                else:
                    flow_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
                    pkt['direction'] = 'backward'
                
                flows[flow_key].append(pkt)
        
        return flows
    
    def calculate_entropy(self, values: List[Any]) -> float:
        """Calculate Shannon entropy of values"""
        if not values:
            return 0.0
        
        value_counts = Counter(values)
        total = len(values)
        entropy = 0.0
        
        for count in value_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def calculate_lambda_features(self, timestamps: List[float], lambda_val: float) -> Dict[str, float]:
        """Calculate features for a specific lambda (time window)"""
        if len(timestamps) < 2:
            return {
                f'mean': 0.0,
                f'variance': 0.0,
                f'weight': 0.0
            }
        
        # Convert to relative timestamps
        base_time = min(timestamps)
        rel_timestamps = [t - base_time for t in timestamps]
        
        # Calculate inter-arrival times
        intervals = [rel_timestamps[i+1] - rel_timestamps[i] for i in range(len(rel_timestamps)-1)]
        
        if not intervals:
            return {
                f'mean': 0.0,
                f'variance': 0.0,
                f'weight': 0.0
            }
        
        # Apply exponential decay weights
        weights = [np.exp(-lambda_val * interval) for interval in intervals]
        
        # Calculate weighted statistics
        if sum(weights) > 0:
            weighted_mean = sum(w * interval for w, interval in zip(weights, intervals)) / sum(weights)
            weighted_variance = sum(w * (interval - weighted_mean)**2 for w, interval in zip(weights, intervals)) / sum(weights)
            total_weight = sum(weights)
        else:
            weighted_mean = np.mean(intervals)
            weighted_variance = np.var(intervals)
            total_weight = 1.0
        
        return {
            'mean': weighted_mean,
            'variance': weighted_variance,
            'weight': total_weight
        }
    
    def calculate_jitter_features(self, timestamps: List[float], lambda_val: float) -> Dict[str, float]:
        """Calculate jitter features for HH (header-to-header) timing"""
        if len(timestamps) < 3:
            return {
                'variance': 0.0,
                'mean': 0.0
            }
        
        # Calculate inter-arrival times
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        # Calculate jitter (variation in inter-arrival times)
        jitters = [abs(intervals[i+1] - intervals[i]) for i in range(len(intervals)-1)]
        
        if not jitters:
            return {
                'variance': 0.0,
                'mean': 0.0
            }
        
        # Apply exponential decay
        weights = [np.exp(-lambda_val * (i+1)) for i in range(len(jitters))]
        
        if sum(weights) > 0:
            weighted_mean = sum(w * jitter for w, jitter in zip(weights, jitters)) / sum(weights)
            weighted_variance = sum(w * (jitter - weighted_mean)**2 for w, jitter in zip(weights, jitters)) / sum(weights)
        else:
            weighted_mean = np.mean(jitters)
            weighted_variance = np.var(jitters)
        
        return {
            'variance': weighted_variance,
            'mean': weighted_mean
        }
    
    def calculate_mutual_information_features(self, packets: List[Dict], lambda_val: float) -> Dict[str, float]:
        """Calculate mutual information features for directional analysis"""
        if len(packets) < 2:
            return {
                'variance': 0.0,
                'mean': 0.0,
                'weight': 0.0
            }
        
        # Extract packet sizes by direction
        forward_sizes = [pkt['size'] for pkt in packets if pkt.get('direction') == 'forward']
        backward_sizes = [pkt['size'] for pkt in packets if pkt.get('direction') == 'backward']
        
        if not forward_sizes and not backward_sizes:
            return {
                'variance': 0.0,
                'mean': 0.0,
                'weight': 0.0
            }
        
        # Calculate directional ratios
        total_forward = sum(forward_sizes) if forward_sizes else 0
        total_backward = sum(backward_sizes) if backward_sizes else 0
        total_size = total_forward + total_backward
        
        if total_size == 0:
            return {
                'variance': 0.0,
                'mean': 0.0,
                'weight': 0.0
            }
        
        forward_ratio = total_forward / total_size
        backward_ratio = total_backward / total_size
        
        # Calculate mutual information approximation
        mi_values = []
        timestamps = [pkt['timestamp'] for pkt in packets]
        
        for i in range(len(packets)):
            # Simplified MI calculation based on directional flow
            direction_entropy = self.calculate_entropy([pkt.get('direction', 'unknown') for pkt in packets])
            size_entropy = self.calculate_entropy([pkt['size'] for pkt in packets])
            
            # Approximate mutual information
            mi_approx = direction_entropy + size_entropy - (direction_entropy * size_entropy)
            mi_values.append(mi_approx)
        
        # Apply lambda weighting
        weights = [np.exp(-lambda_val * i) for i in range(len(mi_values))]
        
        if sum(weights) > 0:
            weighted_mean = sum(w * mi for w, mi in zip(weights, mi_values)) / sum(weights)
            weighted_variance = sum(w * (mi - weighted_mean)**2 for w, mi in zip(weights, mi_values)) / sum(weights)
            total_weight = sum(weights)
        else:
            weighted_mean = np.mean(mi_values)
            weighted_variance = np.var(mi_values)
            total_weight = 1.0
        
        return {
            'variance': weighted_variance,
            'mean': weighted_mean,
            'weight': total_weight
        }
    
    def extract_flow_features(self, flow_packets: List[Dict]) -> Dict[str, float]:
        """Extract all features for a single flow"""
        if not flow_packets:
            return {feature: 0.0 for feature in self.features}
        
        timestamps = [pkt['timestamp'] for pkt in flow_packets]
        timestamps.sort()
        
        features = {}
        
        # Extract HH (Header-to-Header) jitter features
        for lambda_val in self.lambdas:
            jitter_feats = self.calculate_jitter_features(timestamps, lambda_val)
            features[f'HH_jit_L{lambda_val}_variance'] = jitter_feats['variance']
            features[f'HH_jit_L{lambda_val}_mean'] = jitter_feats['mean']
        
        # Extract H (Header) timing features
        for lambda_val in [0.01, 0.1]:
            timing_feats = self.calculate_lambda_features(timestamps, lambda_val)
            features[f'H_L{lambda_val}_variance'] = timing_feats['variance']
            features[f'H_L{lambda_val}_mean'] = timing_feats['mean']
            features[f'H_L{lambda_val}_weight'] = timing_feats['weight']
        
        # Extract HH (Header-to-Header) standard deviation
        if len(timestamps) > 1:
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            features['HH_L0.01_std'] = np.std(intervals) if intervals else 0.0
        else:
            features['HH_L0.01_std'] = 0.0
        
        # Extract MI (Mutual Information) directional features
        for lambda_val in [0.01, 0.1, 1]:
            mi_feats = self.calculate_mutual_information_features(flow_packets, lambda_val)
            features[f'MI_dir_L{lambda_val}_variance'] = mi_feats['variance']
            features[f'MI_dir_L{lambda_val}_mean'] = mi_feats['mean']
            features[f'MI_dir_L{lambda_val}_weight'] = mi_feats['weight']
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in features:
                features[feature] = 0.0
        
        return features
    
    def extract_features_from_pcap(self, pcap_file: str) -> pd.DataFrame:
        """Extract features from a PCAP file"""
        logger.info(f"Extracting features from: {pcap_file}")
        
        # Read packets
        packets = self.read_pcap(pcap_file)
        if not packets:
            logger.warning(f"No packets found in {pcap_file}")
            return pd.DataFrame()
        
        # Create flows
        flows = self.create_flows(packets)
        logger.info(f"Created {len(flows)} flows")
        
        # Extract features for each flow
        feature_data = []
        for flow_key, flow_packets in flows.items():
            flow_features = self.extract_flow_features(flow_packets)
            flow_features['flow_id'] = flow_key
            flow_features['packet_count'] = len(flow_packets)
            feature_data.append(flow_features)
        
        if not feature_data:
            logger.warning("No features extracted")
            return pd.DataFrame()
        
        df = pd.DataFrame(feature_data)
        logger.info(f"Extracted features for {len(df)} flows")
        
        # Ensure feature columns are in correct order
        feature_columns = [col for col in self.features if col in df.columns]
        other_columns = [col for col in df.columns if col not in self.features]
        
        return df[feature_columns + other_columns]
    
    def process_multiple_pcaps(self, pcap_directory: str) -> pd.DataFrame:
        """Process multiple PCAP files in a directory"""
        logger.info(f"Processing PCAP files in: {pcap_directory}")
        
        if not os.path.exists(pcap_directory):
            logger.error(f"Directory not found: {pcap_directory}")
            return pd.DataFrame()
        
        pcap_files = [f for f in os.listdir(pcap_directory) if f.endswith('.pcap')]
        logger.info(f"Found {len(pcap_files)} PCAP files")
        
        all_features = []
        
        for pcap_file in pcap_files:
            pcap_path = os.path.join(pcap_directory, pcap_file)
            try:
                df = self.extract_features_from_pcap(pcap_path)
                if not df.empty:
                    df['source_file'] = pcap_file
                    all_features.append(df)
            except Exception as e:
                logger.error(f"Error processing {pcap_file}: {e}")
                continue
        
        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            logger.info(f"Combined features: {len(combined_df)} total flows")
            return combined_df
        else:
            logger.warning("No features extracted from any PCAP files")
            return pd.DataFrame()
    
    def save_features(self, features_df: pd.DataFrame, output_file: str):
        """Save extracted features to CSV file"""
        if features_df.empty:
            logger.warning("No features to save")
            return
        
        features_df.to_csv(output_file, index=False)
        logger.info(f"Features saved to: {output_file}")

def main():
    """Main function for testing feature extraction"""
    extractor = NetworkFeatureExtractor()
    
    # Test with a single PCAP file
    pcap_file = "../samplePackets/pcaps/00_benign.pcap"
    if os.path.exists(pcap_file):
        features_df = extractor.extract_features_from_pcap(pcap_file)
        print(f"Extracted features shape: {features_df.shape}")
        print(f"Features: {list(features_df.columns)}")
        
        # Save features
        extractor.save_features(features_df, "test_features.csv")
    else:
        print(f"Test PCAP file not found: {pcap_file}")

if __name__ == "__main__":
    main()
