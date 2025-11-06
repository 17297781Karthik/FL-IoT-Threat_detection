#!/usr/bin/env python3
"""
Real-time IoT Threat Detection Pipeline
Complete pipeline for processing PCAP files and detecting threats
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
import argparse

from feature_extractor import NetworkFeatureExtractor
from threat_predictor import IoTThreatPredictor
from alerting import ThreatEmailAlerter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('iot_threat_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealTimeThreatDetectionPipeline:
    """
    Complete pipeline for real-time IoT threat detection
    """
    
    def __init__(self, pcap_path: str, model_path: str, enable_alerts: bool = False):
        self.pcap_path = pcap_path
        self.model_path = model_path
        self.enable_alerts = enable_alerts
        
        self.feature_extractor = NetworkFeatureExtractor()
        self.threat_predictor = IoTThreatPredictor(model_path)
        
        if self.enable_alerts:
            try:
                self.alerter = ThreatEmailAlerter()
                logger.info("Email alerting enabled")
            except Exception as e:
                logger.warning(f"Email alerting disabled: {e}")
                self.enable_alerts = False
        
        self.results_dir = "detection_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("Pipeline initialized successfully")
    
    def process_single_pcap(self, pcap_file: str) -> Dict[str, Any]:
        """Process a single PCAP file through the complete pipeline"""
        logger.info(f"Processing PCAP file: {pcap_file}")
        
        start_time = time.time()
        
        try:
            # Step 1: Extract features
            logger.info("Step 1: Extracting features...")
            features_df = self.feature_extractor.extract_features_from_pcap(pcap_file)
            
            if features_df.empty:
                logger.warning(f"No features extracted from {pcap_file}")
                return {
                    'status': 'failed',
                    'error': 'No features extracted',
                    'file': pcap_file
                }
            
            # Step 2: Make predictions
            logger.info("Step 2: Making threat predictions...")
            predictions = self.threat_predictor.predict(features_df)
            
            if not predictions:
                logger.warning(f"No predictions made for {pcap_file}")
                return {
                    'status': 'failed',
                    'error': 'No predictions made',
                    'file': pcap_file
                }
            
            # Step 3: Analyze results
            analysis = self._analyze_predictions(predictions)
            
            processing_time = time.time() - start_time
            
            result = {
                'status': 'success',
                'file': pcap_file,
                'processing_time': processing_time,
                'total_flows': len(predictions),
                'predictions': predictions,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.enable_alerts and analysis.get('malicious_flows', 0) > 0:
                self._send_threat_alerts(analysis, pcap_file)
            
            logger.info(f"Successfully processed {pcap_file} in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pcap_file}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'file': pcap_file,
                'timestamp': datetime.now().isoformat()
            }
    
    def process_pcap_directory(self) -> List[Dict[str, Any]]:
        """Process all PCAP files in the specified directory"""
        logger.info(f"Processing PCAP directory: {self.pcap_path}")
        
        if not os.path.exists(self.pcap_path):
            logger.error(f"PCAP directory not found: {self.pcap_path}")
            return []
        
        pcap_files = [f for f in os.listdir(self.pcap_path) if f.endswith('.pcap')]
        logger.info(f"Found {len(pcap_files)} PCAP files")
        
        results = []
        
        for pcap_file in pcap_files:
            pcap_path = os.path.join(self.pcap_path, pcap_file)
            result = self.process_single_pcap(pcap_path)
            results.append(result)
            
            # Save individual result
            result_file = os.path.join(self.results_dir, f"{pcap_file}_result.json")
            self._save_result(result, result_file)
        
        return results
    
    def _analyze_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prediction results for summary statistics"""
        if not predictions:
            return {}
        
        total_flows = len(predictions)
        malicious_flows = sum(1 for p in predictions if p['is_malicious'])
        benign_flows = total_flows - malicious_flows
        
        attack_counts = {}
        threat_levels = {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0, 'Uncertain': 0}
        confidence_scores = []
        
        for pred in predictions:
            attack_type = pred['predicted_attack']
            attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1
            
            threat_level = pred['threat_level']
            threat_levels[threat_level] = threat_levels.get(threat_level, 0) + 1
            
            confidence_scores.append(pred['confidence'])
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        max_confidence = max(confidence_scores)
        min_confidence = min(confidence_scores)
        
        analysis = {
            'total_flows': total_flows,
            'malicious_flows': malicious_flows,
            'benign_flows': benign_flows,
            'malicious_percentage': (malicious_flows / total_flows) * 100,
            'attack_type_distribution': attack_counts,
            'threat_level_distribution': threat_levels,
            'confidence_statistics': {
                'average': avg_confidence,
                'maximum': max_confidence,
                'minimum': min_confidence
            },
            'high_confidence_threats': [
                p for p in predictions 
                if p['is_malicious'] and p['confidence'] >= 0.8
            ]
        }
        
        return analysis
    
    def _send_threat_alerts(self, analysis: Dict[str, Any], pcap_file: str):
        if not self.enable_alerts:
            return
        
        attack_distribution = analysis.get('attack_type_distribution', {})
        
        for attack_type, count in attack_distribution.items():
            if attack_type.lower() == 'benign':
                continue
            
            confidence_stats = analysis.get('confidence_statistics', {})
            avg_confidence = confidence_stats.get('average', 0) * 100
            
            detection_data = {
                'attack_type': attack_type,
                'confidence': f"{avg_confidence:.1f}%",
                'file': os.path.basename(pcap_file),
                'total_flows': analysis.get('total_flows', 0),
                'malicious_flows': count,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            try:
                self.alerter.send_alert(detection_data)
                logger.info(f"Alert sent for {attack_type}")
            except Exception as e:
                logger.error(f"Failed to send alert for {attack_type}: {e}")
    
    def _save_result(self, result: Dict[str, Any], output_file: str):
        """Save result to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving result to {output_file}: {e}")
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary report for all processed files"""
        logger.info("Generating summary report...")
        
        total_files = len(results)
        successful_files = sum(1 for r in results if r['status'] == 'success')
        failed_files = total_files - successful_files
        
        total_flows = 0
        total_malicious = 0
        all_attack_types = {}
        all_threat_levels = {'None': 0, 'Low': 0, 'Medium': 0, 'High': 0, 'Uncertain': 0}
        processing_times = []
        
        for result in results:
            if result['status'] == 'success':
                analysis = result['analysis']
                total_flows += analysis['total_flows']
                total_malicious += analysis['malicious_flows']
                processing_times.append(result['processing_time'])
                
                # Aggregate attack types
                for attack, count in analysis['attack_type_distribution'].items():
                    all_attack_types[attack] = all_attack_types.get(attack, 0) + count
                
                # Aggregate threat levels
                for level, count in analysis['threat_level_distribution'].items():
                    all_threat_levels[level] += count
        
        summary = {
            'generation_time': datetime.now().isoformat(),
            'files_processed': {
                'total': total_files,
                'successful': successful_files,
                'failed': failed_files,
                'success_rate': (successful_files / total_files) * 100 if total_files > 0 else 0
            },
            'flow_analysis': {
                'total_flows': total_flows,
                'malicious_flows': total_malicious,
                'benign_flows': total_flows - total_malicious,
                'malicious_percentage': (total_malicious / total_flows) * 100 if total_flows > 0 else 0
            },
            'attack_distribution': all_attack_types,
            'threat_level_distribution': all_threat_levels,
            'performance': {
                'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
                'total_processing_time': sum(processing_times),
                'fastest_processing': min(processing_times) if processing_times else 0,
                'slowest_processing': max(processing_times) if processing_times else 0
            }
        }
        
        return summary
    
    def run_pipeline(self):
        """Run the complete threat detection pipeline"""
        logger.info("Starting IoT Threat Detection Pipeline")
        logger.info("=" * 60)
        
        # Print configuration
        logger.info(f"PCAP Path: {self.pcap_path}")
        logger.info(f"Model Path: {self.model_path}")
        logger.info(f"Results Directory: {self.results_dir}")
        
        start_time = time.time()
        
        # Process all PCAP files
        results = self.process_pcap_directory()
        
        # Generate summary report
        summary = self.generate_summary_report(results)
        
        # Save summary report
        summary_file = os.path.join(self.results_dir, "summary_report.json")
        self._save_result(summary, summary_file)
        
        total_time = time.time() - start_time
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Files processed: {summary['files_processed']['total']}")
        logger.info(f"Success rate: {summary['files_processed']['success_rate']:.1f}%")
        logger.info(f"Total flows analyzed: {summary['flow_analysis']['total_flows']}")
        logger.info(f"Malicious flows detected: {summary['flow_analysis']['malicious_flows']}")
        logger.info(f"Malicious percentage: {summary['flow_analysis']['malicious_percentage']:.1f}%")
        logger.info(f"Results saved to: {self.results_dir}")
        
        return results, summary

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='IoT Threat Detection Pipeline')
    parser.add_argument(
        '--pcap-path',
        default='../samplePackets/pcaps',
        help='Path to PCAP files directory (default: ../samplePackets/pcaps)'
    )
    parser.add_argument(
        '--model-path',
        default='../SavedGlobalModel/final_model.pth',
        help='Path to trained model file (default: ../SavedGlobalModel/final_model.pth)'
    )
    parser.add_argument(
        '--single-file',
        help='Process a single PCAP file instead of directory'
    )
    parser.add_argument(
        '--enable-alerts',
        action='store_true',
        help='Enable email alerts for detected threats'
    )
    
    args = parser.parse_args()
    
    pipeline = RealTimeThreatDetectionPipeline(args.pcap_path, args.model_path, args.enable_alerts)
    
    if args.single_file:
        result = pipeline.process_single_pcap(args.single_file)
        print(json.dumps(result, indent=2))
    else:
        pipeline.run_pipeline()

if __name__ == "__main__":
    main()
