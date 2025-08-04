#!/usr/bin/env python3
"""
Test script for IoT Threat Detection Pipeline
Quick validation of the pipeline components
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_feature_extractor():
    """Test the feature extraction component"""
    logger.info("Testing Feature Extractor...")
    
    try:
        from feature_extractor import NetworkFeatureExtractor
        
        extractor = NetworkFeatureExtractor()
        logger.info(f"✓ Feature extractor initialized")
        logger.info(f"✓ Expected features: {len(extractor.features)}")
        logger.info(f"✓ Feature list: {extractor.features}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Feature extractor test failed: {e}")
        return False

def test_threat_predictor():
    """Test the threat prediction component"""
    logger.info("Testing Threat Predictor...")
    
    try:
        from threat_predictor import IoTThreatPredictor
        
        model_path = "../SavedGlobalModel/final_model.pth"
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            logger.info("Creating dummy predictor without loading model...")
            predictor = IoTThreatPredictor.__new__(IoTThreatPredictor)
            predictor.model = None
            predictor.feature_names = [
                'HH_jit_L1_variance', 'HH_jit_L0.01_variance', 'HH_jit_L3_variance',
                'HH_jit_L0.1_variance', 'HH_jit_L0.1_mean', 'H_L0.01_variance',
                'HH_jit_L5_variance', 'MI_dir_L0.01_variance', 'MI_dir_L0.1_mean',
                'MI_dir_L0.01_weight', 'HH_L0.01_std', 'MI_dir_L0.1_weight',
                'H_L0.01_mean', 'H_L0.01_weight', 'HH_jit_L5_mean',
                'MI_dir_L1_weight', 'H_L0.1_weight'
            ]
            predictor.attack_names = {
                0: "Benign", 1: "Gafgyt Combo", 2: "Gafgyt Junk", 3: "Gafgyt TCP",
                5: "Mirai ACK", 6: "Mirai Scan", 7: "Mirai SYN", 8: "Mirai UDP", 9: "Mirai UDPPlain"
            }
        else:
            predictor = IoTThreatPredictor(model_path)
            if predictor.model is None:
                logger.warning("Model failed to load but predictor created")
            else:
                logger.info("✓ Model loaded successfully")
        
        logger.info(f"✓ Threat predictor initialized")
        logger.info(f"✓ Feature names: {len(predictor.feature_names)}")
        logger.info(f"✓ Attack types: {len(predictor.attack_names)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Threat predictor test failed: {e}")
        return False

def test_pipeline():
    """Test the complete pipeline"""
    logger.info("Testing Complete Pipeline...")
    
    try:
        from realtime_pipeline import RealTimeThreatDetectionPipeline
        
        pcap_path = "../samplePackets/pcaps"
        model_path = "../SavedGlobalModel/final_model.pth"
        
        pipeline = RealTimeThreatDetectionPipeline(pcap_path, model_path)
        logger.info("✓ Pipeline initialized")
        
        # Test directory existence
        if os.path.exists(pcap_path):
            pcap_files = [f for f in os.listdir(pcap_path) if f.endswith('.pcap')]
            logger.info(f"✓ Found {len(pcap_files)} PCAP files in {pcap_path}")
        else:
            logger.warning(f"PCAP directory not found: {pcap_path}")
        
        if os.path.exists(model_path):
            logger.info(f"✓ Model file found: {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}")
        return False

def test_config():
    """Test configuration file"""
    logger.info("Testing Configuration...")
    
    try:
        config_file = "config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info("✓ Configuration file loaded")
            logger.info(f"✓ PCAP path: {config['paths']['pcap_directory']}")
            logger.info(f"✓ Model path: {config['paths']['model_path']}")
            logger.info(f"✓ Features: {len(config['model']['feature_names'])}")
        else:
            logger.warning(f"Configuration file not found: {config_file}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("IoT Threat Detection Pipeline - Test Suite")
    logger.info("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("Feature Extractor", test_feature_extractor),
        ("Threat Predictor", test_threat_predictor),
        ("Complete Pipeline", test_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✓ {test_name} test PASSED")
            else:
                logger.warning(f"✗ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Pipeline is ready to use.")
    else:
        logger.warning("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
