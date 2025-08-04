#!/usr/bin/env python3
"""
Setup script for IoT Threat Detection Pipeline
Helps with initial setup and verification
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineSetup:
    """Setup and verification for the IoT Threat Detection Pipeline"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.project_root = self.base_dir.parent
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        logger.info("Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 7):
            logger.error(f"Python 3.7+ required, found {version.major}.{version.minor}")
            return False
        
        logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def install_requirements(self):
        """Install Python requirements"""
        logger.info("Installing Python requirements...")
        
        requirements_file = self.base_dir / "requirements.txt"
        if not requirements_file.exists():
            logger.error("requirements.txt not found")
            return False
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            logger.info("✓ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {e}")
            return False
    
    def check_directories(self):
        """Check and create necessary directories"""
        logger.info("Checking directory structure...")
        
        required_dirs = [
            self.project_root / "samplePackets" / "pcaps",
            self.project_root / "SavedGlobalModel",
            self.base_dir / "detection_results"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.warning(f"Creating missing directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
            else:
                logger.info(f"✓ Directory exists: {dir_path}")
        
        return True
    
    def check_model_file(self):
        """Check if the model file exists"""
        logger.info("Checking model file...")
        
        model_path = self.project_root / "SavedGlobalModel" / "final_model.pth"
        
        if model_path.exists():
            logger.info(f"✓ Model file found: {model_path}")
            
            # Check file size
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Model file size: {size_mb:.2f} MB")
            return True
        else:
            logger.warning(f"Model file not found: {model_path}")
            logger.info("  You'll need to train a model or copy an existing one")
            return False
    
    def check_pcap_files(self):
        """Check for PCAP files"""
        logger.info("Checking for PCAP files...")
        
        pcap_dir = self.project_root / "samplePackets" / "pcaps"
        
        if not pcap_dir.exists():
            logger.warning(f"PCAP directory not found: {pcap_dir}")
            return False
        
        pcap_files = list(pcap_dir.glob("*.pcap"))
        
        if pcap_files:
            logger.info(f"✓ Found {len(pcap_files)} PCAP files")
            for pcap_file in pcap_files[:5]:  # Show first 5
                logger.info(f"  - {pcap_file.name}")
            if len(pcap_files) > 5:
                logger.info(f"  ... and {len(pcap_files) - 5} more")
            return True
        else:
            logger.warning("No PCAP files found")
            logger.info("  You can generate test data using the packet simulator")
            return False
    
    def verify_imports(self):
        """Verify that all required modules can be imported"""
        logger.info("Verifying module imports...")
        
        modules = [
            ("torch", "PyTorch"),
            ("pandas", "Pandas"),
            ("numpy", "NumPy"),
            ("scapy", "Scapy"),
            ("sklearn", "Scikit-learn")
        ]
        
        success = True
        
        for module, name in modules:
            try:
                __import__(module)
                logger.info(f"✓ {name} imported successfully")
            except ImportError as e:
                logger.error(f"✗ Failed to import {name}: {e}")
                success = False
        
        return success
    
    def test_pipeline_components(self):
        """Test individual pipeline components"""
        logger.info("Testing pipeline components...")
        
        # Change to RealTimeService directory
        original_cwd = os.getcwd()
        os.chdir(self.base_dir)
        
        try:
            # Test feature extractor
            logger.info("Testing feature extractor...")
            try:
                from feature_extractor import NetworkFeatureExtractor
                extractor = NetworkFeatureExtractor()
                logger.info(f"✓ Feature extractor initialized ({len(extractor.features)} features)")
            except Exception as e:
                logger.error(f"✗ Feature extractor failed: {e}")
                return False
            
            # Test threat predictor
            logger.info("Testing threat predictor...")
            try:
                from threat_predictor import IoTThreatPredictor
                # Create predictor without loading model (for testing)
                predictor = IoTThreatPredictor.__new__(IoTThreatPredictor)
                predictor.feature_names = extractor.features
                predictor.attack_names = {i: f"Attack_{i}" for i in range(10)}
                predictor.model = None
                logger.info(f"✓ Threat predictor structure verified")
            except Exception as e:
                logger.error(f"✗ Threat predictor failed: {e}")
                return False
            
            # Test pipeline
            logger.info("Testing complete pipeline...")
            try:
                from realtime_pipeline import RealTimeThreatDetectionPipeline
                pipeline = RealTimeThreatDetectionPipeline("../samplePackets/pcaps", "../SavedGlobalModel/final_model.pth")
                logger.info("✓ Pipeline initialized successfully")
            except Exception as e:
                logger.error(f"✗ Pipeline failed: {e}")
                return False
            
            return True
            
        finally:
            os.chdir(original_cwd)
    
    def generate_sample_config(self):
        """Generate a sample configuration file"""
        logger.info("Checking configuration...")
        
        config_file = self.base_dir / "config.json"
        
        if config_file.exists():
            logger.info("✓ Configuration file exists")
            return True
        
        logger.info("Creating sample configuration file...")
        
        config = {
            "paths": {
                "pcap_directory": "../samplePackets/pcaps",
                "model_path": "../SavedGlobalModel/final_model.pth",
                "results_directory": "detection_results"
            },
            "model": {
                "num_features": 17,
                "num_classes": 10
            },
            "processing": {
                "confidence_threshold": 0.5,
                "high_confidence_threshold": 0.8
            }
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"✓ Configuration file created: {config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to create configuration: {e}")
            return False
    
    def run_full_setup(self):
        """Run complete setup process"""
        logger.info("IoT Threat Detection Pipeline Setup")
        logger.info("=" * 50)
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Directory Structure", self.check_directories),
            ("Configuration", self.generate_sample_config),
            ("Dependencies", self.install_requirements),
            ("Module Imports", self.verify_imports),
            ("Model File", self.check_model_file),
            ("PCAP Files", self.check_pcap_files),
            ("Pipeline Components", self.test_pipeline_components)
        ]
        
        results = []
        
        for check_name, check_func in checks:
            logger.info(f"\n--- {check_name} ---")
            try:
                result = check_func()
                results.append((check_name, result))
            except Exception as e:
                logger.error(f"Check failed with error: {e}")
                results.append((check_name, False))
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("SETUP SUMMARY")
        logger.info("=" * 50)
        
        passed = 0
        for check_name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{check_name}: {status}")
            if result:
                passed += 1
        
        total = len(results)
        logger.info(f"\nTotal: {passed}/{total} checks passed")
        
        if passed == total:
            logger.info("\n🎉 Setup completed successfully!")
            logger.info("You can now run the pipeline using:")
            logger.info("  python realtime_pipeline.py")
            logger.info("  python monitor.py")
        else:
            logger.warning(f"\n⚠️  Setup incomplete. {total - passed} issues need attention.")
            
            if not any(check[1] for check in results if check[0] in ["Model File", "PCAP Files"]):
                logger.info("\nTo get started:")
                logger.info("1. Train a model or copy final_model.pth to SavedGlobalModel/")
                logger.info("2. Add PCAP files to samplePackets/pcaps/ or generate them")
                logger.info("3. Run: python test_pipeline.py")
        
        return passed == total

def main():
    """Main setup function"""
    setup = PipelineSetup()
    success = setup.run_full_setup()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
