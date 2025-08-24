#!/usr/bin/env python3
"""
Verbose BADNET Monitoring Script
Shows detailed prediction results, attack names, timestamps, and feature values
Author: ARYA KONER
"""

import time
import numpy as np
import pandas as pd
import joblib
import psutil
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class VerboseCyberMonitor:
    def __init__(self):
        """Initialize verbose cyber attack monitor."""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.attack_count = 0
        self.total_predictions = 0
        self.prediction_history = deque(maxlen=100)
        
        # Load the trained model and preprocessors
        self.load_model()
        
        # Attack type mapping
        self.attack_types = {
            0: "NORMAL",
            1: "ATTACK"
        }
        
        # Attack categories (based on training data)
        self.attack_categories = {
            "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop"],
            "Probe": ["ipsweep", "nmap", "portsweep", "satan"],
            "R2L": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"],
            "U2R": ["buffer_overflow", "loadmodule", "rootkit", "perl"]
        }
    
    def load_model(self):
        """Load the trained model and preprocessors."""
        try:
            # Load feature names
            feature_file = "processed_data/supervised_learning/train_features.txt"
            with open(feature_file, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            print(f"Loaded {len(self.feature_names)} features")
            
            # Load scaler
            scaler_path = "processed_data/supervised_learning/feature_scaler.pkl"
            self.scaler = joblib.load(scaler_path)
            print("Loaded feature scaler")
            
            # Load the trained model
            model_path = "processed_data/supervised_learning/best_model.pkl"
            self.model = joblib.load(model_path)
            print(f"Loaded trained model: {type(self.model).__name__}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running in simulation mode")
    
    def get_system_features(self):
        """Get detailed system features."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Get CPU frequency
            try:
                cpu_freq = psutil.cpu_freq()
                cpu_freq_current = cpu_freq.current if cpu_freq else 0
            except:
                cpu_freq_current = 0
            
            features = {
                'cpu_usage': cpu_percent,
                'cpu_frequency': cpu_freq_current,
                'memory_usage': memory.percent,
                'memory_available': memory.available / (1024**3),  # GB
                'memory_used': memory.used / (1024**3),  # GB
                'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
                'process_count': process_count,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'high_cpu': 1 if cpu_percent > 80 else 0,
                'high_memory': 1 if memory.percent > 80 else 0,
                'high_traffic': 1 if (net_io.packets_sent + net_io.packets_recv) > 10000 else 0,
                'disk_activity': 1 if disk_io and (disk_io.read_bytes + disk_io.write_bytes) > 1000000 else 0,
            }
            
            return features
            
        except Exception as e:
            print(f"Error getting system features: {e}")
            return {}
    
    def get_network_features(self):
        """Get detailed network features."""
        try:
            net_io = psutil.net_io_counters()
            
            # Try to get connection count
            try:
                connections = psutil.net_connections()
                connection_count = len([conn for conn in connections if conn.status == 'ESTABLISHED'])
            except:
                connection_count = 0
            
            features = {
                'connection_count': connection_count,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packet_rate': (net_io.packets_sent + net_io.packets_recv) / max(1, connection_count),
                'byte_rate': (net_io.bytes_sent + net_io.bytes_recv) / max(1, connection_count),
                'high_traffic': 1 if (net_io.packets_sent + net_io.packets_recv) > 10000 else 0,
                'suspicious_ports': 0,
            }
            
            return features
            
        except Exception as e:
            print(f"Error getting network features: {e}")
            return {}
    
    def preprocess_features(self, features_dict):
        """Preprocess features to match the training data format."""
        try:
            # Create a DataFrame with the expected features
            df = pd.DataFrame([features_dict])
            
            # Add missing features with default values
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Ensure correct order
            df = df[self.feature_names]
            
            # Apply preprocessing
            if self.scaler:
                df_scaled = self.scaler.transform(df)
            else:
                df_scaled = df.values
            
            return df_scaled
            
        except Exception as e:
            print(f"Error preprocessing features: {e}")
            return None
    
    def predict_attack(self, features):
        """Predict if current activity indicates an attack."""
        try:
            if self.model is None:
                return False, 0.0, "UNKNOWN"
            
            # Make prediction
            prediction = self.model.predict(features)
            probability = self.model.predict_proba(features)[0][1] if hasattr(self.model, 'predict_proba') else 0.5
            
            # Determine attack type based on probability
            if probability > 0.8:
                attack_type = "HIGH_PROBABILITY_ATTACK"
            elif probability > 0.5:
                attack_type = "MODERATE_SUSPICION"
            elif probability > 0.2:
                attack_type = "LOW_SUSPICION"
            else:
                attack_type = "NORMAL"
            
            return bool(prediction[0]), probability, attack_type
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return False, 0.0, "ERROR"
    
    def display_detailed_status(self, features, attack_detected, probability, attack_type):
        """Display detailed monitoring status."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*80)
        print(f"TIMESTAMP: {timestamp}")
        print(f"PREDICTION #{self.total_predictions + 1}")
        print("="*80)
        
        # System Status
        print("SYSTEM STATUS:")
        print(f"   CPU Usage: {features.get('cpu_usage', 0):.1f}%")
        print(f"   Memory Usage: {features.get('memory_usage', 0):.1f}%")
        print(f"   Process Count: {features.get('process_count', 0)}")
        print(f"   Disk Activity: {'HIGH' if features.get('disk_activity', 0) else ' Normal'}")
        
        # Network Status
        print("\nNETWORK STATUS:")
        print(f"   Connections: {features.get('connection_count', 0)}")
        print(f"   Packets Sent: {features.get('packets_sent', 0):,}")
        print(f"   Packets Received: {features.get('packets_recv', 0):,}")
        print(f"   Traffic Level: {'HIGH' if features.get('high_traffic', 0) else ' Normal'}")
        
        # AI Prediction Results
        print("\nAI PREDICTION RESULTS:")
        print(f"   Attack Detected: {'YES' if attack_detected else ' NO'}")
        print(f"   Attack Probability: {probability:.2%}")
        print(f"   Threat Level: {attack_type}")
        
        # Store prediction in history
        prediction_record = {
            'timestamp': timestamp,
            'prediction_number': self.total_predictions + 1,
            'attack_detected': attack_detected,
            'probability': probability,
            'attack_type': attack_type,
            'cpu_usage': features.get('cpu_usage', 0),
            'memory_usage': features.get('memory_usage', 0),
            'connection_count': features.get('connection_count', 0)
        }
        
        self.prediction_history.append(prediction_record)
        
        # Show recent prediction history
        if len(self.prediction_history) > 1:
            print("\n RECENT PREDICTION HISTORY:")
            for i, record in enumerate(list(self.prediction_history)[-3:]):
                status = "ATTACK" if record['attack_detected'] else " NORMAL"
                print(f"   #{record['prediction_number']}: {record['timestamp']} - {status} ({record['probability']:.2%})")
        
        print("="*80)
    
    def monitor_verbose(self, duration=None, interval=5):
        """Run verbose monitoring with detailed output."""
        start_time = time.time()
        
        print("Starting VERBOSE BADNET Monitoring...")
        print(f" Update Interval: {interval} seconds")
        print(f" Duration: {'Continuous' if duration is None else f'{duration} seconds'}")
        print("Mode: Detailed Prediction Analysis")
        print("="*80)
        
        try:
            while True:
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Get both types of features
                system_features = self.get_system_features()
                network_features = self.get_network_features()
                
                # Combine features
                combined_features = {**system_features, **network_features}
                
                if combined_features:
                    # Preprocess features
                    processed_features = self.preprocess_features(combined_features)
                    
                    if processed_features is not None:
                        # Make prediction
                        attack_detected, probability, attack_type = self.predict_attack(processed_features)
                        
                        # Update counters
                        self.total_predictions += 1
                        if attack_detected:
                            self.attack_count += 1
                        
                        # Display detailed status
                        self.display_detailed_status(combined_features, attack_detected, probability, attack_type)
                
                # Wait for next cycle
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        finally:
            self.show_final_summary()
    
    def show_final_summary(self):
        """Show final monitoring summary."""
        print("\n" + "="*80)
        print("FINAL MONITORING SUMMARY")
        print("="*80)
        print(f"Total Predictions: {self.total_predictions}")
        print(f"Attacks Detected: {self.attack_count}")
        print(f"Detection Rate: {self.attack_count/max(1, self.total_predictions)*100:.2f}%")
        
        if self.prediction_history:
            print(f"\n PREDICTION TIMELINE:")
            for record in self.prediction_history:
                status = "ATTACK" if record['attack_detected'] else " NORMAL"
                print(f"   {record['timestamp']} - #{record['prediction_number']} - {status} ({record['probability']:.2%})")
        
        print("="*80)

def main():
    """Main function for verbose monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verbose BADNET Monitoring - Shows Detailed Prediction Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 verbose_monitor.py --interval 5 --duration 30
  python3 verbose_monitor.py --interval 10 --duration 0
        """
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Monitoring interval in seconds (default: 5)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        help='Monitoring duration in seconds (default: continuous)'
    )
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = VerboseCyberMonitor()
    
    try:
        # Start verbose monitoring
        monitor.monitor_verbose(args.duration, args.interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
    
    except Exception as e:
        print(f"\nError during monitoring: {e}")

if __name__ == "__main__":
    main()
