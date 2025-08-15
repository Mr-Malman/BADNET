#!/usr/bin/env python3
"""
AI Model Connection Demonstration
"""

import joblib
import numpy as np
import pandas as pd
import psutil
from datetime import datetime

def demonstrate_ai_connection():
    """Demonstrate the complete AI model connection process."""
    
    print("AI Model Connection Demonstration")
    print("=" * 50)
    
    # Step 1: Load the trained model
    print("\nStep 1: Loading the Trained Model")
    print("-" * 30)
    
    try:
        model = joblib.load('processed_data/supervised_learning/best_model.pkl')
        print(f"Model loaded: {type(model).__name__}")
        print(f"Model parameters: {model.get_params()}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Step 2: Load the feature scaler
    print("\nStep 2: Loading the Feature Scaler")
    print("-" * 30)
    
    try:
        scaler = joblib.load('processed_data/supervised_learning/feature_scaler.pkl')
        print(f"Scaler loaded: {type(scaler).__name__}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return
    
    # Step 3: Load feature names
    print("\nStep 3: Loading Feature Names")
    print("-" * 30)
    
    try:
        with open('processed_data/supervised_learning/train_features.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(feature_names)} features")
        print(f"First 5 features: {feature_names[:5]}")
    except Exception as e:
        print(f"Error loading features: {e}")
        return
    
    # Step 4: Extract live system features
    print("\nStep 4: Extracting Live System Features")
    print("-" * 30)
    
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()
        
        # Create feature dictionary
        live_features = {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_available': memory.available / (1024**3),
            'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
            'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'high_cpu': 1 if cpu_percent > 80 else 0,
            'high_memory': 1 if memory.percent > 80 else 0,
            'high_traffic': 1 if (net_io.packets_sent + net_io.packets_recv) > 10000 else 0,
            'disk_activity': 1 if disk_io and (disk_io.read_bytes + disk_io.write_bytes) > 1000000 else 0,
        }
        
        print(f"Extracted {len(live_features)} live features")
        print(f"CPU Usage: {cpu_percent:.1f}%")
        print(f"Memory Usage: {memory.percent:.1f}%")
        print(f"Network Packets: {net_io.packets_sent + net_io.packets_recv}")
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return
    
    # Step 5: Preprocess features
    print("\nStep 5: Preprocessing Features")
    print("-" * 30)
    
    try:
        # Create DataFrame with live features
        df = pd.DataFrame([live_features])
        
        # Add missing features with default values
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Ensure correct order
        df = df[feature_names]
        
        print(f"Created DataFrame with {df.shape[1]} features")
        print(f"Feature order matches training data")
        
        # Apply scaling
        df_scaled = scaler.transform(df)
        
        print(f"Features scaled successfully")
        print(f"Scaled shape: {df_scaled.shape}")
        
    except Exception as e:
        print(f"Error preprocessing features: {e}")
        return
    
    # Step 6: Make prediction
    print("\nStep 6: Making AI Prediction")
    print("-" * 30)
    
    try:
        # Make prediction
        prediction = model.predict(df_scaled)
        probability = model.predict_proba(df_scaled)[0][1]
        
        print(f"Prediction made successfully")
        print(f"Attack Detected: {bool(prediction[0])}")
        print(f"Attack Probability: {probability:.2%}")
        
        # Determine alert level
        if probability > 0.8:
            alert_level = "CRITICAL"
        elif probability > 0.5:
            alert_level = " WARNING"
        else:
            alert_level = "NORMAL"
        
        print(f"Alert Level: {alert_level}")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return
    
    # Step 7: Show complete connection summary
    print("\nStep 7: Connection Summary")
    print("-" * 30)
    
    print("AI Model Connection Successful!")
    print(f"Model Type: {type(model).__name__}")
    print(f"Features: {len(feature_names)}")
    print(f"Prediction: {'ATTACK' if bool(prediction[0]) else 'NORMAL'}")
    print(f"Confidence: {probability:.2%}")
    print(f"Alert: {alert_level}")
    
    # Step 8: Real-time simulation
    print("\nStep 8: Real-Time Simulation")
    print("-" * 30)
    
    print("Simulating real-time monitoring for 10 seconds...")
    print("Press Ctrl+C to stop")
    
    try:
        for i in range(10):
            # Get updated features
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Update features
            live_features['cpu_usage'] = cpu_percent
            live_features['memory_usage'] = memory.percent
            live_features['high_cpu'] = 1 if cpu_percent > 80 else 0
            live_features['high_memory'] = 1 if memory.percent > 80 else 0
            
            # Preprocess and predict
            df = pd.DataFrame([live_features])
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            df = df[feature_names]
            df_scaled = scaler.transform(df)
            prediction = model.predict(df_scaled)
            probability = model.predict_proba(df_scaled)[0][1]
            
            # Display status
            timestamp = datetime.now().strftime("%H:%M:%S")
            status = f"[{timestamp}] CPU: {cpu_percent:.1f}% | Mem: {memory.percent:.1f}% | Attack: {bool(prediction[0])} ({probability:.1%})"
            print(f"\r{status}", end="", flush=True)
            
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")
    
    print("\n\nAI Model Connection Demonstration Complete!")
    print("=" * 50)
    print("The AI model is successfully connected and making real-time predictions!")

if __name__ == "__main__":
    demonstrate_ai_connection()
