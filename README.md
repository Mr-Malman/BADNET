# BADNET - Cyber Attack Detection & Monitoring System

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author: ARYA KONER](https://img.shields.io/badge/Author-ARYA%20KONER-green.svg)](https://github.com/Mr-Malman)

```
██████╗  █████╗ ██████╗ ███╗   ██╗███████╗████████╗
██╔══██╗██╔══██╗██╔══██╗████╗  ██║██╔════╝╚══██╔══╝
██████╔╝███████║██║  ██║██╔██╗ ██║█████╗     ██║   
██╔══██╗██╔══██║██║  ██║██║╚██╗██║██╔══╝     ██║   
██████╔╝██║  ██║██████╔╝██║ ╚████║███████╗   ██║   
╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   
```

**Advanced AI-Powered Cyber Attack Detection & Monitoring System**

## Overview

**BADNET** is a comprehensive AI-powered cyber attack detection and monitoring system designed to provide real-time security intelligence. Built with Python and machine learning, it can detect various types of cyber attacks including DoS, Probe, R2L, and U2R attacks.

### Key Features

- **AI-Powered Detection**: Uses 9 different machine learning models
- **Real-time Monitoring**: System and network activity analysis
- **Intelligent Alerts**: Probability-based attack detection
- **Performance Analytics**: Comprehensive reporting and metrics
- **Easy-to-use CLI**: Beautiful terminal interface with Rich
- **Comprehensive Logging**: Detailed audit trails and analysis

## Quick Start

### Prerequisites

- Python 3.7 or higher
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mr-Malman/BADNET
   cd BADNET
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv cyber_env
   source cyber_env/bin/activate  # On macOS/Linux
   # or
   cyber_env\Scripts\activate     # On Windows
   ```
   ```bash
   source cyber_env/bin/activate
pip install -r requirements.txt
# then run the rest of the setup steps from the script if needed:
bash -x ./setup.sh 2>&1 | tee setup.debug.log
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train AI models**
   ```bash
   ./run_cyber_attack_system.sh
   python3 save_best_model.py
   ```

5. **Verify installation**
   ```bash
   ./BADNET status
   ```

## Usage

### Interactive Mode

Run BADNET without arguments to enter interactive mode:

```bash
./BADNET
```

This will display the main menu with all available options.

### Command Line Mode

Use specific commands for direct execution:

```bash
# Check system status
./BADNET status

# Start monitoring
./BADNET monitor --mode both --interval 5

# View system information
./BADNET system-info

# View logs
./BADNET logs
```

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| `status` | Check system status | `./BADNET status` |
| `train` | Train AI models | `./BADNET train` |
| `monitor` | Start monitoring | `./BADNET monitor --mode both` |
| `demo` | Run AI demo | `./BADNET demo` |
| `system-info` | System information | `./BADNET system-info` |
| `logs` | View logs | `./BADNET logs` |
| `config` | Configure settings | `./BADNET config` |
| `help-extended` | Detailed help | `./BADNET help-extended` |
| `version` | Version info | `./BADNET version` |

### Monitor Options

- `--mode, -m`: Monitoring mode (`system`, `network`, or `both`)
- `--interval, -i`: Monitoring interval in seconds (default: 5)
- `--duration, -d`: Monitoring duration in seconds (optional)
- `--save-log, -s`: Save monitoring log to file (flag)

### Examples

```bash
# Monitor both system and network for 1 hour
./BADNET monitor --mode both --interval 10 --duration 3600

# Monitor system activity and save logs
./BADNET monitor --mode system --interval 5 --save-log

# Quick network monitoring
./BADNET monitor --mode network --interval 3
```

## Monitoring Modes

### System Monitoring
- **CPU Usage**: Real-time CPU utilization
- **Memory Usage**: RAM usage and availability
- **Disk Activity**: Read/write operations
- **Process Count**: Number of running processes
- **System Load**: Overall system performance

### Network Monitoring
- **Connection Count**: Active network connections
- **Packet Rates**: Network packet transmission
- **Byte Rates**: Data transfer rates
- **Port Analysis**: Suspicious port activity
- **Protocol Analysis**: Network protocol monitoring

### Combined Monitoring
- **Comprehensive Analysis**: Full system and network coverage
- **Correlation Detection**: Identify patterns across different metrics
- **Enhanced Accuracy**: Better attack detection through multiple data sources

## Alert System

### Alert Levels

- **WARNING** (50-80% probability): Moderate suspicious activity
- **CRITICAL** (80%+ probability): High probability of attack

### Alert Information

Each alert includes:
- **Timestamp**: Exact time of detection
- **Probability Score**: Confidence level (0-100%)
- **Alert Level**: WARNING or CRITICAL
- **Monitoring Mode**: System, Network, or Both
- **Suspicious Indicators**: Specific metrics that triggered the alert

## Troubleshooting

### Common Issues

1. **Virtual Environment Not Active**
   ```bash
   source cyber_env/bin/activate
   pip install -r requirements.txt
   ```

2. **Model Files Missing**
   ```bash
   ./run_cyber_attack_system.sh
   python3 save_best_model.py
   ```

3. **Permission Denied**
   ```bash
   # On macOS, grant permissions in System Preferences
   # On Linux, run with appropriate permissions
   sudo ./BADNET monitor
   ```

4. **High CPU Usage**
   ```bash
   # Increase monitoring interval
   ./BADNET monitor --interval 10
   ```

# 🚀 BADNET QUICK REFERENCE CARD

## **Essential Commands - All in One Place!**

### **🔧 Basic Commands**
```bash
./BADNET status              # Check system health
./BADNET train               # Train AI models
./BADNET demo                # Test AI connection
./BADNET system-info         # View system status
./BADNET version             # Show version info
./BADNET help-extended       # Detailed help
```

### **📊 Monitoring Commands**
```bash
# Basic Monitoring
./BADNET monitor --mode both --interval 10 --duration 3600

# Verbose Monitoring (Detailed Results)
./BADNET verbose-monitor --interval 5 --duration 30
./BADNET verbose-monitor --interval 10 --duration 0

# 24/7 Continuous Monitoring
./BADNET monitor-24x7
./BADNET monitor-background
```

### **📜 Direct Script Usage**
```bash
# Verbose Monitoring Script
./verbose_monitor.sh

# 24/7 Monitoring Script
./monitor_24x7.sh

# Python Script Direct
python3 verbose_monitor.py --interval 5 --duration 30
```

### **🎮 Interactive Menu**
```bash
./BADNET                    # No arguments = Interactive menu
```

---

## **📋 What Each Command Shows**

### **Basic Monitor Output:**
```
[22:16:46] CPU: 14.2% | Mem: 74.8% | Conn: 0 | Attacks: 0/3
```

### **Verbose Monitor Output:**
```
TIMESTAMP: 2025-08-24 22:16:46
PREDICTION #2
SYSTEM STATUS: CPU: 14.2%, Memory: 74.8%, Process Count: 483
NETWORK STATUS: Connections: 0, Packets: 971,795, Traffic: HIGH
AI PREDICTION: Attack Detected: NO, Probability: 0.00%, Threat: NORMAL
PREDICTION HISTORY: Timeline of all results
```

---

## **⚡ Quick Start Commands**

### **1. First Time Setup**
```bash
./setup.sh
./BADNET status
```

### **2. Train AI Models**
```bash
./BADNET train
```

### **3. Test Monitoring (30 seconds)**
```bash
./BADNET verbose-monitor --interval 3 --duration 30
```

### **4. Start 24/7 Monitoring**
```bash
./BADNET monitor-24x7
```

---

## **🔍 Monitoring Modes**

| Mode | Command | What You See |
|------|---------|--------------|
| **Basic** | `./BADNET monitor` | Simple status updates |
| **Verbose** | `./BADNET verbose-monitor` | **Full details, timestamps, history** |
| **24/7** | `./BADNET monitor-24x7` | Continuous operation |
| **Background** | `./BADNET monitor-background` | Runs in background |

---

## **📊 Verbose Monitoring Features**

✅ **Timestamps** - Exact time for each prediction  
✅ **Prediction Numbers** - #1, #2, #3... tracking  
✅ **Attack Results** - YES/NO for each prediction  
✅ **Probability** - Exact percentages (0.00%, 25.5%, etc.)  
✅ **Threat Levels** - NORMAL, LOW_SUSPICION, MODERATE_SUSPICION, HIGH_PROBABILITY_ATTACK  
✅ **System Status** - CPU, Memory, Processes, Disk  
✅ **Network Status** - Connections, Packets, Traffic  
✅ **History** - Timeline of all results  

---

## **⏱️ Recommended Intervals**

| Use Case | Interval | Duration | Command |
|----------|----------|----------|---------|
| **Quick Test** | 2-3 seconds | 30 seconds | `--interval 3 --duration 30` |
| **Standard** | 5-10 seconds | 5 minutes | `--interval 5 --duration 300` |
| **Production** | 10-30 seconds | Continuous | `--interval 15 --duration 0` |
| **24/7** | 10 seconds | Forever | `./BADNET monitor-24x7` |

---

## **🚨 Alert Levels**

| Probability | Level | Color | Action |
|-------------|-------|-------|--------|
| **0-20%** | NORMAL | 🟢 | Continue monitoring |
| **21-50%** | LOW_SUSPICION | 🟡 | Watch closely |
| **51-80%** | MODERATE_SUSPICION | 🟠 | Investigate |
| **81-100%** | HIGH_PROBABILITY_ATTACK | 🔴 | **IMMEDIATE ACTION** |

---

## **💡 Pro Tips**

1. **Start with verbose monitoring** to see all details
2. **Use 5-10 second intervals** for best performance
3. **Run 24/7 monitoring** for production deployment
4. **Check system status first** before monitoring
5. **Use interactive menu** for easy navigation

---

## **📞 Need Help?**

```bash
./BADNET --help              # Basic help
./BADNET help-extended       # Detailed help
./BADNET                     # Interactive menu
```

**🎯 Remember: Use `verbose-monitor` to see ALL the prediction details, timestamps, and results!**

# BADNET - Complete Command Reference & Help Guide

## 🚀 **BADNET Cyber Attack Detection & Monitoring System**

**Author:** ARYA KONER  
**Version:** 1.0.0  
**Description:** AI-Powered Cyber Attack Detection & Monitoring System

---

## 📋 **TABLE OF CONTENTS**

1. [Basic Commands](#basic-commands)
2. [Advanced Monitoring Commands](#advanced-monitoring-commands)
3. [Interactive Menu Options](#interactive-menu-options)
4. [Command Line Usage](#command-line-usage)
5. [Script Usage](#script-usage)
6. [Monitoring Modes](#monitoring-modes)
7. [Verbose Monitoring Features](#verbose-monitoring-features)
8. [24/7 Operation](#247-operation)
9. [Examples & Use Cases](#examples--use-cases)
10. [Troubleshooting](#troubleshooting)

---

## 🔧 **BASIC COMMANDS**

### **Core System Commands**

| Command | Description | Usage |
|---------|-------------|-------|
| `status` | Check system status and component availability | `./BADNET status` |
| `train` | Train AI models using cyber attack dataset | `./BADNET train` |
| `demo` | Run AI model connection demonstration | `./BADNET demo` |
| `system-info` | Display system information and current status | `./BADNET system-info` |
| `logs` | View and manage monitoring logs | `./BADNET logs` |
| `config` | Configure BADNET settings and preferences | `./BADNET config` |
| `help-extended` | Show detailed help and usage examples | `./BADNET help-extended` |
| `version` | Show BADNET version and information | `./BADNET version` |

### **Basic Monitoring Command**

| Command | Description | Usage |
|---------|-------------|-------|
| `monitor` | Start real-time cyber attack monitoring | `./BADNET monitor --mode both --interval 10` |

---

## 🚀 **ADVANCED MONITORING COMMANDS**

### **Verbose Monitoring Commands**

| Command | Description | Usage |
|---------|-------------|-------|
| `verbose-monitor` | Run verbose monitoring with detailed prediction results | `./BADNET verbose-monitor --interval 5 --duration 30` |
| `monitor-24x7` | Start 24/7 continuous monitoring service | `./BADNET monitor-24x7` |
| `monitor-background` | Run monitoring in background mode | `./BADNET monitor-background` |

### **Direct Script Usage**

| Script | Description | Usage |
|--------|-------------|-------|
| `verbose_monitor.sh` | Run verbose monitoring script directly | `./verbose_monitor.sh` |
| `monitor_24x7.sh` | Run 24/7 monitoring script directly | `./monitor_24x7.sh` |
| `verbose_monitor.py` | Run Python verbose monitor directly | `python3 verbose_monitor.py --interval 5` |

---

## 🎮 **INTERACTIVE MENU OPTIONS**

When you run `./BADNET` without arguments, you get an interactive menu:

```
BADNET Main Menu
1. Check System Status
2. Train AI Models
3. Start Basic Monitoring
4. Start Verbose Monitoring
5. Start 24/7 Monitoring
6. Start Background Monitoring
7. AI Model Demo
8. System Information
9. View Logs
10. Configuration
11. Help
12. Version Info
0. Exit
```

---

## 💻 **COMMAND LINE USAGE**

### **Basic Monitoring**

```bash
# Monitor both system and network for 1 hour
./BADNET monitor --mode both --interval 10 --duration 3600

# Monitor system only for 30 minutes
./BADNET monitor --mode system --interval 5 --duration 1800

# Monitor network only continuously
./BADNET monitor --mode network --interval 15 --duration 0
```

### **Verbose Monitoring**

```bash
# Verbose monitoring for 30 seconds with 5-second intervals
./BADNET verbose-monitor --interval 5 --duration 30

# Continuous verbose monitoring with 10-second intervals
./BADNET verbose-monitor --interval 10 --duration 0

# Quick verbose monitoring with 3-second intervals
./BADNET verbose-monitor --interval 3 --duration 15
```

### **24/7 Monitoring**

```bash
# Start 24/7 monitoring service
./BADNET monitor-24x7

# Start background monitoring
./BADNET monitor-background

# Run 24/7 script directly
./monitor_24x7.sh
```

---

## 📜 **SCRIPT USAGE**

### **Verbose Monitor Script**

```bash
# Make executable and run
chmod +x verbose_monitor.sh
./verbose_monitor.sh

# This will show:
# • Timestamps for each prediction
# • Attack detection results (YES/NO)
# • Attack probability percentages
# • Threat levels
# • System status (CPU, Memory, Process Count, Disk Activity)
# • Network status (Connections, Packets, Traffic Level)
# • Prediction history timeline
```

### **24/7 Monitor Script**

```bash
# Make executable and run
chmod +x monitor_24x7.sh
./monitor_24x7.sh

# This will:
# • Run continuous monitoring
# • Auto-restart on failures
# • Log all activities
# • Run indefinitely until stopped
```

### **Python Script Direct**

```bash
# Activate virtual environment first
source cyber_env/bin/activate

# Run verbose monitoring
python3 verbose_monitor.py --interval 5 --duration 30

# Run with different parameters
python3 verbose_monitor.py --interval 10 --duration 0
```

---

## 🔍 **MONITORING MODES**

### **Basic Monitoring Modes**

| Mode | Description | What it Monitors |
|------|-------------|------------------|
| `system` | System activity monitoring | CPU, memory, disk, processes |
| `network` | Network activity monitoring | Connections, traffic, ports |
| `both` | Comprehensive monitoring | Both system and network (recommended) |

### **Advanced Monitoring Types**

| Type | Description | Output Level |
|------|-------------|--------------|
| `basic` | Standard monitoring | Basic status updates |
| `verbose` | Detailed monitoring | Full prediction results, timestamps, history |
| `24x7` | Continuous service | Background operation with auto-restart |
| `background` | Background monitoring | Logged to file, runs in background |

---

## 📊 **VERBOSE MONITORING FEATURES**

### **What You Get with Verbose Monitoring**

✅ **Detailed Timestamps** - Exact time for each prediction  
✅ **Prediction Numbers** - Sequential tracking (#1, #2, #3...)  
✅ **Attack Detection Results** - Clear YES/NO for each prediction  
✅ **Probability Percentages** - Exact attack probability (0.00%, 25.5%, etc.)  
✅ **Threat Levels** - NORMAL, LOW_SUSPICION, MODERATE_SUSPICION, HIGH_PROBABILITY_ATTACK  
✅ **System Status** - CPU, Memory, Process Count, Disk Activity  
✅ **Network Status** - Connections, Packets, Traffic Level  
✅ **Prediction History** - Timeline of all monitoring results  

### **Sample Verbose Output**

```
================================================================================
TIMESTAMP: 2025-08-24 22:03:36
PREDICTION #2
================================================================================
SYSTEM STATUS:
   CPU Usage: 30.6%
   Memory Usage: 74.5%
   Process Count: 485
   Disk Activity: HIGH

NETWORK STATUS:
   Connections: 0
   Packets Sent: 958,790
   Packets Received: 1,129,817
   Traffic Level: HIGH

AI PREDICTION RESULTS:
   Attack Detected: NO
   Attack Probability: 0.00%
   Threat Level: NORMAL

RECENT PREDICTION HISTORY:
   #2: 2025-08-24 22:03:36 - NORMAL (0.00%)
   #3: 2025-08-24 22:03:40 - NORMAL (0.00%)
================================================================================
```

---

## ⏰ **24/7 OPERATION**

### **Methods for 24/7 Monitoring**

#### **Method 1: BADNET CLI Command**
```bash
./BADNET monitor-24x7
```

#### **Method 2: Direct Script Execution**
```bash
./monitor_24x7.sh
```

#### **Method 3: Background Service**
```bash
./BADNET monitor-background
```

#### **Method 4: System Service (Linux)**
```bash
sudo cp badnet.service /etc/systemd/system/
sudo systemctl enable badnet
sudo systemctl start badnet
```

### **24/7 Features**

- **Continuous Operation** - Runs indefinitely
- **Auto-Restart** - Automatically restarts on failures
- **Resource Management** - Efficient memory and CPU usage
- **Logging** - Comprehensive activity logging
- **Health Checks** - Continuous system validation

---

## 📝 **EXAMPLES & USE CASES**

### **Quick Testing (5 minutes)**
```bash
./BADNET verbose-monitor --interval 3 --duration 300
```

### **Production Monitoring (1 hour)**
```bash
./BADNET monitor --mode both --interval 10 --duration 3600
```

### **Development Testing (30 seconds)**
```bash
python3 verbose_monitor.py --interval 2 --duration 30
```

### **24/7 Production Deployment**
```bash
./BADNET monitor-24x7
```

### **Background Operation**
```bash
./BADNET monitor-background
# Check status: ps aux | grep verbose_monitor
# View logs: tail -f monitoring.log
```

---

## 🛠️ **TROUBLESHOOTING**

### **Common Issues & Solutions**

#### **Issue: "Virtual environment not found"**
```bash
# Solution: Run setup script
./setup.sh
```

#### **Issue: "Module not found"**
```bash
# Solution: Activate virtual environment
source cyber_env/bin/activate
pip install -r requirements.txt
```

#### **Issue: "Script not executable"**
```bash
# Solution: Make scripts executable
chmod +x verbose_monitor.sh
chmod +x monitor_24x7.sh
```

#### **Issue: "Model files not found"**
```bash
# Solution: Train models first
./BADNET train
```

### **Performance Optimization**

- **Fast Monitoring**: 2-3 second intervals (for testing)
- **Standard Monitoring**: 5-10 second intervals (recommended)
- **Production Monitoring**: 10-30 second intervals (for 24/7)

### **Resource Usage**

- **Memory**: ~67MB during monitoring
- **CPU**: <1% during normal operation
- **Disk**: Minimal (only logging)
- **Network**: Local monitoring only

---

## 🎯 **QUICK START GUIDE**

### **1. First Time Setup**
```bash
./setup.sh
./BADNET status
```

### **2. Train AI Models**
```bash
./BADNET train
```

### **3. Test Basic Monitoring**
```bash
./BADNET monitor --mode both --interval 5 --duration 30
```

### **4. Test Verbose Monitoring**
```bash
./BADNET verbose-monitor --interval 5 --duration 30
```

### **5. Start 24/7 Monitoring**
```bash
./BADNET monitor-24x7
```

---

## 📞 **SUPPORT & HELP**

### **Getting Help**
```bash
# Basic help
./BADNET --help

# Detailed help
./BADNET help-extended

# Interactive menu
./BADNET
```

### **Documentation Files**
- `README.md` - Basic project information
- `DEPLOYMENT_GUIDE.md` - Installation and deployment
- `BADNET_COMPLETE_HELP.md` - This comprehensive help file

### **Author Information**
- **Author**: ARYA KONER
- **GitHub**: https://github.com/hackarya
- **Project**: BADNET - AI-Powered Cyber Attack Detection & Monitoring System

---

**🎉 You now have access to ALL BADNET commands and features! Use the interactive menu or command line for the best experience.**


## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**ARYA KONER**

- **GitHub**: [@aryakoner](https://github.com/aryakoner)
- **Linkedin**: [@aryakoner](https://www.linkedin.com/in/hackarya007/)
- **Project**: BADNET | AI-Powered Cyber Surveillance System

## Acknowledgments

- **Machine Learning**: scikit-learn team
- **CLI Framework**: Typer and Rich libraries
- **System Monitoring**: psutil library
- **Open Source Community**: All contributors and maintainers

## Support

If you encounter any issues:

1. **Check the documentation**: This README and help commands
2. **Review logs**: Use `./BADNET logs` to check for errors
3. **Run diagnostics**: Use `./BADNET status` to verify system health
4. **Open an issue**: Create a GitHub issue with detailed information

---

**BADNET - Protecting systems with AI-powered intelligence since 2025** 
# BADNET
# BADNET
