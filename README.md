# BADNET - Cyber Attack Detection & Monitoring System

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author: ARYA KONER](https://img.shields.io/badge/Author-ARYA%20KONER-green.svg)](https://github.com/aryakoner)

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

- 🤖 **AI-Powered Detection**: Uses 9 different machine learning models
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
   git clone https://github.com/aryakoner/BADNET.git
   cd BADNET
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv cyber_env
   source cyber_env/bin/activate  # On macOS/Linux
   # or
   cyber_env\Scripts\activate     # On Windows
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

## Project Structure

```
BADNET/
├── BADNET.py              # Main CLI application
├── BADNET                 # Launcher script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── run_cyber_attack_system.sh  # Training pipeline
├── save_best_model.py    # Model saving utility
├── cyber_monitor_simple.py     # Monitoring tool
├── demo_ai_connection.py # AI connection demo
├── codes/                # Training code
├── data/                 # Training data
├── processed_data/       # Processed features and models
└── evaluations/          # Model evaluation results
```

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
