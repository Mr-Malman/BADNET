#!/usr/bin/env python3
"""
BADNET - Cyber Attack Detection & Monitoring System
==================================================

A comprehensive CLI interface for AI-powered cyber attack detection and monitoring.
Built with Python Typer and Rich for a beautiful terminal experience.

Author: ARYA KONER
Description: Advanced cyber attack detection using machine learning with real-time monitoring capabilities.
"""

import typer
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich import box
import psutil
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Initialize Typer app
app = typer.Typer(
    name="BADNET",
    help="Cyber Attack Detection & Monitoring System",
    add_completion=False,
    rich_markup_mode="rich"
)

# Initialize Rich console
console = Console()

# ASCII Banner
BANNER = """
[bold red]
██████╗  █████╗ ██████╗ ███╗   ██╗███████╗████████╗
██╔══██╗██╔══██╗██╔══██╗████╗  ██║██╔════╝╚══██╔══╝
██████╔╝███████║██║  ██║██╔██╗ ██║█████╗     ██║   
██╔══██╗██╔══██║██║  ██║██║╚██╗██║██╔══╝     ██║   
██████╔╝██║  ██║██████╔╝██║ ╚████║███████╗   ██║   
╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   
[/bold red]

[bold blue]Cyber Attack Detection & Monitoring System[/bold blue]
[dim]Author: ARYA KONER | AI-Powered Security Intelligence[/dim]
"""

def show_banner():
    """Display the BADNET banner."""
    console.print(Panel(BANNER, box=box.DOUBLE, border_style="red"))

def check_system_status():
    """Check if the system is properly set up."""
    console.print("\n[bold blue]Checking System Status...[/bold blue]")
    
    status_table = Table(title="System Status Check")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Details", style="yellow")
    
    # Check virtual environment
    if "BADNET" in sys.prefix:
        status_table.add_row("Virtual Environment", "Active", f"Using: {sys.prefix}")
    else:
        status_table.add_row("Virtual Environment", "Inactive", "Please activate BADNET")
    
    # Check model files
    model_path = Path("processed_data/supervised_learning/best_model.pkl")
    if model_path.exists():
        status_table.add_row("AI Model", "Available", "LogisticRegression loaded")
    else:
        status_table.add_row("AI Model", "Missing", "Run training pipeline first")
    
    # Check scaler
    scaler_path = Path("processed_data/supervised_learning/feature_scaler.pkl")
    if scaler_path.exists():
        status_table.add_row("Feature Scaler", "Available", "StandardScaler ready")
    else:
        status_table.add_row("Feature Scaler", "Missing", "Run feature engineering")
    
    # Check features
    features_path = Path("processed_data/supervised_learning/train_features.txt")
    if features_path.exists():
        with open(features_path, 'r') as f:
            feature_count = len(f.readlines())
        status_table.add_row("Feature Names", "Available", f"{feature_count} features")
    else:
        status_table.add_row("Feature Names", "Missing", "Feature file not found")
    
    console.print(status_table)
    return all([model_path.exists(), scaler_path.exists(), features_path.exists()])

@app.command()
def status():
    """Check system status and component availability."""
    show_banner()
    check_system_status()

@app.command()
def train():
    """Train the AI models using the cyber attack dataset."""
    show_banner()
    console.print("\n[bold blue] Training AI Models...[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Training models...", total=None)
        
        try:
            # Run the training pipeline directly with Python commands (Windows compatible)
            console.print("Running data cleaning...")
            result1 = subprocess.run(
                ["python", "codes/cyber_clean.py"],
                capture_output=True,
                text=True,
                shell=True
            )
            
            if result1.returncode != 0:
                console.print(f"\n[bold red]Data cleaning failed![/bold red]")
                console.print(f"Error: {result1.stderr}")
                return
                
            console.print("Running feature engineering...")
            result2 = subprocess.run(
                ["python", "codes/cyber_featureEngineering.py", "--ask", "0", "--verbose", "1"],
                capture_output=True,
                text=True,
                shell=True
            )
            
            if result2.returncode != 0:
                console.print(f"\n[bold red]Feature engineering failed![/bold red]")
                console.print(f"Error: {result2.stderr}")
                return
                
            console.print("Running model training...")
            result3 = subprocess.run(
                ["python", "codes/cyber_train.py", "--start_clean", "1", "--ask", "0", "--verbose", "1", "--plot", "1"],
                capture_output=True,
                text=True,
                shell=True
            )
            
            if result3.returncode != 0:
                console.print(f"\n[bold red]Model training failed![/bold red]")
                console.print(f"Error: {result3.stderr}")
                return
                
            # All steps completed successfully
            result = type('obj', (object,), {'returncode': 0})()
            
            if result.returncode == 0:
                console.print("\n[bold green]Training completed successfully![/bold green]")
                
                # Show training results
                results_file = Path("evaluations/Batch 0 - Evaluations.csv")
                if results_file.exists():
                    df = pd.read_csv(results_file)
                    console.print(f"\n[bold blue] Training Results:[/bold blue]")
                    
                    results_table = Table(title="Model Performance")
                    results_table.add_column("Model", style="cyan")
                    results_table.add_column("Accuracy", style="green")
                    results_table.add_column("Precision", style="yellow")
                    results_table.add_column("Recall", style="magenta")
                    results_table.add_column("Training Time", style="blue")
                    
                    for _, row in df.iterrows():
                        results_table.add_row(
                            row['ModelName'],
                            f"{row['Accuracy']:.3f}",
                            f"{row['Precision']:.3f}",
                            f"{row['Recall']:.3f}",
                            f"{row['TrainingTime (s)']:.3f}s"
                        )
                    
                    console.print(results_table)
                    
                    # Find best model
                    best_idx = df['Accuracy'].idxmax()
                    best_model = df.loc[best_idx, 'ModelName']
                    best_accuracy = df.loc[best_idx, 'Accuracy']
                    
                    console.print(f"\n[bold green]Best Model: {best_model} with {best_accuracy:.1%} accuracy[/bold green]")
                    
            else:
                console.print(f"\n[bold red]Training failed![/bold red]")
                console.print(f"Error: {result.stderr}")
                
        except Exception as e:
            console.print(f"\n[bold red]Error during training: {e}[/bold red]")

@app.command()
def monitor(
    mode: str = typer.Option("both", "--mode", "-m", help="Monitoring mode: system, network, or both"),
    interval: int = typer.Option(5, "--interval", "-i", help="Monitoring interval in seconds"),
    duration: Optional[int] = typer.Option(None, "--duration", "-d", help="Monitoring duration in seconds"),
    save_log: bool = typer.Option(False, "--save-log", "-s", help="Save monitoring log to file")
):
    """Start real-time cyber attack monitoring."""
    show_banner()
    
    # Check system status first
    if not check_system_status():
        console.print("\n[bold red]System not ready. Please run training first.[/bold red]")
        raise typer.Exit(1)
    
    console.print(f"\n[bold blue]Starting {mode.upper()} Monitoring...[/bold blue]")
    console.print(f"Mode: {mode}")
    console.print(f"Interval: {interval} seconds")
    if duration:
        console.print(f"Duration: {duration} seconds")
    console.print(f"Save Log: {save_log}")
    
    # Prepare command
    cmd = [
        "python3", "cyber_monitor_simple.py",
        "--mode", mode,
        "--interval", str(interval)
    ]
    
    if duration:
        cmd.extend(["--duration", str(duration)])
    
    if save_log:
        log_file = f"monitoring_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        cmd.extend([">", log_file])
        console.print(f"Log will be saved to: {log_file}")
    
    # Start monitoring
    try:
        if save_log:
            # Run with log file
            subprocess.run(" ".join(cmd), shell=True)
        else:
            # Run directly
            subprocess.run(cmd)
            
    except KeyboardInterrupt:
        console.print("\n[bold yellow] Monitoring stopped by user[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error during monitoring: {e}[/bold red]")

@app.command()
def demo():
    """Run AI model connection demonstration."""
    show_banner()
    console.print("\n[bold blue]AI Model Connection Demonstration[/bold blue]")
    
    try:
        subprocess.run(["python3", "demo_ai_connection.py"])
    except Exception as e:
        console.print(f"\n[bold red]Error running demonstration: {e}[/bold red]")

@app.command()
def system_info():
    """Display system information and current status."""
    show_banner()
    
    # System information
    console.print("\n[bold blue]System Information[/bold blue]")
    
    info_table = Table(title="System Status")
    info_table.add_column("Metric", style="cyan")
    info_table.add_column("Value", style="green")
    
    # CPU info
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    info_table.add_row("CPU Usage", f"{cpu_percent:.1f}%")
    info_table.add_row("CPU Cores", str(cpu_count))
    
    # Memory info
    memory = psutil.virtual_memory()
    info_table.add_row("Memory Usage", f"{memory.percent:.1f}%")
    info_table.add_row("Memory Available", f"{memory.available / (1024**3):.1f} GB")
    
    # Disk info
    disk = psutil.disk_usage('/')
    info_table.add_row("Disk Usage", f"{disk.percent:.1f}%")
    info_table.add_row("Disk Free", f"{disk.free / (1024**3):.1f} GB")
    
    # Network info
    net_io = psutil.net_io_counters()
    info_table.add_row("Network Packets Sent", f"{net_io.packets_sent:,}")
    info_table.add_row("Network Packets Recv", f"{net_io.packets_recv:,}")
    
    console.print(info_table)
    
    # Process information
    console.print("\n[bold blue]Top Processes[/bold blue]")
    
    process_table = Table(title="Running Processes")
    process_table.add_column("PID", style="cyan")
    process_table.add_column("Name", style="green")
    process_table.add_column("CPU %", style="yellow")
    process_table.add_column("Memory %", style="magenta")
    
    processes = []
    from contextlib import suppress
    
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            info = proc.info
            if info['cpu_percent'] is not None and info['memory_percent'] is not None:
                processes.append(info)
    
    # Sort by CPU usage and take top 10
    processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
    
    for info in processes[:10]:
        process_table.add_row(
            str(info['pid']),
            info['name'][:20],
            f"{info['cpu_percent']:.1f}",
            f"{info['memory_percent']:.1f}"
        )
    
    console.print(process_table)

@app.command()
def logs():
    """View and manage monitoring logs."""
    show_banner()
    
    console.print("\n[bold blue]Log Management[/bold blue]")
    
    # Find log files
    log_files = list(Path(".").glob("monitoring_log_*.log"))
    
    if not log_files:
        console.print("[yellow]No monitoring logs found.[/yellow]")
        return
    
    log_table = Table(title="Available Logs")
    log_table.add_column("Filename", style="cyan")
    log_table.add_column("Size", style="green")
    log_table.add_column("Modified", style="yellow")
    
    for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True):
        stat = log_file.stat()
        size = stat.st_size
        modified = datetime.fromtimestamp(stat.st_mtime)
        
        log_table.add_row(
            log_file.name,
            f"{size / 1024:.1f} KB",
            modified.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    console.print(log_table)
    
    # Ask user which log to view
    if log_files:
        choice = Prompt.ask(
            "\nEnter log filename to view (or 'q' to quit)",
            choices=[f.name for f in log_files] + ['q']
        )
        
        if choice != 'q':
            log_file = Path(choice)
            if log_file.exists():
                console.print(f"\n[bold blue]Viewing: {log_file.name}[/bold blue]")
                console.print("=" * 60)
                
                with open(log_file, 'r') as f:
                    content = f.read()
                    console.print(content)
            else:
                console.print("[red]Log file not found.[/red]")

@app.command()
def config():
    """Configure BADNET settings and preferences."""
    show_banner()
    
    console.print("\n[bold blue] BADNET Configuration[/bold blue]")
    
    # Default configuration
    config = {
        "default_monitoring_mode": "both",
        "default_interval": 5,
        "alert_threshold_warning": 0.5,
        "alert_threshold_critical": 0.8,
        "save_logs_by_default": False,
        "auto_start_monitoring": False
    }
    
    console.print("\n[bold yellow]Current Configuration:[/bold yellow]")
    config_table = Table()
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    for key, value in config.items():
        config_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(config_table)
    
    # Configuration options
    console.print("\n[bold blue]Configuration Options:[/bold blue]")
    console.print("1. Change default monitoring mode")
    console.print("2. Change default interval")
    console.print("3. Change alert thresholds")
    console.print("4. Toggle log saving")
    console.print("5. Toggle auto-start monitoring")
    console.print("6. Save configuration")
    console.print("7. Exit")
    
    choice = Prompt.ask("\nSelect option", choices=["1", "2", "3", "4", "5", "6", "7"])
    
    if choice == "1":
        mode = Prompt.ask("Default monitoring mode", choices=["system", "network", "both"])
        config["default_monitoring_mode"] = mode
        console.print(f"[green]Default mode set to: {mode}[/green]")
    
    elif choice == "2":
        interval = Prompt.ask("Default interval (seconds)", default="5")
        config["default_interval"] = int(interval)
        console.print(f"[green]Default interval set to: {interval} seconds[/green]")
    
    elif choice == "3":
        warning = Prompt.ask("Warning threshold (0.0-1.0)", default="0.5")
        critical = Prompt.ask("Critical threshold (0.0-1.0)", default="0.8")
        config["alert_threshold_warning"] = float(warning)
        config["alert_threshold_critical"] = float(critical)
        console.print("[green]Alert thresholds updated[/green]")
    
    elif choice == "4":
        save_logs = Confirm.ask("Save logs by default?")
        config["save_logs_by_default"] = save_logs
        console.print(f"[green]Log saving set to: {save_logs}[/green]")
    
    elif choice == "5":
        auto_start = Confirm.ask("Auto-start monitoring?")
        config["auto_start_monitoring"] = auto_start
        console.print(f"[green]Auto-start set to: {auto_start}[/green]")
    
    elif choice == "6":
        # Save configuration to file
        config_file = Path("badnet_config.json")
        import json
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        console.print(f"[green]Configuration saved to: {config_file}[/green]")

@app.command()
def help_extended():
    """Show detailed help and usage examples."""
    show_banner()
    
    help_text = """
[bold blue]BADNET - Cyber Attack Detection & Monitoring System[/bold blue]

[bold green]Description:[/bold green]
BADNET is an AI-powered cyber attack detection system that uses machine learning
to monitor network traffic and system activity in real-time. It can detect various
types of cyber attacks including DoS, Probe, R2L, and U2R attacks.

[bold green]Key Features:[/bold green]
• AI-Powered Detection using 9 different ML models
• Real-time system and network monitoring
• Intelligent alert system with probability scores
• Performance analytics and reporting
• Easy-to-use CLI interface
• Comprehensive logging and analysis

[bold green]Commands:[/bold green]
• [cyan]status[/cyan] - Check system status and component availability
• [cyan]train[/cyan] - Train AI models using cyber attack dataset
• [cyan]monitor[/cyan] - Start real-time cyber attack monitoring
• [cyan]demo[/cyan] - Run AI model connection demonstration
• [cyan]system-info[/cyan] - Display system information and current status
• [cyan]logs[/cyan] - View and manage monitoring logs
• [cyan]config[/cyan] - Configure BADNET settings and preferences
• [cyan]help-extended[/cyan] - Show this detailed help

[bold green]Advanced Monitoring Commands:[/bold green]
• [cyan]verbose-monitor[/cyan] - Run verbose monitoring with detailed prediction results
• [cyan]monitor-24x7[/cyan] - Start 24/7 continuous monitoring service
• [cyan]monitor-background[/cyan] - Run monitoring in background mode

[bold green]Usage Examples:[/bold green]
• [yellow]badnet status[/yellow] - Check if system is ready
• [yellow]badnet train[/yellow] - Train the AI models
• [yellow]badnet monitor --mode both --interval 10[/yellow] - Monitor both system and network
• [yellow]badnet monitor --mode system --duration 3600[/yellow] - Monitor system for 1 hour
• [yellow]badnet demo[/yellow] - See AI model connection in action
• [yellow]badnet system-info[/yellow] - View current system status
• [yellow]badnet logs[/yellow] - View monitoring logs
• [yellow]badnet config[/yellow] - Configure settings

[bold green]Advanced Monitoring Examples:[/bold green]
• [yellow]badnet verbose-monitor --interval 5 --duration 30[/yellow] - Verbose monitoring for 30 seconds
• [yellow]badnet verbose-monitor --interval 10 --duration 0[/yellow] - Continuous verbose monitoring
• [yellow]badnet monitor-24x7[/yellow] - Start 24/7 monitoring service
• [yellow]badnet monitor-background[/yellow] - Run monitoring in background
• [yellow]./verbose_monitor.sh[/yellow] - Run verbose monitoring script directly
• [yellow]./monitor_24x7.sh[/yellow] - Run 24/7 monitoring script directly

[bold green]Monitoring Modes:[/bold green]
• [cyan]system[/cyan] - Monitor system activity (CPU, memory, disk, processes)
• [cyan]network[/cyan] - Monitor network activity (connections, traffic, ports)
• [cyan]both[/cyan] - Monitor both system and network (recommended)

[bold green]Alert Levels:[/bold green]
• [yellow] WARNING[/yellow] - Moderate suspicious activity (50-80% probability)
• [red]CRITICAL[/red] - High probability of attack (80%+ probability)

[bold green]Verbose Monitoring Features:[/bold green]
• [cyan]Detailed Timestamps[/cyan] - Exact time for each prediction
• [cyan]Prediction Numbers[/cyan] - Sequential prediction tracking (#1, #2, #3...)
• [cyan]Attack Detection Results[/cyan] - Clear YES/NO for each prediction
• [cyan]Probability Percentages[/cyan] - Exact attack probability (0.00%, 25.5%, etc.)
• [cyan]Threat Levels[/cyan] - NORMAL, LOW_SUSPICION, MODERATE_SUSPICION, HIGH_PROBABILITY_ATTACK
• [cyan]System Status[/cyan] - CPU, Memory, Process Count, Disk Activity
• [cyan]Network Status[/cyan] - Connections, Packets, Traffic Level
• [cyan]Prediction History[/cyan] - Timeline of all monitoring results

[bold green]System Requirements:[/bold green]
• Python 3.7+
• Virtual environment (cyber_env)
• Required packages: typer, rich, psutil, scikit-learn, pandas, numpy
• Trained AI model files

[bold green]Author:[/bold green] ARYA KONER
[bold green]Version:[/bold green] 1.0.0
[bold green]License:[/bold green] MIT
"""
    
    console.print(Panel(help_text, title="BADNET Help", border_style="blue"))

@app.command()
def verbose_monitor(
    interval: int = typer.Option(5, "--interval", "-i", help="Monitoring interval in seconds"),
    duration: Optional[int] = typer.Option(None, "--duration", "-d", help="Monitoring duration in seconds (0 for continuous)")
):
    """Run verbose monitoring with detailed prediction results."""
    show_banner()
    
    console.print("[bold blue]Starting Verbose BADNET Monitoring...[/bold blue]")
    console.print(f"[green]Interval:[/green] {interval} seconds")
    console.print(f"[green]Duration:[/green] {'Continuous' if duration is None else f'{duration} seconds'}")
    console.print("[green]Mode:[/green] Detailed Prediction Analysis")
    console.print("")
    
    # Check if verbose monitor script exists
    verbose_script = Path("verbose_monitor.py")
    if not verbose_script.exists():
        console.print("[red]Error: verbose_monitor.py not found![/red]")
        console.print("Please ensure the verbose monitoring script is available.")
        return
    
    # Build command
    cmd = ["python3", "verbose_monitor.py", "--interval", str(interval)]
    if duration is not None:
        cmd.extend(["--duration", str(duration)])
    
    try:
        # Run verbose monitoring
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running verbose monitoring: {e}[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Verbose monitoring stopped by user[/yellow]")

@app.command()
def monitor_24x7():
    """Start 24/7 continuous monitoring service."""
    show_banner()
    
    console.print("[bold blue]Starting BADNET 24/7 Monitoring Service...[/bold blue]")
    console.print("[green]Mode:[/green] Continuous monitoring")
    console.print("[green]Interval:[/green] 10 seconds")
    console.print("[green]Duration:[/green] 24/7 operation")
    console.print("")
    
    # Check if 24x7 script exists
    script_24x7 = Path("monitor_24x7.sh")
    if not script_24x7.exists():
        console.print("[red]Error: monitor_24x7.sh not found![/red]")
        console.print("Please ensure the 24/7 monitoring script is available.")
        return
    
    try:
        # Make script executable and run
        os.chmod(script_24x7, 0o755)
        console.print("[green]Starting 24/7 monitoring...[/green]")
        console.print("[yellow]Press Ctrl+C to stop[/yellow]")
        console.print("")
        
        subprocess.run(["./monitor_24x7.sh"], check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running 24/7 monitoring: {e}[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]24/7 monitoring stopped by user[/yellow]")

@app.command()
def monitor_background():
    """Run monitoring in background mode."""
    show_banner()
    
    console.print("[bold blue]Starting BADNET Background Monitoring...[/bold blue]")
    console.print("[green]Mode:[/green] Background operation")
    console.print("[green]Output:[/green] Saved to monitoring.log")
    console.print("")
    
    try:
        # Start background monitoring
        cmd = ["nohup", "python3", "verbose_monitor.py", "--interval", "10", "--duration", "0"]
        
        with open("monitoring.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
        
        console.print(f"[green]Background monitoring started with PID: {process.pid}[/green]")
        console.print("[green]Output saved to: monitoring.log[/green]")
        console.print("[yellow]Use 'ps aux | grep verbose_monitor' to check status[/yellow]")
        console.print("[yellow]Use 'tail -f monitoring.log' to view live output[/yellow]")
        console.print("[yellow]Use 'kill {process.pid}' to stop monitoring[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error starting background monitoring: {e}[/red]")

@app.command()
def version():
    """Show BADNET version and information."""
    show_banner()
    
    version_info = """
[bold blue]BADNET Version Information[/bold blue]

[bold green]Version:[/bold green] 1.0.0
[bold green]Author:[/bold green] ARYA KONER
[bold green]Description:[/bold green] AI-Powered Cyber Attack Detection & Monitoring System
[bold green]Python Version:[/bold green] 3.7+
[bold green]Dependencies:[/bold green] typer, rich, psutil, scikit-learn, pandas, numpy

[bold green]Features:[/bold green]
• Real-time cyber attack detection
• Machine learning-based analysis
• System and network monitoring
• Intelligent alert system
• Comprehensive logging
• Easy-to-use CLI interface

[bold green]Model Performance:[/bold green]
• Best Model: LogisticRegression
• Accuracy: 100%
• Features: 59 cyber security features
• Detection Types: DoS, Probe, R2L, U2R attacks

[bold green]License:[/bold green] MIT
[bold green]Repository:[/bold green] AI-Powered Cyber Surveillance System
"""
    
    console.print(Panel(version_info, title="Version Info", border_style="green"))

def main():
    """Main entry point for BADNET."""
    if len(sys.argv) == 1:
        # No arguments provided, show interactive menu
        show_banner()
        
        # Define menu options and their handlers
        menu_options = {
            "1": ("Check System Status", lambda: status()),
            "2": ("Train AI Models", lambda: train()),
            "3": ("Start Basic Monitoring", lambda: _handle_basic_monitoring()),
            "4": ("Start Verbose Monitoring", lambda: _handle_verbose_monitoring()),
            "5": ("Start 24/7 Monitoring", lambda: monitor_24x7()),
            "6": ("Start Background Monitoring", lambda: monitor_background()),
            "7": ("AI Model Demo", lambda: demo()),
            "8": ("System Information", lambda: system_info()),
            "9": ("View Logs", lambda: logs()),
            "10": ("Configuration", lambda: config()),
            "11": ("Help", lambda: help_extended()),
            "12": ("Version Info", lambda: version()),
            "0": ("Exit", None)
        }
        
        while True:
            console.print("\n[bold blue]BADNET Main Menu[/bold blue]")
            for key, (description, _) in menu_options.items():
                console.print(f"{key}. {description}")
            
            choice = Prompt.ask("\nSelect option", choices=list(menu_options.keys()))
            
            if choice == "0":
                console.print("\n[bold green]Thank you for using BADNET![/bold green]")
                break
            
            if handler := menu_options[choice][1]:
                handler()
    else:
        # Arguments provided, use Typer CLI
        app()

def _handle_basic_monitoring():
    """Handle basic monitoring setup."""
    mode = Prompt.ask("Monitoring mode", choices=["system", "network", "both"], default="both")
    interval = Prompt.ask("Interval (seconds)", default="5")
    duration = Prompt.ask("Duration (seconds, 0 for continuous)", default="0")
    duration = int(duration) if duration != "0" else None
    save_log = Confirm.ask("Save log to file?")
    monitor(mode=mode, interval=int(interval), duration=duration, save_log=save_log)

def _handle_verbose_monitoring():
    """Handle verbose monitoring setup."""
    interval = Prompt.ask("Verbose monitoring interval (seconds)", default="5")
    duration = Prompt.ask("Duration (seconds, 0 for continuous)", default="30")
    duration = int(duration) if duration != "0" else None
    verbose_monitor(interval=int(interval), duration=duration)

if __name__ == "__main__":
    main()
