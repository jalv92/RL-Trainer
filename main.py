#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL TRAINER - Interactive Command-Line Menu System
Comprehensive interface for managing reinforcement learning trading system

Features:
- Requirements installation with progress tracking
- Data processing with instrument selection
- Training model management (test/production)
- Model evaluation with result logging
- Colored output and error handling
- Input validation and logging

Author: RL Trading System Team
Date: October 2025
"""

import os
import sys
import subprocess
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Import model utilities
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from model_utils import detect_models_in_folder, display_model_selection

# Try to import colorama for colored output
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    print("Warning: colorama not installed. Install with: pip install colorama")
    COLORAMA_AVAILABLE = False

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    TQDM_AVAILABLE = False


class Colors:
    """Color constants for terminal output."""
    
    if COLORAMA_AVAILABLE:
        RED = Fore.RED
        GREEN = Fore.GREEN
        BLUE = Fore.BLUE
        YELLOW = Fore.YELLOW
        MAGENTA = Fore.MAGENTA
        CYAN = Fore.CYAN
        WHITE = Fore.WHITE
        BOLD = Style.BRIGHT
        RESET = Style.RESET_ALL
    else:
        RED = "\033[31m"
        GREEN = "\033[32m"
        BLUE = "\033[34m"
        YELLOW = "\033[33m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
        BOLD = "\033[1m"
        RESET = "\033[0m"


class RLTrainerMenu:
    """Main menu system for RL Trading System."""
    
    def __init__(self):
        """Initialize the menu system."""
        self.setup_logging()
        self.project_dir = Path(__file__).resolve().parent
        self.src_dir = self.project_dir / "src"
        self.logs_dir = self.project_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Valid instruments for data processing
        self.valid_instruments = ["NQ", "ES", "YM", "RTY", "MNQ", "MES", "M2K", "MYM"]
        
        # Menu options
        self.main_menu_options = {
            "1": "Requirements Installation",
            "2": "Data Processing",
            "3": "Training Model",
            "4": "Evaluator",
            "5": "Exit"
        }
        
        self.training_menu_options = {
            "1": "Training Test (Local Testing)",
            "2": "Training Pod (Production)",
            "3": "Continue from Existing Model",
            "4": "Back to Main Menu"
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create main log file
        log_file = log_dir / f"rl_trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("RL Trainer Menu System Initialized")
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_banner(self):
        """Display the RL TRAINER banner with Unicode block characters similar to CLAUDE CODE style."""
        # RL TRAINER ASCII art using full Unicode block characters - large blocky style
        rl_trainer_art = [
            "███████  ██      ",
            "██   ██  ██      ",
            "███████  ██      ",
            "██████   ██      ",
            "██  ███  ██      ",
            "██   ██  ███████ ",
            "",
            "████████ ███████   █████  ██ ███    ██ ███████ ███████ ",
            "   ██    ██   ██  ██   ██ ██ ████   ██ ██      ██   ██ ",
            "   ██    ███████  ███████ ██ ██ ██  ██ █████   ███████ ",
            "   ██    ██████   ██   ██ ██ ██  ██ ██ ██      ██████  ",
            "   ██    ██  ███  ██   ██ ██ ██   ████ ██      ██  ███ ",
            "   ██    ██   ██  ██   ██ ██ ██    ███ ███████ ██   ██ "
        ]
        
        # Alternative ultra-blocky style (more similar to screenshot)
        blocky_style = [
            "██████╗ ██╗     ",
            "██╔══██╗██║     ",
            "██████╔╝██║     ",
            "██╔══██╗██║     ",
            "██║  ██║███████╗",
            "╚═╝  ╚═╝╚══════╝",
            "",
            "████████╗██████╗  █████╗ ██╗███╗   ██╗███████╗██████╗ ",
            "╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██╔════╝██╔══██╗",
            "   ██║   ██████╔╝███████║██║██╔██╗ ██║█████╗  ██████╔╝",
            "   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██╔══╝  ██╔══██╗",
            "   ██║   ██║  ██║██║  ██║██║██║ ╚████║███████╗██║  ██║",
            "   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝"
        ]

        # Pure block characters style (most similar to CLAUDE CODE screenshot)
        pure_blocks = """
█████████  ██         
██     ██  ██         
█████████  ██         
██     ██  ██         
██     ██  ██         
██     ██  ███████████

████████ ████████   ████████  ████  ██     ██ ████████ ████████  
   ██    ██     ██  ██    ██   ██   ████   ██ ██       ██     ██ 
   ██    ████████   ████████   ██   ██ ██  ██ ████████ ████████  
   ██    ██   ██    ██    ██   ██   ██  ██ ██ ██       ██   ██   
   ██    ██     ██  ██    ██   ██   ██   ████ ████████ ██     ██ """

        # Using the blocky_style which looks cleaner
        banner_color = f"{Colors.CYAN}{Colors.BOLD}"

        header_text = " Welcome to the RL TRAINER research preview! "
        header_border = "═" * (len(header_text) + 2)

        banner_sections = [
            f"{Colors.YELLOW}╔{header_border}╗{Colors.RESET}",
            f"{Colors.YELLOW}║ {Colors.RESET}{Colors.WHITE}{header_text}{Colors.RESET}{Colors.YELLOW} ║{Colors.RESET}",
            f"{Colors.YELLOW}╚{header_border}╝{Colors.RESET}",
            "",
            f"{banner_color}" + "\n".join(blocky_style) + f"{Colors.RESET}",
            "",
            f"{Colors.CYAN}{Colors.BOLD}{'═' * 78}{Colors.RESET}",
            f"{Colors.CYAN}{Colors.BOLD}  Reinforcement Learning Trading System - Interactive Menu Interface{Colors.RESET}",
            f"{Colors.CYAN}{Colors.BOLD}  Version 1.0.0 - October 2025{Colors.RESET}",
            f"{Colors.CYAN}{Colors.BOLD}{'═' * 78}{Colors.RESET}",
            "",
            f"{Colors.GREEN}{Colors.BOLD}Welcome to RL TRAINER! Your comprehensive trading system management tool.{Colors.RESET}",
        ]

        print("\n".join(banner_sections))

    def detect_and_select_market(self) -> Optional[str]:
        """
        Detect available market data files and prompt user to select one.

        Returns:
            Selected market symbol (e.g., 'ES', 'NQ') or None if cancelled
        """
        try:
            # Import market detection utilities from model_utils
            sys.path.insert(0, str(self.src_dir))
            from model_utils import detect_available_markets
            from market_specs import get_market_spec

            # Detect available markets
            data_dir = self.project_dir / 'data'
            available_markets = detect_available_markets(str(data_dir))

            if not available_markets:
                print(f"\n{Colors.RED}No market data files found in data/ directory.{Colors.RESET}")
                print(f"{Colors.YELLOW}Please run 'Data Processing' first to prepare market data.{Colors.RESET}")
                return None

            # Display header
            print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}║                   MARKET SELECTION                            ║{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
            print()

            # If only one market, auto-select it
            if len(available_markets) == 1:
                market = available_markets[0]['market']
                market_spec = get_market_spec(market)
                spec_info = f"${market_spec.contract_multiplier} x {market_spec.tick_size} tick = ${market_spec.tick_value:.2f}"

                print(f"{Colors.GREEN}Detected 1 market:{Colors.RESET}")
                print(f"  • {Colors.BOLD}{market}{Colors.RESET} - {available_markets[0]['minute_file']}")
                print(f"    {Colors.CYAN}{spec_info}{Colors.RESET}")
                print(f"\n{Colors.GREEN}Auto-selecting {market} for training.{Colors.RESET}")
                return market

            # Multiple markets - show selection menu
            print(f"{Colors.GREEN}Detected {len(available_markets)} markets:{Colors.RESET}\n")

            for i, market_info in enumerate(available_markets, 1):
                market = market_info['market']
                market_spec = get_market_spec(market)
                spec_info = f"${market_spec.contract_multiplier} x {market_spec.tick_size} tick = ${market_spec.tick_value:.2f}"

                print(f"{Colors.BOLD}  {i}. {market:<8}{Colors.RESET} - {market_info['minute_file']:<25}")
                print(f"     {Colors.CYAN}{spec_info} | Commission: ${market_spec.commission}/side{Colors.RESET}")
                print()

            print(f"{Colors.YELLOW}  0. Cancel{Colors.RESET}")
            print()

            # Get user selection
            valid_choices = [str(i) for i in range(len(available_markets) + 1)]
            choice = self.get_user_input(
                f"{Colors.BOLD}Select market to train on (0-{len(available_markets)}): {Colors.RESET}",
                valid_choices
            )

            if choice == "0" or choice is None:
                print(f"\n{Colors.YELLOW}Market selection cancelled.{Colors.RESET}")
                return None

            # Return selected market symbol
            selected_market = available_markets[int(choice) - 1]['market']
            print(f"\n{Colors.GREEN}Selected market: {Colors.BOLD}{selected_market}{Colors.RESET}")
            return selected_market

        except Exception as e:
            self.logger.error(f"Error detecting markets: {str(e)}", exc_info=True)
            print(f"\n{Colors.RED}Error detecting markets: {str(e)}{Colors.RESET}")
            return None

    def display_main_menu(self):
        """Display the main menu options."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                        MAIN MENU                              ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
        print()
        
        for key, value in self.main_menu_options.items():
            if key == "5":  # Exit option
                print(f"{Colors.RED}  {key}. {value}{Colors.RESET}")
            else:
                print(f"{Colors.GREEN}  {key}. {value}{Colors.RESET}")
        
        print()
    
    def display_training_menu(self):
        """Display the training submenu options."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                      TRAINING MENU                            ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
        print()

        for key, value in self.training_menu_options.items():
            if key == "4":  # Back option
                print(f"{Colors.YELLOW}  {key}. {value}{Colors.RESET}")
            else:
                print(f"{Colors.GREEN}  {key}. {value}{Colors.RESET}")

        print()
    
    def get_user_input(self, prompt: str, valid_options: Optional[List[str]] = None) -> Optional[str]:
        """
        Get and validate user input.
        
        Args:
            prompt: The prompt to display to the user
            valid_options: List of valid input options
            
        Returns:
            Validated user input
        """
        while True:
            try:
                user_input = input(f"{Colors.BLUE}{prompt}{Colors.RESET}").strip()
                
                if valid_options and user_input not in valid_options:
                    print(f"{Colors.RED}Invalid option. Please choose from: {', '.join(valid_options)}{Colors.RESET}")
                    continue

                return user_input

            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Operation cancelled by user.{Colors.RESET}")
                return None
            except EOFError:
                print(f"\n{Colors.YELLOW}Input stream ended.{Colors.RESET}")
                return None
    
    def validate_instrument(self, instrument: str) -> bool:
        """
        Validate instrument selection.
        
        Args:
            instrument: The instrument to validate
            
        Returns:
            True if valid, False otherwise
        """
        return instrument.upper() in self.valid_instruments
    
    def display_instruments(self):
        """Display available instruments for selection."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Available Instruments:{Colors.RESET}")
        for i, instrument in enumerate(self.valid_instruments, 1):
            print(f"{Colors.GREEN}  {i}. {instrument}{Colors.RESET}")
        print()
    
    def run_command_with_progress(self, command: List[str], description: str, 
                              log_file: str = None) -> Tuple[bool, str]:
        """
        Run a command with progress indication and logging.
        
        Args:
            command: Command to execute as list of strings
            description: Description of the operation
            log_file: Log file to save output
            
        Returns:
            Tuple of (success, output)
        """
        print(f"\n{Colors.YELLOW}{description}{Colors.RESET}")
        print(f"{Colors.CYAN}Executing: {' '.join(command)}{Colors.RESET}")
        print(f"{Colors.MAGENTA}{'=' * 60}{Colors.RESET}")

        try:
            # Always run from project root directory
            working_directory = self.project_dir

            # Set up environment with project root in PYTHONPATH for imports
            env = os.environ.copy()
            pythonpath = str(self.project_dir)
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = pythonpath

            # Start the process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,  # Prevent interactive prompts from hanging
                encoding='utf-8',  # Explicitly use UTF-8 to handle Unicode characters (✅, ⚠️, etc.)
                errors='replace',  # Replace invalid characters instead of crashing
                cwd=working_directory,
                env=env
            )

            output_lines = []

            # Read output line by line
            if TQDM_AVAILABLE:
                # Use tqdm for progress indication
                with tqdm(desc="Streaming Output", total=0, bar_format="{desc}", leave=False) as progress_bar:
                    for line in iter(process.stdout.readline, ''):
                        if not line:
                            break
                        cleaned = line.rstrip()
                        if cleaned:
                            output_lines.append(cleaned)
                            progress_bar.write(f"{Colors.WHITE}{cleaned}{Colors.RESET}")
                process.wait()
            else:
                # Simple output without progress bar
                output, _ = process.communicate()
                output_lines = output.split('\n') if output else []
                for line in output_lines:
                    cleaned = line.strip()
                    if cleaned:
                        print(f"{Colors.WHITE}{cleaned}{Colors.RESET}")
            
            # Check return code
            success = process.returncode == 0
            
            if success:
                print(f"\n{Colors.GREEN}✓ {description} completed successfully!{Colors.RESET}")
            else:
                print(f"\n{Colors.RED}✗ {description} failed with return code {process.returncode}{Colors.RESET}")
            
            # Save to log file if specified
            if log_file:
                log_path = self.logs_dir / log_file
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'=' * 60}\n")
                    f.write(f"{datetime.now().isoformat()} - {description}\n")
                    f.write(f"Command: {' '.join(command)}\n")
                    f.write(f"Return Code: {process.returncode}\n")
                    f.write(f"Output:\n{'\n'.join(output_lines)}\n")
            
            return success, '\n'.join(output_lines)
            
        except subprocess.TimeoutExpired:
            print(f"\n{Colors.RED}✗ {description} timed out{Colors.RESET}")
            return False, "Operation timed out"
        except Exception as e:
            print(f"\n{Colors.RED}✗ {description} failed: {str(e)}{Colors.RESET}")
            return False, str(e)
    
    def check_package_installed(self, package_name: str) -> bool:
        """
        Check if a package is installed.

        Args:
            package_name: Name of the package to check

        Returns:
            True if installed, False otherwise
        """
        try:
            # Handle package names with version specifiers
            pkg = package_name.split('>=')[0].split('==')[0].split('<')[0].strip()
            # Handle special cases
            if pkg == 'sb3-contrib':
                pkg = 'sb3_contrib'
            __import__(pkg)
            return True
        except ImportError:
            return False

    def check_installed_requirements(self) -> Tuple[List[str], List[str]]:
        """
        Check which requirements are already installed.

        Returns:
            Tuple of (installed_packages, missing_packages)
        """
        requirements_file = self.project_dir / "requirements.txt"
        installed = []
        missing = []

        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Extract package name
                package = line.split('>=')[0].split('==')[0].split('<')[0].strip()

                if self.check_package_installed(package):
                    installed.append(package)
                else:
                    missing.append(package)

        return installed, missing

    def install_requirements(self):
        """Install requirements using pip with smart checking."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                  REQUIREMENTS INSTALLATION                    ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")

        # Check if requirements.txt exists
        requirements_file = self.project_dir / "requirements.txt"
        if not requirements_file.exists():
            print(f"{Colors.RED}Error: requirements.txt not found in project directory{Colors.RESET}")
            return False

        print(f"{Colors.GREEN}Found requirements.txt at: {requirements_file}{Colors.RESET}")

        # Check what's already installed
        print(f"\n{Colors.CYAN}Checking installed packages...{Colors.RESET}")
        installed, missing = self.check_installed_requirements()

        total = len(installed) + len(missing)

        # Display status
        print(f"\n{Colors.BOLD}Installation Status:{Colors.RESET}")
        print(f"{Colors.GREEN}  ✓ Installed: {len(installed)}/{total} packages{Colors.RESET}")
        if missing:
            print(f"{Colors.YELLOW}  ✗ Missing: {len(missing)}/{total} packages{Colors.RESET}")

        # If all installed
        if not missing:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All requirements are already installed!{Colors.RESET}")
            print(f"\n{Colors.CYAN}Installed packages:{Colors.RESET}")
            for pkg in installed[:8]:  # Show first 8
                print(f"  • {pkg}")
            if len(installed) > 8:
                print(f"  ... and {len(installed) - 8} more")

            print(f"\n{Colors.YELLOW}Options:{Colors.RESET}")
            print(f"  {Colors.GREEN}1. Return to Main Menu{Colors.RESET}")
            print(f"  {Colors.CYAN}2. Reinstall All Packages (if having issues){Colors.RESET}")
            print(f"  {Colors.CYAN}3. Upgrade All Packages{Colors.RESET}")

            choice = self.get_user_input(
                f"\n{Colors.YELLOW}Select option (1-3): {Colors.RESET}",
                ["1", "2", "3"]
            )

            if choice is None or choice == "1":
                print(f"{Colors.GREEN}Returning to main menu...{Colors.RESET}")
                return True
            elif choice == "2":
                command = [sys.executable, "-m", "pip", "install", "--force-reinstall", "-r", "requirements.txt"]
                description = "Reinstalling All Requirements"
            else:  # choice == "3"
                command = [sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt"]
                description = "Upgrading All Requirements"

        # If some missing
        else:
            print(f"\n{Colors.YELLOW}Missing packages:{Colors.RESET}")
            for pkg in missing:
                print(f"  • {pkg}")

            if installed:
                print(f"\n{Colors.GREEN}Already installed:{Colors.RESET}")
                for pkg in installed[:5]:  # Show first 5
                    print(f"  • {pkg}")
                if len(installed) > 5:
                    print(f"  ... and {len(installed) - 5} more")

            print(f"\n{Colors.YELLOW}Options:{Colors.RESET}")
            print(f"  {Colors.GREEN}1. Install Missing Packages Only{Colors.RESET}")
            print(f"  {Colors.CYAN}2. Reinstall All Packages{Colors.RESET}")
            print(f"  {Colors.RED}3. Cancel / Return to Main Menu{Colors.RESET}")

            choice = self.get_user_input(
                f"\n{Colors.YELLOW}Select option (1-3): {Colors.RESET}",
                ["1", "2", "3"]
            )

            if choice is None or choice == "3":
                print(f"{Colors.YELLOW}Installation cancelled. Returning to main menu...{Colors.RESET}")
                return False
            elif choice == "1":
                command = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
                description = "Installing Missing Requirements"
            else:  # choice == "2"
                command = [sys.executable, "-m", "pip", "install", "--force-reinstall", "-r", "requirements.txt"]
                description = "Reinstalling All Requirements"

        # Run installation
        print()  # Blank line for readability
        success, output = self.run_command_with_progress(
            command,
            description,
            "installation.log"
        )

        if success:
            print(f"\n{Colors.GREEN}✓ Installation completed successfully!{Colors.RESET}")
            print(f"{Colors.CYAN}All dependencies are now installed.{Colors.RESET}")
            print(f"{Colors.CYAN}You can now use all features of the RL Trading System.{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}✗ Installation failed. Check logs for details.{Colors.RESET}")
            print(f"{Colors.YELLOW}Tip: Try running 'pip install -r requirements.txt' manually{Colors.RESET}")

        return success
    
    def process_data(self):
        """Handle data processing with instrument selection."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                     DATA PROCESSING                            ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
        
        # Display available instruments
        self.display_instruments()
        
        # Get instrument selection
        while True:
            instrument_input = self.get_user_input(
                f"{Colors.YELLOW}Enter instrument number or name (e.g., '1' or 'ES'): {Colors.RESET}"
            )

            if instrument_input is None:
                print(f"{Colors.YELLOW}Data processing cancelled.{Colors.RESET}")
                return False

            if instrument_input.lower() in ["cancel", "exit"]:
                print(f"{Colors.YELLOW}Data processing cancelled.{Colors.RESET}")
                return False
            
            # Check if input is a number
            if instrument_input.isdigit():
                instrument_idx = int(instrument_input) - 1
                if 0 <= instrument_idx < len(self.valid_instruments):
                    instrument = self.valid_instruments[instrument_idx]
                    break
                else:
                    print(f"{Colors.RED}Invalid number. Please choose 1-{len(self.valid_instruments)}.{Colors.RESET}")
            else:
                # Check if input is a valid instrument name
                if self.validate_instrument(instrument_input):
                    instrument = instrument_input.upper()
                    break
                else:
                    print(f"{Colors.RED}Invalid instrument. Please choose from the list.{Colors.RESET}")
        
        print(f"{Colors.GREEN}Selected instrument: {instrument}{Colors.RESET}")
        
        # Confirm selection
        confirm = self.get_user_input(
            f"{Colors.YELLOW}Process data for {instrument}? (y/n): {Colors.RESET}",
            ["y", "n", "Y", "N"]
        )

        if confirm is None or confirm.lower() != 'y':
            print(f"{Colors.YELLOW}Data processing cancelled.{Colors.RESET}")
            return False
        
        # Run data processing
        update_script = self.src_dir / "update_training_data.py"
        if not update_script.exists():
            print(f"{Colors.RED}Error: update_training_data.py not found{Colors.RESET}")
            return False
        
        command = [sys.executable, str(update_script), "--market", instrument]
        success, output = self.run_command_with_progress(
            command,
            f"Processing {instrument} Data",
            f"data_processing_{instrument.lower()}.log"
        )
        
        if success:
            print(f"\n{Colors.GREEN}✓ Data processing completed for {instrument}!{Colors.RESET}")
            print(f"{Colors.CYAN}Data files created: {instrument}_D1M.csv, {instrument}_D1S.csv{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}✗ Data processing failed for {instrument}. Check logs for details.{Colors.RESET}")
        
        return success
    
    def train_model(self):
        """Handle model training with test/production options."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                      TRAINING MODEL                             ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")

        while True:
            self.display_training_menu()
            choice = self.get_user_input(
                f"{Colors.YELLOW}Select training option: {Colors.RESET}",
                list(self.training_menu_options.keys())
            )

            if choice is None:
                print(f"{Colors.YELLOW}Training menu cancelled. Returning to main menu...{Colors.RESET}")
                break

            if choice == "1":
                self.run_training_test()
                break
            elif choice == "2":
                self.run_training_production()
                break
            elif choice == "3":
                self.continue_from_model()
                break
            elif choice == "4":
                print(f"{Colors.YELLOW}Returning to main menu...{Colors.RESET}")
                break
            else:
                print(f"{Colors.RED}Invalid option. Please try again.{Colors.RESET}")
    
    def run_training_test(self):
        """Run training in test mode."""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}TRAINING TEST MODE{Colors.RESET}")
        print(f"{Colors.CYAN}This will run a quick test with reduced dataset for local testing.{Colors.RESET}")

        # Confirm
        confirm = self.get_user_input(
            f"{Colors.YELLOW}Proceed with test training? (y/n): {Colors.RESET}",
            ["y", "n", "Y", "N"]
        )

        if confirm is None or confirm.lower() != 'y':
            print(f"{Colors.YELLOW}Test training cancelled.{Colors.RESET}")
            return False

        # Market selection
        selected_market = self.detect_and_select_market()
        if selected_market is None:
            print(f"{Colors.YELLOW}Test training cancelled - no market selected.{Colors.RESET}")
            return False

        # Phase 1 Test Training
        phase1_script = self.src_dir / "train_phase1.py"
        if phase1_script.exists():
            print(f"\n{Colors.GREEN}Starting Phase 1 Test Training...{Colors.RESET}")
            command = [sys.executable, str(phase1_script), "--test", "--market", selected_market, "--non-interactive"]
            success, output = self.run_command_with_progress(
                command,
                "Phase 1 Test Training",
                "training_phase1_test.log"
            )

            if not success:
                print(f"{Colors.RED}Phase 1 test training failed.{Colors.RESET}")
                return False

        # Phase 2 Test Training
        phase2_script = self.src_dir / "train_phase2.py"
        if phase2_script.exists():
            print(f"\n{Colors.GREEN}Starting Phase 2 Test Training...{Colors.RESET}")
            command = [sys.executable, str(phase2_script), "--test", "--market", selected_market, "--non-interactive"]
            success, output = self.run_command_with_progress(
                command,
                "Phase 2 Test Training",
                "training_phase2_test.log"
            )
            
            if not success:
                print(f"{Colors.RED}Phase 2 test training failed.{Colors.RESET}")
                return False
        
        print(f"\n{Colors.GREEN}✓ Test training completed successfully!{Colors.RESET}")
        print(f"{Colors.CYAN}Models saved in models/{Colors.RESET}")
        return True
    
    def run_training_production(self):
        """Run training in production mode."""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}TRAINING PRODUCTION MODE{Colors.RESET}")
        print(f"{Colors.CYAN}This will run full production training on the complete dataset.{Colors.RESET}")
        print(f"{Colors.RED}Warning: This may take several hours to complete.{Colors.RESET}")

        # Confirm
        confirm = self.get_user_input(
            f"{Colors.YELLOW}Proceed with production training? (y/n): {Colors.RESET}",
            ["y", "n", "Y", "N"]
        )

        if confirm is None or confirm.lower() != 'y':
            print(f"{Colors.YELLOW}Production training cancelled.{Colors.RESET}")
            return False

        # Market selection
        selected_market = self.detect_and_select_market()
        if selected_market is None:
            print(f"{Colors.YELLOW}Production training cancelled - no market selected.{Colors.RESET}")
            return False

        # Phase 1 Production Training
        phase1_script = self.src_dir / "train_phase1.py"
        if phase1_script.exists():
            print(f"\n{Colors.GREEN}Starting Phase 1 Production Training...{Colors.RESET}")
            command = [sys.executable, str(phase1_script), "--market", selected_market, "--non-interactive"]
            success, output = self.run_command_with_progress(
                command,
                "Phase 1 Production Training",
                "training_phase1_production.log"
            )

            if not success:
                print(f"{Colors.RED}Phase 1 production training failed.{Colors.RESET}")
                return False

        # Phase 2 Production Training
        phase2_script = self.src_dir / "train_phase2.py"
        if phase2_script.exists():
            print(f"\n{Colors.GREEN}Starting Phase 2 Production Training...{Colors.RESET}")
            command = [sys.executable, str(phase2_script), "--market", selected_market, "--non-interactive"]
            success, output = self.run_command_with_progress(
                command,
                "Phase 2 Production Training",
                "training_phase2_production.log"
            )
            
            if not success:
                print(f"{Colors.RED}Phase 2 production training failed.{Colors.RESET}")
                return False
        
        print(f"\n{Colors.GREEN}✓ Production training completed successfully!{Colors.RESET}")
        print(f"{Colors.CYAN}Models saved in models/{Colors.RESET}")
        return True

    def continue_from_model(self):
        """Continue training from an existing model in the models folder."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                CONTINUE TRAINING FROM MODEL                   ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")

        # Detect available Phase 1 models
        print(f"\n{Colors.CYAN}Scanning for available Phase 1 models...{Colors.RESET}")
        models = detect_models_in_folder(phase='phase1')

        # Display and get selection
        selected_idx = display_model_selection(models, phase_filter='phase1')

        if selected_idx == -1:
            print(f"{Colors.YELLOW}No model selected. Returning to training menu...{Colors.RESET}")
            return False

        selected_model = models[selected_idx]

        print(f"\n{Colors.GREEN}Selected model: {selected_model['name']}{Colors.RESET}")
        print(f"{Colors.CYAN}Path: {selected_model['path']}{Colors.RESET}")

        if not selected_model['vecnorm_path']:
            print(f"\n{Colors.RED}⚠ Warning: No VecNormalize file found for this model!{Colors.RESET}")
            print(f"{Colors.YELLOW}Training may not work correctly without normalization statistics.{Colors.RESET}")
            confirm = self.get_user_input(
                f"{Colors.YELLOW}Continue anyway? (y/n): {Colors.RESET}",
                ["y", "n", "Y", "N"]
            )
            if confirm is None or confirm.lower() != 'y':
                print(f"{Colors.YELLOW}Operation cancelled.{Colors.RESET}")
                return False

        # Ask for test or production mode
        print(f"\n{Colors.CYAN}Select training mode:{Colors.RESET}")
        print(f"{Colors.GREEN}  1. Test Mode (Local Testing - reduced timesteps){Colors.RESET}")
        print(f"{Colors.GREEN}  2. Production Mode (Full Training){Colors.RESET}")

        mode_choice = self.get_user_input(
            f"{Colors.YELLOW}Select mode (1 or 2): {Colors.RESET}",
            ["1", "2"]
        )

        if mode_choice is None:
            print(f"{Colors.YELLOW}Operation cancelled.{Colors.RESET}")
            return False

        test_mode = (mode_choice == "1")
        mode_name = "Test" if test_mode else "Production"

        # Confirm
        print(f"\n{Colors.BOLD}{Colors.YELLOW}CONFIRMATION{Colors.RESET}")
        print(f"{Colors.CYAN}Model: {selected_model['name']}{Colors.RESET}")
        print(f"{Colors.CYAN}Mode: {mode_name}{Colors.RESET}")

        confirm = self.get_user_input(
            f"{Colors.YELLOW}Proceed with continuing training? (y/n): {Colors.RESET}",
            ["y", "n", "Y", "N"]
        )

        if confirm is None or confirm.lower() != 'y':
            print(f"{Colors.YELLOW}Training cancelled.{Colors.RESET}")
            return False

        # Market selection
        selected_market = self.detect_and_select_market()
        if selected_market is None:
            print(f"{Colors.YELLOW}Training cancelled - no market selected.{Colors.RESET}")
            return False

        # Run Phase 1 training with continuation
        phase1_script = self.src_dir / "train_phase1.py"
        if not phase1_script.exists():
            print(f"{Colors.RED}Error: Phase 1 training script not found!{Colors.RESET}")
            return False

        print(f"\n{Colors.GREEN}Starting Phase 1 Continuation Training ({mode_name} Mode)...{Colors.RESET}")

        # Build command with --continue flag
        command = [sys.executable, str(phase1_script), "--continue", "--model-path", selected_model['path'], "--market", selected_market, "--non-interactive"]
        if test_mode:
            command.append("--test")

        success, output = self.run_command_with_progress(
            command,
            f"Phase 1 Continuation Training ({mode_name})",
            f"training_phase1_continue_{mode_name.lower()}.log"
        )

        if not success:
            print(f"{Colors.RED}Phase 1 continuation training failed.{Colors.RESET}")
            return False

        print(f"\n{Colors.GREEN}✓ Continuation training completed successfully!{Colors.RESET}")
        print(f"{Colors.CYAN}Updated model saved in models/{Colors.RESET}")
        return True

    def run_evaluation(self):
        """Run model evaluation."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║                         EVALUATOR                              ║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}")
        
        # Check if evaluation script exists
        eval_script = self.src_dir / "evaluate_phase2.py"
        if not eval_script.exists():
            print(f"{Colors.RED}Error: evaluate_phase2.py not found{Colors.RESET}")
            return False
        
        print(f"{Colors.GREEN}Found evaluation script: {eval_script}{Colors.RESET}")
        
        # Confirm
        confirm = self.get_user_input(
            f"{Colors.YELLOW}Proceed with model evaluation? (y/n): {Colors.RESET}",
            ["y", "n", "Y", "N"]
        )

        if confirm is None or confirm.lower() != 'y':
            print(f"{Colors.YELLOW}Evaluation cancelled.{Colors.RESET}")
            return False

        # Market selection
        selected_market = self.detect_and_select_market()
        if selected_market is None:
            print(f"{Colors.YELLOW}Evaluation cancelled - no market selected.{Colors.RESET}")
            return False

        # Run evaluation
        command = [sys.executable, str(eval_script), "--market", selected_market, "--non-interactive"]
        success, output = self.run_command_with_progress(
            command,
            "Model Evaluation",
            "evaluation.log"
        )
        
        if success:
            print(f"\n{Colors.GREEN}✓ Evaluation completed successfully!{Colors.RESET}")
            print(f"{Colors.CYAN}Results saved in results/{Colors.RESET}")

            # Check for evaluation results
            results_dir = self.project_dir / "results"
            if results_dir.exists():
                print(f"{Colors.YELLOW}Evaluation outputs:{Colors.RESET}")
                for file in results_dir.glob("*"):
                    if file.is_file():
                        print(f"  - {file.name}")
        else:
            print(f"\n{Colors.RED}✗ Evaluation failed. Check logs for details.{Colors.RESET}")
        
        return success
    
    def show_instructions(self):
        """Show first-time user instructions."""
        instructions = f"""
{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}
{Colors.BOLD}{Colors.CYAN}║                    GETTING STARTED                              ║{Colors.RESET}
{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}

{Colors.GREEN}Welcome to RL TRAINER! This system helps you manage your{Colors.RESET}
{Colors.GREEN}reinforcement learning trading pipeline from setup to evaluation.{Colors.RESET}

{Colors.BOLD}{Colors.YELLOW}MAIN FEATURES:{Colors.RESET}
{Colors.WHITE}1. Requirements Installation{Colors.RESET}
  {Colors.CYAN}   • Installs all necessary dependencies{Colors.RESET}
  {Colors.CYAN}   • Sets up the trading environment{Colors.RESET}

{Colors.WHITE}2. Data Processing{Colors.RESET}
  {Colors.CYAN}   • Select from 8 trading instruments{Colors.RESET}
  {Colors.CYAN}   • Process market data for training{Colors.RESET}

{Colors.WHITE}3. Training Model{Colors.RESET}
  {Colors.CYAN}   • Test Mode: Quick local testing{Colors.RESET}
  {Colors.CYAN}   • Production Mode: Full training pipeline{Colors.RESET}

{Colors.WHITE}4. Evaluator{Colors.RESET}
  {Colors.CYAN}   • Comprehensive model evaluation{Colors.RESET}
  {Colors.CYAN}   • Performance metrics and visualizations{Colors.RESET}

{Colors.BOLD}{Colors.YELLOW}SUPPORTED INSTRUMENTS:{Colors.RESET}
{Colors.WHITE}NQ  - Nasdaq 100 E-mini     ES  - S&P 500 E-mini{Colors.RESET}
{Colors.WHITE}YM  - Dow Jones E-mini      RTY - Russell 2000 E-mini{Colors.RESET}
{Colors.WHITE}MNQ - Micro Nasdaq 100      MES - Micro S&P 500{Colors.RESET}
{Colors.WHITE}M2K - Micro Russell 2000    MYM - Micro Dow Jones{Colors.RESET}

{Colors.BOLD}{Colors.YELLOW}TIPS:{Colors.RESET}
{Colors.WHITE}• All operations are logged for debugging{Colors.RESET}
{Colors.WHITE}• Press Ctrl+C to cancel any operation{Colors.RESET}
{Colors.WHITE}• Check the logs directory for detailed output{Colors.RESET}
{Colors.WHITE}• Ensure sufficient disk space for data processing{Colors.RESET}

{Colors.GREEN}Press Enter to continue to the main menu...{Colors.RESET}
"""
        print(instructions)
        input()
    
    def run(self):
        """Main menu loop."""
        # Show instructions on first run
        if not (self.logs_dir / ".instructions_shown").exists():
            self.show_instructions()
            (self.logs_dir / ".instructions_shown").touch()
        
        while True:
            try:
                self.clear_screen()
                self.display_banner()
                self.display_main_menu()
                
                choice = self.get_user_input(
                    f"{Colors.YELLOW}Select an option: {Colors.RESET}",
                    list(self.main_menu_options.keys())
                )

                if choice is None:
                    print(f"\n{Colors.YELLOW}Returning to shell...{Colors.RESET}")
                    break

                if choice == "1":
                    self.install_requirements()
                elif choice == "2":
                    self.process_data()
                elif choice == "3":
                    self.train_model()
                elif choice == "4":
                    self.run_evaluation()
                elif choice == "5":
                    print(f"\n{Colors.GREEN}Thank you for using RL TRAINER!{Colors.RESET}")
                    print(f"{Colors.CYAN}Goodbye!{Colors.RESET}")
                    break
                else:
                    print(f"{Colors.RED}Invalid option. Please try again.{Colors.RESET}")
                    time.sleep(2)
                
                if choice != "5":
                    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.RESET}")
                    
            except KeyboardInterrupt:
                print(f"\n\n{Colors.YELLOW}Exiting RL TRAINER...{Colors.RESET}")
                break
            except Exception as e:
                print(f"\n{Colors.RED}An error occurred: {str(e)}{Colors.RESET}")
                self.logger.error(f"Menu error: {str(e)}")
                input(f"{Colors.YELLOW}Press Enter to continue...{Colors.RESET}")


def main():
    """Main entry point for the RL Trainer menu system."""
    try:
        # Create and run the menu system
        menu = RLTrainerMenu()
        menu.run()
    except Exception as e:
        print(f"{Colors.RED}Fatal error: {str(e)}{Colors.RESET}")
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()