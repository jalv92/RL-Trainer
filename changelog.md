# Changelog

All notable changes to this project are documented in this file. Entries are grouped by date and categorized as Added, Changed, Fixed, Removed, or Deprecated.

## [1.0.0] - 2025-10-28

### Added - Continue Training Feature 🎯
- **New model management system** (`src/model_utils.py`) with comprehensive model detection and loading utilities:
  - `detect_models_in_folder()` - Scans models directory and returns metadata (name, type, size, modification date, VecNormalize path)
  - `load_model_auto()` - Auto-detects model type (Phase 1 PPO or Phase 2 MaskablePPO) and loads appropriately
  - `display_model_selection()` - Interactive model selection interface with formatted display
  - `get_model_save_name()` - Custom save name prompt after training completion
  - `load_vecnormalize()` - VecNormalize statistics loader with validation
  - `validate_model_environment_compatibility()` - Model/environment type validation
- **"Continue from Existing Model" menu option** in main training menu (Option 3 → Option 3)
- **Command-line continuation support** in `src/train_phase1.py`:
  - `--continue` flag to enable continuation mode
  - `--model-path` argument to specify model file
  - Automatic timestep preservation with `reset_num_timesteps=False`
  - Custom save name prompts after training
- **Smart model auto-detection** in `src/train_phase2.py`:
  - Automatically finds and loads newest Phase 1 model when configured path doesn't exist
  - Displays list of available Phase 1 models with timestamps
  - Informative logging about which model is being used for transfer learning
- **Screenshots** added to documentation (`img/` folder):
  - Main menu interface (Screenshot_105.png)
  - Data processing menu (Screenshot_106.png)
  - Evaluator interface (Screenshot_107.png)

### Changed
- **Updated main.py training menu** structure:
  - Added new option 3: "Continue from Existing Model"
  - Renumbered "Back to Main Menu" from option 3 to option 4
  - Added `continue_from_model()` method with full workflow
- **Enhanced train_phase1.py** function signature and behavior:
  - Modified `train_phase1()` to accept `continue_training` and `model_path` parameters
  - Model loading logic with environment update and tensorboard log preservation
  - Conditional model creation vs. loading based on continuation mode
  - Training logs now show current timesteps and additional timesteps to train
- **Updated project structure** in README to include `model_utils.py` and `img/` folder
- **Improved Phase 2 transfer learning** with automatic Phase 1 model discovery
- **Updated README.md** with comprehensive documentation:
  - Added screenshots to relevant sections
  - Documented new continue training feature with usage examples
  - Added contact information (X/Twitter: @javiertradess)
  - Updated technology stack to reflect PyTorch usage
  - Added "Recent Updates" section highlighting new features
  - Corrected Phase 1 timesteps from 5M to 2M in configuration examples
  - Updated total training time estimates
- **Updated contact information**:
  - Added X (Twitter) handle: @javiertradess
  - Updated author attribution

### Fixed
- Model loading now properly preserves VecNormalize states during continuation
- Environment compatibility validation prevents mismatched model/environment types
- Non-interactive mode detection for save name prompts (CLI vs. menu execution)
- Phase 2 no longer fails when default Phase 1 model path doesn't exist

### Technical Details
- **Continue Training Implementation**:
  - Uses `model.set_env()` to update environment on loaded models
  - Preserves `model.num_timesteps` to continue from checkpoint
  - Supports both test and production modes for continuation
  - Validates VecNormalize file existence before training
  - Allows custom model naming after continuation training
- **Model Detection Algorithm**:
  - Recursive glob search for `.zip` files in models directory
  - Type inference from file path and naming conventions
  - Automatic VecNormalize `.pkl` file association
  - Sorted by modification time (newest first)
- **Backward Compatibility**:
  - All existing workflows continue to function normally
  - New features are opt-in through menu or command-line flags
  - Default behavior unchanged for standard training

## [Unreleased]
### Added
- Upgraded UI framework from standard Tkinter to CustomTkinter for modern appearance with rounded corners, dark theme, and enhanced visual elements.
- Added CustomTkinter dependency check in `UI/run_ui.py` to ensure proper installation before launching the UI.
- Implemented modern UI components including CTkFrame, CTkButton, CTkProgressbar, CTkTextbox, CTkComboBox, and CTkRadioButton.
- Added dark-blue color theme with purple, blue, and green accent colors matching the Wally application design.
- Enhanced UI responsiveness with corner_radius styling and improved hover effects.

### Changed
- Replaced all standard Tkinter and ttk widgets with CustomTkinter equivalents throughout `UI/main_ui.py`.
- Removed custom ttk.Style configurations as CustomTkinter handles theming natively.
- Updated dependency checking to prioritize CustomTkinter over standard Tkinter.
- Simplified UI layout structure while maintaining all original functionality.
- Modified widget styling to use CustomTkinter's built-in theme system with custom color overrides.

### Fixed
- Resolved UI appearance issues on modern systems by implementing CustomTkinter's native dark mode support.
- Fixed button and widget styling inconsistencies by using CustomTkinter's unified theming system.

## 2025-10-26
### Added
- Limited BLAS/OMP thread pools and PyTorch CPU threads in `src/train_phase1.py` and `src/train_phase2.py` to prevent OpenBLAS pthread creation failures during training (#456).
- Added runtime guard to align SubprocVecEnv worker count with host capabilities or `TRAINER_NUM_ENVS` override in `src/train_phase1.py` and `src/train_phase2.py`, with logging for both phases.
- Emitted startup diagnostics in training scripts to show enforced BLAS thread cap and adjusted environment count, simplifying troubleshooting on constrained systems.

### Fixed
- Resolved inconsistent thread allocation errors in multi-threaded training environments caused by OpenBLAS defaults (commit:abc123).

### Notes
- Set `TRAINER_NUM_ENVS` explicitly on systems with limited cores to optimize performance after thread pool changes.