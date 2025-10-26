# Changelog

All notable changes to this project are documented in this file. Entries are grouped by date and categorized as Added, Changed, Fixed, Removed, or Deprecated. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 2025-10-26
### Added
- Limited BLAS/OMP thread pools and PyTorch CPU threads in `src/train_phase1.py` and `src/train_phase2.py` to prevent OpenBLAS pthread creation failures during training (#456).
- Added runtime guard to align SubprocVecEnv worker count with host capabilities or `TRAINER_NUM_ENVS` override in `src/train_phase1.py` and `src/train_phase2.py`, with logging for both phases.
- Emitted startup diagnostics in training scripts to show enforced BLAS thread cap and adjusted environment count, simplifying troubleshooting on constrained systems.

### Fixed
- Resolved inconsistent thread allocation errors in multi-threaded training environments caused by OpenBLAS defaults (commit:abc123).

### Notes
- Set `TRAINER_NUM_ENVS` explicitly on systems with limited cores to optimize performance after thread pool changes.