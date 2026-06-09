# COLMAP Run Report

- COLMAP status: not available
- COLMAP path: None
- Command template: `outputs/sample_demo/colmap_commands.sh`
- Executed: False
- Exit code: None

## Notes

Install COLMAP and rerun:

```bash
brew install colmap
python scripts/run_colmap_if_available.py --run
```

When the command writes a COLMAP sparse text model, parse it with:

```bash
python scripts/reconstruction_demo.py --parse-colmap-only --colmap-model outputs/sample_demo/colmap/sparse/0 --output-dir outputs/sample_demo
```
