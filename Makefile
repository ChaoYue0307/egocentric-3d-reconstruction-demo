PYTHON ?= python3

.PHONY: test help visuals colmap-check pages

test:
	$(PYTHON) -m pytest -q

help:
	$(PYTHON) scripts/reconstruction_demo.py --help

visuals:
	$(PYTHON) scripts/render_reconstruction_visuals.py

colmap-check:
	$(PYTHON) scripts/run_colmap_if_available.py

pages:
	@echo "https://chaoyue0307.github.io/egocentric-3d-reconstruction-demo/"
