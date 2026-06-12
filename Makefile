PYTHON ?= python3

.PHONY: test help visuals colmap-check hand-masks pages

test:
	$(PYTHON) -m pytest -q

help:
	$(PYTHON) scripts/reconstruction_demo.py --help

visuals:
	$(PYTHON) scripts/render_reconstruction_visuals.py

colmap-check:
	$(PYTHON) scripts/run_colmap_if_available.py

hand-masks:
	$(PYTHON) scripts/reconstruction_demo.py --data-root ../data/sample/xperience-10m-sample --output-dir outputs/cam1_hand_mask_demo --video fisheye_cam1.mp4 --camera-name cam1 --frame-stride 180 --max-frames 24 --write-hand-masks

pages:
	@echo "https://chaoyue0307.github.io/egocentric-3d-reconstruction-demo/"
