.PHONY: setup lint format typecheck imports test validate check gltf goldens web-dev web-build web-test report demo fleet

setup:
	uv sync --all-extras
	cd web/skyview && npm ci

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run pyright

imports:
	uv run lint-imports

## Fast tier (every push, < 10 min on CI)
test:
	uv run pytest -m "not full" -q

## Full validation tier (nightly / before tagging a milestone)
validate:
	uv run pytest -q

check: lint typecheck imports test

## Validation report (docs/validation/m2_report.md + figures; m1_report.md is the frozen M1 artefact)
report:
	uv run python -m validation.report

## The M2 airframe fleet (runs/fleets/m2-fleet): 10,000 accepted airframes, ~20 min on 10 cores
fleet:
	uv run aeroadapt fleet m2-fleet --n 10000

## Demo video (runs/demo/m1_demo.mp4), regenerated from seeds
demo:
	uv run --with imageio-ffmpeg python tools/make_demo.py

## Calibration glTF model and cross-language frame goldens for the viewer
gltf:
	uv run python tools/make_gltf.py

goldens:
	uv run python tools/export_frame_goldens.py

web-dev:
	cd web/skyview && npm run dev

web-build:
	cd web/skyview && npm run build

web-test:
	cd web/skyview && npm run typecheck && npm test && npm run e2e
