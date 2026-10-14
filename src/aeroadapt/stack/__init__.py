"""Flight stack: estimation, guidance, control and contingency logic.

Everything here runs on sensor data only. Importing the simulator (or anything that
exposes true state) is forbidden and enforced in CI by import-linter (ADR-006).
"""
