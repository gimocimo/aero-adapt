"""Closed-loop harnesses: simulator and flight stack together (privileged side, ADR-006)."""

from aeroadapt.harness.closed_loop import LoopLog, LoopSetup, closed_loop, setup_loop

__all__ = ["LoopLog", "LoopSetup", "closed_loop", "setup_loop"]
