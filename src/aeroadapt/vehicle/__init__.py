"""Vehicles behind one interface (ADR-004).

``protocol`` is safe to import anywhere. Adapters such as ``aerojax_vehicle`` are privileged
(they wrap the simulator) and are imported explicitly, never from this package root.
"""
