"""Regenerate tools/jsbsim/aircraft/aerosonde_bm/aerosonde_bm.xml from the aerojax airframe."""

from __future__ import annotations

from aeroadapt.airframes import load_airframe, to_params
from aeroadapt.sim.jsbsim.export import airframe_xml
from aeroadapt.sim.jsbsim.fdm import JSBSIM_ROOT

DESCRIPTION = "Beard and McLain Aerosonde, exported from aerojax (V14)"

if __name__ == "__main__":
    out = JSBSIM_ROOT / "aircraft" / "aerosonde_bm" / "aerosonde_bm.xml"
    out.write_text(
        airframe_xml(to_params(load_airframe("aerosonde_bm")), "aerosonde_bm", DESCRIPTION)
    )
    print(out)
