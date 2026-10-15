"""Scenario → everything the closed loop needs (configs loaded, parameters built)."""

from __future__ import annotations

from dataclasses import dataclass

from aeroadapt.airframes import Airframe, load_airframe
from aeroadapt.faultlab import FaultConfig, build_fault_config
from aeroadapt.scenarios.schema import Scenario
from aeroadapt.scenarios.site import Site, load_site
from aeroadapt.sim.aerojax import EnvParams, SimConfig, make_env
from aeroadapt.stack.config import StackCfg, StackParams, build_stack_params, load_stack_cfg
from aeroadapt.stack.types import Mission, make_mission
from aeroadapt.vehicle.protocol import InitialCondition


@dataclass(frozen=True)
class ResolvedScenario:
    scenario: Scenario
    airframe: Airframe
    site: Site
    env: EnvParams
    faults: FaultConfig
    initial: InitialCondition
    mission: Mission
    stack_cfg: StackCfg
    stack: StackParams
    config: SimConfig

    @property
    def n_steps(self) -> int:
        return round(self.scenario.duration_s / self.config.dt_control)


def build_env(scenario: Scenario, site: Site) -> EnvParams:
    e = scenario.environment
    return make_env(
        alt_origin_msl_m=site.ground_msl_m,
        magnetic_ned_gauss=site.magnetic.field_ned_gauss,
        wind_from_deg=e.wind.from_deg,
        wind_speed_mps=e.wind.speed_mps,
        w20_mps=e.turbulence.w20_mps,
        bl_alpha=e.wind.shear_alpha,
        temp_offset_k=site.isa.temp_offset_k if e.temp_offset_k is None else e.temp_offset_k,
        qnh_hpa=site.isa.qnh_hpa if e.qnh_hpa is None else e.qnh_hpa,
    )


def resolve(scenario: Scenario, config: SimConfig = SimConfig()) -> ResolvedScenario:  # noqa: B008
    airframe = load_airframe(scenario.airframe)
    site = load_site(scenario.site)
    stack_cfg = load_stack_cfg(scenario.stack.params or scenario.airframe)
    if stack_cfg.airframe != airframe.id:
        raise ValueError(f"stack params {stack_cfg.id} are for {stack_cfg.airframe}")
    sp = build_stack_params(stack_cfg, airframe, site, config.dt_control)
    if scenario.stack.contingency == "none":
        sp = sp._replace(cont=sp.cont._replace(enabled=False))
    if scenario.stack.motor_detector is not None:
        sp = sp._replace(cont=sp.cont._replace(esc_enabled=scenario.stack.motor_detector == "esc"))
    m, ini = scenario.mission, scenario.initial
    home = m.home_ne_m if m.home_ne_m is not None else ini.pos_ne_m
    mission = make_mission(
        m.waypoints_3d(),
        loop=m.on_complete == "loop",
        speed=m.speed_mps,
        home=(home[0], home[1], m.alt_agl_m),
    )
    initial = InitialCondition(
        pos_ne_m=ini.pos_ne_m,
        alt_agl_m=ini.alt_agl_m,
        heading_deg=ini.heading_deg,
        airspeed_mps=ini.airspeed_mps,
        soc=ini.soc,
    )
    return ResolvedScenario(
        scenario=scenario,
        airframe=airframe,
        site=site,
        env=build_env(scenario, site),
        faults=build_fault_config(scenario.faults),
        initial=initial,
        mission=mission,
        stack_cfg=stack_cfg,
        stack=sp,
        config=config,
    )
