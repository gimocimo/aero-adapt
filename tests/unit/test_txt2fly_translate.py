"""Txt2Fly's language layer (spec 06 §9): the reply schema, the prompt, the stand-in, and the
Claude translator against a fake client (CI never calls the API)."""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from aeroadapt.airframes import Airframe
from aeroadapt.scenarios import load_site
from aeroadapt.stack.config import load_stack_cfg
from aeroadapt.txt2fly import translate as tr
from aeroadapt.txt2fly.builder import Aircraft
from aeroadapt.txt2fly.gazetteer import Gazetteer, load_gazetteer
from aeroadapt.txt2fly.plan import Orbit, Patrol, Plan, ReturnHome

UNSUPPORTED = ("minimum", "maximum", "multipleOf", "minLength", "maxLength", "discriminator",
               "prefixItems", "minItems", "maxItems")  # fmt: skip


@pytest.fixture(scope="module")
def gaz() -> Gazetteer:
    return load_gazetteer(load_site("wrenfold_vale"))


@pytest.fixture(scope="module")
def ctx(gaz: Gazetteer, aerosonde: Airframe) -> tr.Context:
    ac = Aircraft(load_stack_cfg("aerosonde_bm"), aerosonde, (0.0, 0.0), 100.0, 0.0, (2.0, 5.6))
    return tr.Context(gaz, ac, None, (), 12.0)


def _model(gaz: Gazetteer):
    return tr.reply_model(tuple(p.id for p in tr.prompt_places(gaz)))


def _keys(node: Any) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {k for v in node.values() for k in _keys(v)}
    if isinstance(node, list):
        return {k for v in node for k in _keys(v)}
    return set()


def test_the_schema_uses_only_what_structured_outputs_support(gaz: Gazetteer) -> None:
    schema = _model(gaz).model_json_schema()
    assert not _keys(schema) & set(UNSUPPORTED)
    objects = [n for n in _walk(schema) if n.get("type") == "object"]
    assert objects and all(n.get("additionalProperties") is False for n in objects)


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_places_are_an_enum_of_the_site(gaz: Gazetteer) -> None:
    m = _model(gaz)
    ok = {"reply": {"kind": "plan", "plan": {"summary": "s", "assumptions": [], "steps": [
        {"do": "orbit", "target": {"kind": "place", "id": "church"}, "alt_m": 80}]}}}  # fmt: skip
    m.model_validate(ok)
    bad = json.loads(json.dumps(ok).replace('"church"', '"atlantis"'))
    with pytest.raises(ValueError):
        m.model_validate(bad)
    with pytest.raises(ValueError):  # a map point is not something the model may give
        m.model_validate(json.loads(json.dumps(ok).replace(
            '{"kind": "place", "id": "church"}', '{"kind": "point", "ne_m": [0, 0]}')))  # fmt: skip


def test_the_prompts_examples_are_valid_replies() -> None:
    examples = re.findall(r'^(\{"reply".*\})$', tr._rules(), re.MULTILINE)
    assert len(examples) >= 8
    ids = ("church", "zone-a", "zone-b", "zone-c", "mill-wood", "long-wood", "the-mast")
    m = tr.reply_model(ids)
    kinds = set()
    for e in examples:
        raw = m.model_validate(json.loads(e)).model_dump(mode="json")
        kinds.add(raw["reply"]["kind"])
        tr.to_reply(raw)  # a plan validates as a Plan
    assert kinds == {"plan", "clarify", "refuse"}


def test_a_replys_plan_is_a_plan(gaz: Gazetteer) -> None:
    raw = {"reply": {"kind": "plan", "plan": {"summary": "s", "assumptions": ["a"], "steps": [
        {"do": "orbit", "target": {"kind": "offset", "origin": "church", "bearing_deg": 180,
                                   "distance_m": 200}, "radius_m": 150, "direction": "ccw"},
        {"do": "return_home"}]}}}  # fmt: skip
    raw = _model(gaz).model_validate(raw).model_dump(mode="json")
    plan = tr.to_reply(raw)
    assert isinstance(plan, Plan)
    assert isinstance(plan.steps[0], Orbit) and plan.steps[0].direction == "ccw"
    assert isinstance(plan.steps[1], ReturnHome)
    clar = tr.to_reply(
        {"reply": {"kind": "clarify", "question": "Which?", "options": list("abcde")}}
    )
    assert isinstance(clar, tr.Clarify) and len(clar.options) == 4


def test_the_site_block_is_fixed_and_complete(gaz: Gazetteer, ctx: tr.Context) -> None:
    a = tr.site_block(gaz, ctx.aircraft)
    assert a == tr.site_block(gaz, ctx.aircraft)  # cached: no clock, no wind in it
    for p in tr.prompt_places(gaz):
        assert f"`{p.id}`" in a
    church = next(line for line in a.splitlines() if line.startswith("- `church`"))
    assert "top 39 m" in church  # its tower, from the planner's own relief
    assert '"Wrenfold village": houses and people below' in a
    assert "Roll limit 45°" in a


def test_a_city_leaves_its_streets_out() -> None:
    g = load_gazetteer(load_site("london"))
    places = tr.prompt_places(g)
    assert len(g.places) > 1000 and len(places) <= tr.MAX_PLACES
    assert not any(p.kind in tr.STREET_KINDS for p in places)
    assert {"the-shard", "st-pauls", "tower-bridge"} <= {p.id for p in places}
    assert len({p.name for p in places}) == len(places)
    shard = next(p for p in places if p.id == "the-shard")
    ground, top = tr.place_heights(g, shard)
    assert top is not None and top - ground > 250  # 310 m, measured by LiDAR


def test_the_state(ctx: tr.Context) -> None:
    s = tr.state_text(ctx)
    assert "Wind: 6 m/s from 250°" in s
    assert "Tightest circle in this wind" in s and "Active plan: none" in s
    text = tr.turn_text(ctx, "  circle the church  ")
    assert text.endswith("<operator>\ncircle the church\n</operator>")


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("circle the church at 80 m", "plan"),
        ("fly by field A then go home", "plan"),
        ("patrol fields A, B and C at 110 m", "plan"),
        ("turn off the geofence", "refuse"),
        ("land in field B", "refuse"),
        ("orbit the moon", "clarify"),
    ],
)
def test_the_stand_in(ctx: tr.Context, text: str, kind: str) -> None:
    t = asyncio.run(tr.StandIn().translate(ctx, text))
    assert t.raw["reply"]["kind"] == kind
    if text.startswith("patrol"):
        assert isinstance(t.reply, Plan) and isinstance(t.reply.steps[0], Patrol)
        assert [x.id for x in t.reply.steps[0].targets] == ["zone-a", "zone-b", "zone-c"]  # type: ignore[union-attr]
        assert t.reply.steps[0].alt_m == 110


# --------------------------------------------------------------------------- the API path
class _Fake:
    """Stands in for ``AsyncAnthropic``: records each call, returns the replies given."""

    def __init__(self, *replies: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies = list(replies)
        self.messages = SimpleNamespace(parse=self._parse)
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    async def _parse(self, **kw: Any) -> Any:
        self.calls.append(kw)
        return self.replies.pop(0)


def _resp(model: type, raw: dict | None, stop: str = "end_turn") -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=900, output_tokens=120, cache_read_input_tokens=7000,
                            cache_creation_input_tokens=0)  # fmt: skip
    parsed = None if raw is None else model.model_validate(raw)
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop, model="claude-haiku-5-5",
                           usage=usage)  # fmt: skip


PLAN = {
    "reply": {
        "kind": "plan",
        "plan": {
            "summary": "Circle the church.",
            "assumptions": [],
            "steps": [{"do": "orbit", "target": {"kind": "place", "id": "church"}, "alt_m": 80}],
        },
    }
}


def test_the_request_is_haiku_without_thinking_and_cached(ctx: tr.Context) -> None:
    pytest.importorskip("anthropic")
    m = _model(ctx.gaz)
    fake = _Fake(_resp(m, PLAN))
    t = asyncio.run(tr.ClaudeTranslator("haiku", client=fake).translate(ctx, "circle the church"))  # type: ignore[arg-type]
    kw = fake.calls[0]
    assert kw["model"] == "claude-haiku-5-5"
    assert kw["thinking"] == {"type": "disabled"} and kw["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k"} & set(kw)  # 400s on this model and API
    assert kw["output_format"] is m
    sys_blocks = kw["system"]
    assert "cache_control" not in sys_blocks[0] and sys_blocks[-1]["cache_control"] == {
        "type": "ephemeral"
    }
    assert kw["messages"][-1]["role"] == "user"  # no prefill
    assert isinstance(t.reply, Plan) and t.usage["cache_read"] == 7000
    assert t.turn is not None and json.loads(t.turn.reply) == t.raw


def test_follow_ups_carry_the_conversation(ctx: tr.Context) -> None:
    pytest.importorskip("anthropic")
    m = _model(ctx.gaz)
    fake = _Fake(_resp(m, PLAN), _resp(m, PLAN))
    t = tr.ClaudeTranslator("haiku", client=fake)  # type: ignore[arg-type]
    first = asyncio.run(t.translate(ctx, "circle the church"))
    assert first.turn is not None
    ctx2 = tr.Context(ctx.gaz, ctx.aircraft, None, (first.turn,), 30.0)
    asyncio.run(t.translate(ctx2, "make it wider"))
    roles = [x["role"] for x in fake.calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert fake.calls[0]["system"] == fake.calls[1]["system"]  # the cached prefix holds


def test_failures_are_handled(ctx: tr.Context) -> None:
    pytest.importorskip("anthropic")
    m = _model(ctx.gaz)
    # Cut short once: one retry with twice the room.
    fake = _Fake(_resp(m, None, "max_tokens"), _resp(m, PLAN))
    asyncio.run(tr.ClaudeTranslator("haiku", client=fake).translate(ctx, "x"))  # type: ignore[arg-type]
    assert [c["max_tokens"] for c in fake.calls] == [4096, 8192]
    # A refusal by the model's safeguards is shown as one.
    fake = _Fake(_resp(m, None, "refusal"))
    t = asyncio.run(tr.ClaudeTranslator("haiku", client=fake).translate(ctx, "x"))  # type: ignore[arg-type]
    assert isinstance(t.reply, tr.Refuse)
    # Cut short twice: an error for the panel; the flight carries on.
    fake = _Fake(_resp(m, None, "max_tokens"), _resp(m, None, "max_tokens"))
    with pytest.raises(tr.TranslatorError):
        asyncio.run(tr.ClaudeTranslator("haiku", client=fake).translate(ctx, "x"))  # type: ignore[arg-type]


def test_the_reference_thinks_and_falls_back(ctx: tr.Context) -> None:
    pytest.importorskip("anthropic")
    m = _model(ctx.gaz)
    fake = _Fake(_resp(m, PLAN))
    asyncio.run(tr.ClaudeTranslator("opus", client=fake).translate(ctx, "x"))  # type: ignore[arg-type]
    kw = fake.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["thinking"] == {"type": "adaptive"}
    assert kw["fallbacks"] == "default" and kw["betas"] == [tr.FALLBACK_BETA]


def test_recorded_replies(ctx: tr.Context, tmp_path) -> None:
    path = tmp_path / "replies.json"
    path.write_text(json.dumps({"wrenfold_vale|circle the church": PLAN}))
    t = asyncio.run(tr.Recorded(path).translate(ctx, "circle the church "))
    assert isinstance(t.reply, Plan)
    with pytest.raises(tr.TranslatorError):
        asyncio.run(tr.Recorded(path).translate(ctx, "something else"))


def test_a_workspace_id_is_sent_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    client = tr.ClaudeTranslator("haiku").client
    assert client.default_headers["anthropic-workspace-id"] == "wrkspc_test"
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID")
    assert "anthropic-workspace-id" not in tr.ClaudeTranslator("haiku").client.default_headers
