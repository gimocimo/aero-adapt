// Txt2Fly (docs/06 §10): a typed plan is built and checked against the aircraft's state now,
// shown on the map, and flown from the next chunk once confirmed. (The language layer, in week
// 10, puts a text box in front of the same steps.)
import { useEffect, useState } from "react";
import { useStore } from "../state/store";
import type { PlanCheck, PlanProposal } from "./client";

const place = (id: string) => ({ kind: "place", id });

/** Example plans for the Cambridge site (ids and names from its gazetteer). */
export const EXAMPLES: { label: string; plan: Record<string, unknown> }[] = [
  {
    label: "Fly by field A",
    plan: { summary: "A pass over field A along its long axis.", steps: [{ do: "pass_over", target: place("field A") }] },
  },
  {
    label: "Patrol A, B, C",
    plan: {
      summary: "Patrol fields A, B and C, lap after lap, until the battery reserve.",
      steps: [{ do: "patrol", targets: [place("field A"), place("field B"), place("field C")], alt_m: 100 }],
    },
  },
  {
    label: "Circle field C",
    plan: {
      summary: "Circle field C twice at 100 m, then come home.",
      steps: [{ do: "orbit", target: place("field C"), alt_m: 100, turns: 2 }, { do: "return_home" }],
    },
  },
  {
    label: "Church, 30 m circle",
    plan: {
      summary: "Go to the church and circle it at 30 m radius, 80 m above it.",
      assumptions: ["80 m above the ground at the church"],
      steps: [{ do: "orbit", target: place("Saint Peter (Coton Parish Church)"), radius_m: 30, alt_m: 80 }],
    },
  },
  { label: "Return home", plan: { summary: "Return home.", steps: [{ do: "return_home" }] } },
];

const pretty = (plan: Record<string, unknown>) => JSON.stringify(plan, null, 2);

/** Parse the editor's text; a plan is a JSON object with a list of steps. */
export function parsePlan(text: string): { plan: Record<string, unknown> } | { error: string } {
  try {
    const plan: unknown = JSON.parse(text);
    if (typeof plan !== "object" || plan === null || !Array.isArray((plan as { steps?: unknown }).steps)) {
      return { error: "a plan is an object with a list of steps" };
    }
    return { plan: plan as Record<string, unknown> };
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }
}

/** Confirmable: every hard rule passes and every failed soft rule is acknowledged. */
export function canConfirm(p: PlanProposal | null, ack: Record<string, boolean>): boolean {
  return !!p && p.ok && p.warnings.every((w) => ack[w]);
}

function CheckRow({ c }: { c: PlanCheck }) {
  const mark = c.ok ? "✓" : c.hard ? "✗" : "⚠";
  const cls = c.ok ? "ok" : c.hard ? "fail" : "warn";
  return (
    <li className={`check ${cls}`}>
      <span className="mark">{mark}</span>
      <span>
        <b>{c.title}.</b> {c.reason}
        {!c.ok && c.suggestion && <em> Try: {c.suggestion}.</em>}
      </span>
    </li>
  );
}

function Proposal({ p, ack, setAck }: { p: PlanProposal; ack: Record<string, boolean>; setAck: (a: Record<string, boolean>) => void }) {
  const failed = p.checks.filter((c) => c.hard && !c.ok).length;
  const verdict = !p.ok
    ? `Cannot fly: ${failed} check${failed > 1 ? "s" : ""} failed`
    : p.warnings.length
      ? "Can fly once the warning is acknowledged"
      : "Can fly";
  return (
    <div className="proposal" data-testid="plan-proposal">
      <p className={`verdict ${p.ok ? "ok" : "fail"}`} data-testid="plan-verdict">
        {verdict}
      </p>
      {p.summary && <p className="summary">{p.summary}</p>}
      <ol className="steps">
        {p.steps.map((s, i) => (
          <li key={i}>
            {s.label}
            {s.radius !== null && ` · ${s.radius} m circle`}
            {s.alt !== null && ` · ${s.alt} m`}
          </li>
        ))}
      </ol>
      {[...p.assumptions, ...p.notes].map((n, i) => (
        <p key={i} className="note">
          {n}
        </p>
      ))}
      <ul className="checks">
        {p.checks.map((c) => (
          <CheckRow key={c.rule} c={c} />
        ))}
      </ul>
      {p.ok &&
        p.warnings.map((w) => (
          <label key={w} className="ack">
            <input type="checkbox" checked={!!ack[w]} onChange={(e) => setAck({ ...ack, [w]: e.target.checked })} />
            I accept the {p.checks.find((c) => c.rule === w)?.title.toLowerCase() ?? w} warning
          </label>
        ))}
    </div>
  );
}

export function PlanPanel() {
  const live = useStore((s) => s.live);
  const send = useStore((s) => s.liveSend);
  const [text, setText] = useState(pretty(EXAMPLES[0]!.plan));
  const [error, setError] = useState<string | null>(null);
  const [ack, setAck] = useState<Record<string, boolean>>({});
  const p = live?.proposal ?? null;
  const ready = canConfirm(p, ack);
  // Enter confirms a confirmable proposal (outside the editor): one key press.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Enter" || !ready || !p || !send || e.target instanceof HTMLTextAreaElement) return;
      send({ cmd: "confirm_plan", id: p.id, acknowledged: p.warnings });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [ready, p, send]);
  if (!live || !send || !live.site) return null;
  const check = () => {
    const r = parsePlan(text);
    if ("error" in r) {
      setError(r.error);
      return;
    }
    setError(null);
    setAck({});
    send({ cmd: "propose_plan", plan: r.plan });
  };
  return (
    <div className="planpanel" data-testid="plan-panel">
      {live.activePlan && (
        <p className="flying" data-testid="active-plan">
          Flying: {live.activePlan.summary || live.activePlan.steps.map((s) => s.label).join(", ")}
        </p>
      )}
      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex.label} onClick={() => setText(pretty(ex.plan))}>
            {ex.label}
          </button>
        ))}
      </div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        rows={9}
        aria-label="plan (JSON)"
        data-testid="plan-editor"
      />
      <div className="planbuttons">
        <button onClick={check} disabled={live.done} data-testid="plan-check">
          Check
        </button>
        {p && (
          <>
            <button
              className="confirm"
              disabled={!ready}
              onClick={() => send({ cmd: "confirm_plan", id: p.id, acknowledged: p.warnings.filter((w) => ack[w]) })}
              data-testid="plan-confirm"
              title="Fly it (Enter)"
            >
              Confirm
            </button>
            <button onClick={() => send({ cmd: "cancel_plan", id: p.id })}>Cancel</button>
          </>
        )}
      </div>
      {error && <p className="faulterror">{error}</p>}
      {p && <Proposal p={p} ack={ack} setAck={setAck} />}
      {live.site.attribution && <p className="attribution">Places: {live.site.attribution}</p>}
    </div>
  );
}
