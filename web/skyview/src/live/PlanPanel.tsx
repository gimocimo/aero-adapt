// Txt2Fly (docs/06 §10): say where to go in plain words, or type a plan. Either way the plan is
// built and checked against the aircraft's state now, shown on the map, and flown from the next
// chunk once confirmed. The words go to the server's translator (Claude, or a stand-in without
// an API key), which answers with a plan, a question or a refusal.
import { useEffect, useRef, useState } from "react";
import { useStore } from "../state/store";
import type { AskEntry, PlanCheck, PlanProposal } from "./client";

const place = (id: string) => ({ kind: "place", id });

type Example = { label: string; plan: Record<string, unknown> };

/** Example plans per site (ids and names from its gazetteer). */
export const EXAMPLES: Record<string, Example[]> = {
  cambridge_west: [
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
  ],
  wrenfold_vale: [
    {
      label: "Fly by field A",
      plan: { summary: "A pass over field A (Mill Meadow) along its long axis.", steps: [{ do: "pass_over", target: place("field A") }] },
    },
    {
      label: "Patrol A, B, C",
      plan: {
        summary: "Patrol fields A, B and C at 110 m, lap after lap, until the battery reserve.",
        steps: [{ do: "patrol", targets: [place("field A"), place("field B"), place("field C")], alt_m: 110 }],
      },
    },
    {
      label: "Circle field B",
      plan: {
        summary: "Circle field B (Long Field) twice at 100 m, then come home.",
        steps: [{ do: "orbit", target: place("field B"), alt_m: 100, turns: 2 }, { do: "return_home" }],
      },
    },
    {
      label: "Church, 30 m circle",
      plan: {
        summary: "Go to the church and circle it at 30 m radius, at 80 m.",
        steps: [{ do: "orbit", target: place("St Michael's Church"), radius_m: 30, alt_m: 80 }],
      },
    },
    {
      label: "Hanger Wood, 50 m",
      plan: {
        summary: "Circle Hanger Wood once at 50 m, then come home.",
        steps: [{ do: "orbit", target: place("Hanger Wood"), alt_m: 50, turns: 1 }, { do: "return_home" }],
      },
    },
    { label: "Return home", plan: { summary: "Return home.", steps: [{ do: "return_home" }] } },
  ],
};

export function examplesFor(siteId: string | undefined): Example[] {
  return EXAMPLES[siteId ?? ""] ?? EXAMPLES.cambridge_west!;
}

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
      {p.assumptions.map((n, i) => (
        <p key={i} className="note">
          Reading: {n}
        </p>
      ))}
      {p.notes.length > 0 && (
        <div className="changes" data-testid="plan-notes">
          <h3>How it will fly it</h3>
          <ul>
            {p.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </div>
      )}
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

function Answer({ e, ask }: { e: AskEntry; ask: (text: string) => void }) {
  if (e.kind === null) return <p className="answer pending">thinking…</p>;
  if (e.kind === "plan") return <p className="answer plan">Plan: {e.summary || "see below"}</p>;
  if (e.kind === "refuse") return <p className="answer refuse">{e.reason}</p>;
  if (e.kind === "error") return <p className="answer error">{e.message}</p>;
  return (
    <div className="answer clarify">
      <p>{e.question}</p>
      {!!e.options?.length && (
        <div className="options">
          {e.options.map((o) => (
            <button key={o} onClick={() => ask(o)}>
              {o}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** The conversation and the text box. */
function AskBox() {
  const live = useStore((s) => s.live);
  const send = useStore((s) => s.liveSend);
  const [text, setText] = useState("");
  const end = useRef<HTMLLIElement>(null);
  const conversation = live?.conversation ?? [];
  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest" });
  }, [conversation.length]);
  if (!live || !send || !live.translator?.name) return null;
  const busy = conversation.some((e) => e.kind === null);
  const ask = (t: string) => {
    if (!t.trim() || busy) return;
    send({ cmd: "ask", text: t.trim() });
    setText("");
  };
  return (
    <div className="ask" data-testid="ask">
      {conversation.length > 0 && (
        <ul className="conversation" data-testid="conversation">
          {conversation.map((e, i) => (
            <li key={i} ref={i === conversation.length - 1 ? end : undefined}>
              <p className="said">{e.text}</p>
              <Answer e={e} ask={ask} />
            </li>
          ))}
        </ul>
      )}
      <form
        onSubmit={(ev) => {
          ev.preventDefault();
          ask(text);
        }}
      >
        <input
          value={text}
          onChange={(ev) => setText(ev.target.value)}
          placeholder="Tell it where to go: circle the church at 80 m"
          aria-label="instruction"
          data-testid="ask-input"
          disabled={live.done}
        />
        <button type="submit" disabled={busy || !text.trim() || live.done} data-testid="ask-send">
          Send
        </button>
      </form>
      <p className="attribution">Translator: {live.translator.note}</p>
    </div>
  );
}

export function PlanPanel() {
  const live = useStore((s) => s.live);
  const send = useStore((s) => s.liveSend);
  const examples = examplesFor(live?.site?.id);
  const [text, setText] = useState(pretty(examples[0]!.plan));
  const [error, setError] = useState<string | null>(null);
  const [ack, setAck] = useState<Record<string, boolean>>({});
  const p = live?.proposal ?? null;
  const ready = canConfirm(p, ack);
  // A new proposal starts unacknowledged.
  useEffect(() => {
    setAck({});
  }, [p?.id]);
  // Enter confirms a confirmable proposal (outside a text field): one key press.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Enter" || !ready || !p || !send) return;
      if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) return;
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
    send({ cmd: "propose_plan", plan: r.plan });
  };
  return (
    <div className="planpanel" data-testid="plan-panel">
      {live.activePlan && (
        <p className="flying" data-testid="active-plan">
          Flying: {live.activePlan.summary || live.activePlan.steps.map((s) => s.label).join(", ")}
        </p>
      )}
      <AskBox />
      {p && (
        <>
          <div className="planbuttons">
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
          </div>
          <Proposal p={p} ack={ack} setAck={setAck} />
        </>
      )}
      <details className="typed" open={!live.translator?.name}>
        <summary data-testid="typed-plan">Typed plan</summary>
        <div className="examples">
          {examples.map((ex) => (
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
        </div>
        {error && <p className="faulterror">{error}</p>}
      </details>
      {live.site.attribution && <p className="attribution">Places: {live.site.attribution}</p>}
      <p className="attribution">Heights are above home (the ground where the aircraft took off).</p>
    </div>
  );
}
