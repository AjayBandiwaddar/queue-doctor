import React, { useState, useCallback, useRef, useEffect } from "react";
import {
  Activity, Bed, Users, Clock, AlertTriangle, CheckCircle2, Pause,
  ChevronRight, RotateCcw, Siren, Stethoscope, TrendingUp, Info, Sun, Moon,
} from "lucide-react";

/* ────────────────────────────────────────────────────────────────────────
   QUEUE DOCTOR — client-side reimplementation of the deterministic engine
   in server/queue_engine.py + server/graders.py + server/tasks.py from
   ajaybandiwaddar01/queue-doctor. Reward math, deterioration, resource
   rules and grading formulas are ported 1:1 from that source, seeded
   with the real task arrival schedules. This runs entirely in the
   browser — it does not call the live OpenEnv/MCP backend.
   ──────────────────────────────────────────────────────────────────── */

const SEV = {
  1: { name: "IMMEDIATE", tag: "RED", hex: "#e5484d", glow: "rgba(229,72,77,0.35)" },
  2: { name: "VERY URGENT", tag: "ORANGE", hex: "#f0952d", glow: "rgba(240,149,45,0.3)" },
  3: { name: "URGENT", tag: "YELLOW", hex: "#e0b039", glow: "rgba(224,176,57,0.28)" },
  4: { name: "LESS URGENT", tag: "GREEN", hex: "#3ecf8e", glow: "rgba(62,207,142,0.25)" },
  5: { name: "NON-URGENT", tag: "BLUE", hex: "#4ea1ff", glow: "rgba(78,161,255,0.25)" },
};

const TASKS = {
  task_1_easy: {
    id: "task_1_easy",
    name: "Basic Triage",
    difficulty: "Easy",
    maxSteps: 10,
    numDoctors: 1,
    icuBeds: 0,
    grader: "easy",
    optimalReward: 3.96,
    baselineAvg: 0.881,
    brief: "One doctor, six patients, ten steps. One patient minimizes their own symptoms — the queue can't be trusted at face value.",
    arrivals: [
      { step: 0, id: "P001", severity: 1, reportedSeverity: 4 },
      { step: 0, id: "P002", severity: 2, reportedSeverity: 2 },
      { step: 0, id: "P003", severity: 3, reportedSeverity: 3 },
      { step: 0, id: "P004", severity: 3, reportedSeverity: 3 },
      { step: 0, id: "P005", severity: 4, reportedSeverity: 4 },
      { step: 0, id: "P006", severity: 5, reportedSeverity: 5 },
    ],
  },
  task_2_medium: {
    id: "task_2_medium",
    name: "Dynamic Queue Management",
    difficulty: "Medium",
    maxSteps: 20,
    numDoctors: 2,
    icuBeds: 0,
    grader: "medium",
    optimalReward: 14.073,
    baselineAvg: 0.730,
    brief: "Two doctors, patients arriving in ten waves. Two specialist cases eat both doctors at once, and speed alone won't save the score — fairness counts too.",
    arrivals: [
      { step: 0, id: "P001", severity: 3, reportedSeverity: 1 },
      { step: 0, id: "P002", severity: 2, reportedSeverity: 2 },
      { step: 0, id: "P003", severity: 4, reportedSeverity: 4 },
      { step: 0, id: "P004", severity: 5, reportedSeverity: 5 },
      { step: 2, id: "P005", severity: 1, reportedSeverity: 1 },
      { step: 2, id: "P006", severity: 2, reportedSeverity: 2, requiresSpecialist: true },
      { step: 4, id: "P007", severity: 1, reportedSeverity: 1 },
      { step: 4, id: "P008", severity: 3, reportedSeverity: 3 },
      { step: 6, id: "P009", severity: 4, reportedSeverity: 2 },
      { step: 6, id: "P010", severity: 3, reportedSeverity: 3 },
      { step: 8, id: "P011", severity: 2, reportedSeverity: 2 },
      { step: 8, id: "P012", severity: 4, reportedSeverity: 4 },
      { step: 10, id: "P013", severity: 1, reportedSeverity: 1 },
      { step: 10, id: "P014", severity: 2, reportedSeverity: 2, requiresSpecialist: true },
      { step: 10, id: "P015", severity: 3, reportedSeverity: 3 },
      { step: 13, id: "P016", severity: 2, reportedSeverity: 2 },
      { step: 13, id: "P017", severity: 4, reportedSeverity: 4 },
      { step: 15, id: "P023", severity: 2, reportedSeverity: 2 },
      { step: 15, id: "P024", severity: 3, reportedSeverity: 3 },
      { step: 15, id: "P025", severity: 4, reportedSeverity: 4 },
      { step: 15, id: "P026", severity: 2, reportedSeverity: 2 },
      { step: 15, id: "P027", severity: 3, reportedSeverity: 3 },
      { step: 17, id: "P028", severity: 2, reportedSeverity: 2 },
      { step: 17, id: "P029", severity: 3, reportedSeverity: 3 },
      { step: 17, id: "P030", severity: 4, reportedSeverity: 4 },
      { step: 17, id: "P031", severity: 2, reportedSeverity: 2 },
      { step: 17, id: "P032", severity: 3, reportedSeverity: 3 },
      { step: 19, id: "P033", severity: 2, reportedSeverity: 2 },
      { step: 19, id: "P034", severity: 3, reportedSeverity: 3 },
    ],
  },
  task_3_hard: {
    id: "task_3_hard",
    name: "Mass Casualty Resource Management",
    difficulty: "Hard",
    maxSteps: 30,
    numDoctors: 3,
    icuBeds: 1,
    grader: "hard",
    optimalReward: 10.05,
    baselineAvg: 0.693,
    brief: "Three doctors, one ICU bed, thirty steps. A five-patient surge lands at step 12 without warning — some of it will be unsurvivable no matter what you do.",
    arrivals: [
      { step: 0, id: "P001", severity: 1, reportedSeverity: 1, requiresIcu: true },
      { step: 0, id: "P002", severity: 2, reportedSeverity: 2, requiresSpecialist: true },
      { step: 0, id: "P003", severity: 3, reportedSeverity: 3 },
      { step: 0, id: "P004", severity: 4, reportedSeverity: 4 },
      { step: 3, id: "P005", severity: 2, reportedSeverity: 2, deteriorationCountdown: 3 },
      { step: 3, id: "P006", severity: 3, reportedSeverity: 3, deteriorationCountdown: 2 },
      { step: 6, id: "P007", severity: 1, reportedSeverity: 1 },
      { step: 6, id: "P008", severity: 3, reportedSeverity: 2 },
      { step: 6, id: "P009", severity: 4, reportedSeverity: 4 },
      { step: 9, id: "P010", severity: 2, reportedSeverity: 2, deteriorationCountdown: 4 },
      { step: 9, id: "P011", severity: 3, reportedSeverity: 3 },
      { step: 12, id: "P012", severity: 1, reportedSeverity: 1, requiresIcu: true },
      { step: 12, id: "P013", severity: 1, reportedSeverity: 1 },
      { step: 12, id: "P014", severity: 1, reportedSeverity: 1, requiresSpecialist: true },
      { step: 12, id: "P015", severity: 2, reportedSeverity: 2, requiresIcu: true },
      { step: 12, id: "P016", severity: 2, reportedSeverity: 2 },
      { step: 18, id: "P017", severity: 2, reportedSeverity: 2 },
      { step: 18, id: "P018", severity: 3, reportedSeverity: 3 },
      { step: 18, id: "P019", severity: 4, reportedSeverity: 4 },
      { step: 24, id: "P020", severity: 3, reportedSeverity: 3 },
      { step: 24, id: "P021", severity: 2, reportedSeverity: 2 },
    ],
  },
};

/* ── pure engine, ported from queue_engine.py ─────────────────────────── */

function computeReward(p) {
  const w = p.waitTime, s = p.severity;
  if (s === 1) return w === 0 ? 1.0 : w === 1 ? 0.6 : w === 2 ? 0.2 : 0.0;
  if (s === 2) return Math.max(0, 1.0 - w * 0.125);
  if (s === 3) return Math.max(0, 0.85 - w * 0.071);
  if (s === 4) return Math.max(0, 0.6 - w * 0.04);
  return Math.max(0, 0.4 - w * 0.02);
}

function canServe(p, s) {
  const need = p.requiresSpecialist ? 2 : 1;
  return s.availableDoctors >= need && (!p.requiresIcu || s.availableIcu >= 1);
}

function makePatient(a) {
  return {
    id: a.id,
    severity: a.severity,
    reportedSeverity: a.reportedSeverity,
    waitTime: 0,
    deteriorationCountdown: a.deteriorationCountdown ?? -1,
    requiresIcu: !!a.requiresIcu,
    requiresSpecialist: !!a.requiresSpecialist,
    condition: "stable",
  };
}

function initEpisode(taskId) {
  const task = TASKS[taskId];
  const schedule = {};
  task.arrivals.forEach((a) => {
    (schedule[a.step] ||= []).push(makePatient(a));
  });
  const queue = schedule[0] || [];
  delete schedule[0];
  return {
    taskId,
    step: 0,
    numDoctors: task.numDoctors,
    icuCapacity: task.icuBeds,
    availableDoctors: task.numDoctors,
    availableIcu: task.icuBeds,
    queue,
    schedule,
    served: [],
    missedEmergencies: 0,
    cumulativeReward: 0,
    events: [{ step: 0, text: "Episode started. Observe the queue and choose an action.", tone: "info" }],
    done: false,
    lastDelta: null,
  };
}

function advanceStep(state, events) {
  let queue = [...state.queue];
  const arrivals = state.schedule[state.step] || [];
  if (arrivals.length) {
    queue = [...queue, ...arrivals];
    events.push({
      step: state.step,
      text: `New arrivals — ${arrivals.map((p) => `${p.id} (reported ${p.reportedSeverity})`).join(", ")}`,
      tone: "arrival",
    });
    if (arrivals.length >= 4) {
      events.push({ step: state.step, text: `MASS CASUALTY EVENT — ${arrivals.length} patients arrived simultaneously.`, tone: "alarm" });
    }
  }
  const schedule = { ...state.schedule };
  delete schedule[state.step];

  queue = queue.map((p) => {
    if (p.deteriorationCountdown > 0) {
      const cd = p.deteriorationCountdown - 1;
      if (cd === 0) {
        const worsened = Math.max(1, p.severity - 1);
        events.push({ step: state.step, text: `${p.id} deteriorated — severity ${p.severity} → ${worsened}.`, tone: "alarm" });
        return { ...p, severity: worsened, reportedSeverity: worsened, condition: worsened === 1 ? "critical" : "at_risk", deteriorationCountdown: -1 };
      }
      return { ...p, deteriorationCountdown: cd };
    }
    return p;
  });

  let missed = state.missedEmergencies;
  queue = queue.map((p) => {
    if (p.severity === 1) missed += 1;
    return { ...p, waitTime: p.waitTime + 1 };
  });

  return {
    ...state,
    queue,
    schedule,
    availableDoctors: state.numDoctors,
    missedEmergencies: missed,
  };
}

function servePatient(state, patientId) {
  const patient = state.queue.find((p) => p.id === patientId);
  if (!patient) return state;
  const need = patient.requiresSpecialist ? 2 : 1;
  const events = [];

  if (state.availableDoctors < need) {
    events.push({ step: state.step, text: `Cannot serve ${patient.id} — needs ${need} doctors, ${state.availableDoctors} available.`, tone: "blocked" });
    return { ...state, events: [...events, ...state.events].slice(0, 60), lastDelta: null };
  }
  if (patient.requiresIcu && state.availableIcu < 1) {
    events.push({ step: state.step, text: `Cannot admit ${patient.id} — no ICU beds available.`, tone: "blocked" });
    return { ...state, events: [...events, ...state.events].slice(0, 60), lastDelta: null };
  }

  const reward = computeReward(patient);
  let note = "";
  if (patient.requiresSpecialist) note += " · 2 doctors used";
  const availableIcu = state.availableIcu - (patient.requiresIcu ? 1 : 0);
  if (patient.requiresIcu) note += ` · ICU bed consumed (${availableIcu}/${state.icuCapacity} left)`;

  events.push({
    step: state.step,
    text: `Served ${patient.id} — reported ${patient.reportedSeverity}, true ${patient.severity}, waited ${patient.waitTime} step(s) → reward ${reward.toFixed(3)}${note}`,
    tone: reward >= 0.6 ? "good" : reward > 0 ? "ok" : "bad",
  });

  let next = {
    ...state,
    queue: state.queue.filter((p) => p.id !== patientId),
    availableDoctors: state.numDoctors - need,
    availableIcu,
    cumulativeReward: state.cumulativeReward + reward,
    served: [...state.served, {
      patientId: patient.id, trueSeverity: patient.severity, reportedSeverity: patient.reportedSeverity,
      waitTime: patient.waitTime, reward, servedStep: state.step,
    }],
    step: state.step + 1,
    lastDelta: reward,
  };
  next = advanceStep(next, events);
  next.events = [...events, ...state.events].slice(0, 60);
  next.done = next.step >= TASKS[state.taskId].maxSteps;
  return next;
}

function waitAction(state) {
  const events = [];
  const servable = state.queue.filter((p) => canServe(p, state));
  let penalty = 0;
  if (servable.length) {
    const emergencies = servable.filter((p) => p.reportedSeverity === 1).length;
    const urgent = servable.filter((p) => p.reportedSeverity >= 2 && p.reportedSeverity <= 3).length;
    if (emergencies > 0) {
      penalty = -0.3 * emergencies;
      events.push({ step: state.step, text: `Held with ${emergencies} IMMEDIATE patient(s) waiting — penalty ${penalty.toFixed(2)}.`, tone: "bad" });
    } else if (urgent > 0) {
      penalty = -0.1;
      events.push({ step: state.step, text: `Held with urgent patient(s) waiting — penalty -0.10.`, tone: "bad" });
    } else {
      penalty = -0.05;
      events.push({ step: state.step, text: `Held with non-urgent patient(s) waiting — penalty -0.05.`, tone: "ok" });
    }
  } else if (state.queue.length) {
    events.push({ step: state.step, text: `${state.queue.length} patient(s) present but every one is resource-blocked — no penalty.`, tone: "info" });
  } else {
    events.push({ step: state.step, text: "Queue empty — no penalty.", tone: "info" });
  }
  let next = { ...state, cumulativeReward: state.cumulativeReward + penalty, step: state.step + 1, lastDelta: penalty };
  next = advanceStep(next, events);
  next.events = [...events, ...state.events].slice(0, 60);
  next.done = next.step >= TASKS[state.taskId].maxSteps;
  return next;
}

/* ── graders, ported from graders.py ──────────────────────────────────── */

function jfi(values) {
  if (!values.length || values.length === 1) return 1.0;
  const n = values.length, sum = values.reduce((a, b) => a + b, 0);
  const sum2 = values.reduce((a, b) => a + b * b, 0);
  if (sum2 === 0) return 1.0;
  return (sum * sum) / (n * sum2);
}

function grade(state) {
  const task = TASKS[state.taskId];
  const arrivals = task.arrivals;
  const totalReward = state.served.reduce((a, s) => a + s.reward, 0);

  if (task.grader === "easy") {
    const score = Math.min(0.999, Math.max(0.001, totalReward / task.optimalReward));
    return {
      score, totalReward, optimal: task.optimalReward,
      served: state.served.length, total: arrivals.length,
      lines: [
        ["Patients served", `${state.served.length} / ${arrivals.length}`],
        ["Cumulative reward", `${totalReward.toFixed(3)} / ${task.optimalReward.toFixed(3)} optimal`],
      ],
    };
  }

  if (task.grader === "medium") {
    if (!state.served.length) return { score: 0.001, served: 0, total: arrivals.length, lines: [["Patients served", "0"]] };
    const throughput = Math.min(1, totalReward / task.optimalReward);
    const servedFraction = state.served.length / arrivals.length;
    const effective = throughput * servedFraction;
    const fairness = jfi(state.served.map((s) => s.waitTime));
    const score = Math.min(0.999, Math.max(0.001, 0.6 * effective + 0.4 * fairness));
    return {
      score, totalReward, optimal: task.optimalReward,
      served: state.served.length, total: arrivals.length,
      lines: [
        ["Patients served", `${state.served.length} / ${arrivals.length} (${(servedFraction * 100).toFixed(0)}% coverage)`],
        ["Per-patient efficiency", throughput.toFixed(3)],
        ["Effective throughput (60%)", effective.toFixed(3)],
        ["Fairness — Jain's index (40%)", fairness.toFixed(3)],
      ],
    };
  }

  // hard
  if (!state.served.length) return { score: 0.001, served: 0, total: arrivals.length, lines: [["Patients served", "0"]] };
  const critical = arrivals.filter((a) => a.severity <= 2);
  const criticalServed = state.served.filter((s) => s.trueSeverity <= 2).length;
  const survival = critical.length ? criticalServed / critical.length : 1;
  const time = Math.min(1, totalReward / task.optimalReward);
  const fairness = jfi(state.served.map((s) => s.waitTime));
  const resourceArrivals = arrivals.filter((a) => a.requiresIcu || a.requiresSpecialist);
  const servedIds = new Set(state.served.map((s) => s.patientId));
  const resourceServed = resourceArrivals.filter((a) => servedIds.has(a.id)).length;
  const resource = resourceArrivals.length ? resourceServed / resourceArrivals.length : 1;
  const base = Math.min(0.999, Math.max(0.001, 0.35 * survival + 0.25 * time + 0.2 * fairness + 0.2 * resource));
  const missedPenalty = Math.min(0.55, state.missedEmergencies * 0.03);
  const score = Math.max(0, base - missedPenalty);
  return {
    score, totalReward, optimal: task.optimalReward,
    served: state.served.length, total: arrivals.length,
    lines: [
      ["Patients served", `${state.served.length} / ${arrivals.length}`],
      ["Survival — critical patients treated (35%)", survival.toFixed(3)],
      ["Time-to-treatment (25%)", time.toFixed(3)],
      ["Fairness — Jain's index (20%)", fairness.toFixed(3)],
      ["Resource efficiency — ICU/specialist (20%)", resource.toFixed(3)],
      ["Missed-emergency penalty", `-${missedPenalty.toFixed(3)} (${state.missedEmergencies} patient-steps)`],
    ],
  };
}

/* ── UI ──────────────────────────────────────────────────────────────── */

function Pulse() {
  return (
    <svg className="qd-pulse" viewBox="0 0 400 40" preserveAspectRatio="none">
      <polyline points="0,20 40,20 55,20 65,4 75,36 85,10 95,20 130,20 400,20" />
    </svg>
  );
}

function PatientCard({ p, servable, reason, onServe, sinceGlow }) {
  const meta = SEV[p.reportedSeverity];
  const dying = p.deteriorationCountdown > 0 && p.deteriorationCountdown <= 2;
  return (
    <button
      className={`qd-card ${servable ? "qd-card--live" : "qd-card--blocked"}`}
      style={{ "--sev": meta.hex, "--glow": meta.glow }}
      onClick={() => servable && onServe(p.id)}
      disabled={!servable}
    >
      <div className="qd-card-top">
        <span className="qd-tag">{meta.tag}</span>
        <span className="qd-id">{p.id}</span>
      </div>
      <div className="qd-card-name">{meta.name}</div>
      <div className="qd-card-meta">
        <span><Clock size={12} strokeWidth={2.5} /> waited {p.waitTime}</span>
        {p.requiresIcu && <span className="qd-chip">ICU</span>}
        {p.requiresSpecialist && <span className="qd-chip">2 doctors</span>}
      </div>
      {dying && (
        <div className="qd-warn"><AlertTriangle size={12} strokeWidth={2.5} /> worsening in {p.deteriorationCountdown} step{p.deteriorationCountdown === 1 ? "" : "s"}</div>
      )}
      {!servable && <div className="qd-blocked-reason">{reason}</div>}
      {servable && <div className="qd-serve-hint">serve <ChevronRight size={13} /></div>}
    </button>
  );
}

function ResultScreen({ result, task, onRestart, onPickAnother }) {
  const pct = Math.round(result.score * 100);
  return (
    <div className="qd-result">
      <div className="qd-result-score" style={{ "--pct": `${pct}%` }}>
        <svg viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="52" className="qd-ring-bg" />
          <circle cx="60" cy="60" r="52" className="qd-ring-fg" strokeDasharray={`${pct * 3.267} 1000`} />
        </svg>
        <div className="qd-result-score-num">
          <div className="qd-big">{result.score.toFixed(3)}</div>
          <div className="qd-small">normalized score</div>
        </div>
      </div>
      <h2>Episode finalized — {task.name}</h2>
      <p className="qd-result-compare">
        Llama-3.1-8B baseline average on this task: <strong>{task.baselineAvg.toFixed(3)}</strong>
        {" — "}{result.score >= task.baselineAvg ? "you beat the baseline agent." : "the baseline agent edged you out here."}
      </p>
      <div className="qd-breakdown">
        {result.lines.map(([label, val]) => (
          <div key={label} className="qd-breakdown-row">
            <span>{label}</span><span>{val}</span>
          </div>
        ))}
      </div>
      <div className="qd-result-actions">
        <button className="qd-btn qd-btn--primary" onClick={onRestart}><RotateCcw size={15} /> Retry this task</button>
        <button className="qd-btn" onClick={onPickAnother}>Choose another task</button>
      </div>
    </div>
  );
}

function TaskPicker({ onPick }) {
  return (
    <div className="qd-picker">
      <div className="qd-hero">
        <Pulse />
        <div className="qd-hero-text">
          <div className="qd-eyebrow"><Stethoscope size={13} /> Emergency Department Simulator</div>
          <h1>Queue Doctor</h1>
          <p>
            You're running triage. Patients arrive on a fixed schedule, self-report their own severity,
            and their condition keeps changing whether you act or not — this is the same task an RL agent
            is graded on in the original environment. Pick a shift.
          </p>
        </div>
      </div>
      <div className="qd-task-grid">
        {Object.values(TASKS).map((t) => (
          <button key={t.id} className="qd-task-card" onClick={() => onPick(t.id)}>
            <div className="qd-task-diff" data-diff={t.difficulty}>{t.difficulty}</div>
            <h3>{t.name}</h3>
            <p>{t.brief}</p>
            <div className="qd-task-stats">
              <span><Users size={13} /> {t.arrivals.length} patients</span>
              <span><Stethoscope size={13} /> {t.numDoctors} doctor{t.numDoctors > 1 ? "s" : ""}</span>
              {t.icuBeds > 0 && <span><Bed size={13} /> {t.icuBeds} ICU bed</span>}
              <span><Clock size={13} /> {t.maxSteps} steps</span>
            </div>
            <div className="qd-task-start">Start shift <ChevronRight size={14} /></div>
          </button>
        ))}
      </div>
      <p className="qd-footnote">
        <Info size={12} /> Reward math, deterioration and grading formulas are ported from the environment's
        own source (queue_engine.py / graders.py) — this runs the real logic client-side rather than calling
        the hosted OpenEnv backend.
      </p>
    </div>
  );
}

export default function QueueDoctor() {
  const [taskId, setTaskId] = useState(null);
  const [state, setState] = useState(null);
  const [result, setResult] = useState(null);
  const [theme, setTheme] = useState("dark");
  const logRef = useRef(null);

  const start = useCallback((id) => {
    setTaskId(id);
    setState(initEpisode(id));
    setResult(null);
  }, []);

  const handleServe = useCallback((pid) => setState((s) => servePatient(s, pid)), []);
  const handleWait = useCallback(() => setState((s) => waitAction(s)), []);
  const finalize = useCallback(() => setState((s) => { setResult(grade(s)); return s; }), []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = 0;
  }, [state?.events]);

  if (!taskId || !state) {
    return (
      <Shell theme={theme}>
        <ThemeToggle theme={theme} setTheme={setTheme} />
        <TaskPicker onPick={start} />
      </Shell>
    );
  }

  const task = TASKS[taskId];
  const sortedQueue = [...state.queue].sort((a, b) => a.reportedSeverity - b.reportedSeverity || b.waitTime - a.waitTime);
  const advisoryEmergencies = state.queue.filter((p) => p.reportedSeverity === 1);
  const advisoryDeteriorating = state.queue.filter((p) => p.deteriorationCountdown > 0 && p.deteriorationCountdown <= 2);

  return (
    <Shell theme={theme}>
      <ThemeToggle theme={theme} setTheme={setTheme} />
      {result ? (
        <ResultScreen
          result={result}
          task={task}
          onRestart={() => start(taskId)}
          onPickAnother={() => { setTaskId(null); setState(null); setResult(null); }}
        />
      ) : (
        <div className="qd-game">
          <div className="qd-topbar">
            <div className="qd-topbar-item">
              <span className="qd-label">Shift</span>
              <span className="qd-value">{task.name}</span>
            </div>
            <div className="qd-topbar-item">
              <span className="qd-label">Step</span>
              <span className="qd-value qd-mono">{state.step} / {task.maxSteps}</span>
            </div>
            <div className="qd-topbar-item">
              <Stethoscope size={14} />
              <span className="qd-value qd-mono">{state.availableDoctors}/{task.numDoctors}</span>
            </div>
            {task.icuBeds > 0 && (
              <div className="qd-topbar-item">
                <Bed size={14} />
                <span className="qd-value qd-mono">{state.availableIcu}/{task.icuBeds}</span>
              </div>
            )}
            <div className="qd-topbar-item">
              <TrendingUp size={14} />
              <span className={`qd-value qd-mono ${state.lastDelta > 0 ? "qd-up" : state.lastDelta < 0 ? "qd-down" : ""}`}>
                {state.cumulativeReward.toFixed(3)}
              </span>
            </div>
            {state.missedEmergencies > 0 && (
              <div className="qd-topbar-item qd-topbar-item--warn">
                <Siren size={14} />
                <span className="qd-value qd-mono">{state.missedEmergencies}</span>
              </div>
            )}
          </div>

          {(advisoryEmergencies.length > 0 || advisoryDeteriorating.length > 0) && (
            <div className="qd-advisory">
              {advisoryEmergencies.length > 0 && (
                <span><Siren size={13} /> IMMEDIATE reported: {advisoryEmergencies.map((p) => p.id).join(", ")}</span>
              )}
              {advisoryDeteriorating.length > 0 && (
                <span><AlertTriangle size={13} /> worsening soon: {advisoryDeteriorating.map((p) => `${p.id} (${p.deteriorationCountdown})`).join(", ")}</span>
              )}
            </div>
          )}

          <div className="qd-queue">
            {sortedQueue.length === 0 && !state.done && (
              <div className="qd-empty">Queue is empty. Nothing waiting right now — advance the shift.</div>
            )}
            {sortedQueue.map((p) => {
              const servable = canServe(p, state);
              let reason = "";
              if (!servable) {
                const need = p.requiresSpecialist ? 2 : 1;
                reason = p.requiresIcu && state.availableIcu < 1
                  ? "No ICU beds available"
                  : `Needs ${need} doctors, ${state.availableDoctors} available`;
              }
              return <PatientCard key={p.id} p={p} servable={servable} reason={reason} onServe={handleServe} />;
            })}
          </div>

          <div className="qd-actions">
            {!state.done ? (
              <button className="qd-btn qd-btn--wait" onClick={handleWait}>
                <Pause size={15} /> Wait one step
              </button>
            ) : (
              <button className="qd-btn qd-btn--primary" onClick={finalize}>
                <CheckCircle2 size={15} /> Finalize episode
              </button>
            )}
            <div className="qd-served-count">{state.served.length} / {task.arrivals.length} served</div>
          </div>

          <div className="qd-log" ref={logRef}>
            {state.events.map((e, i) => (
              <div key={i} className={`qd-log-row qd-log--${e.tone}`}>
                <span className="qd-log-step">t{e.step}</span>{e.text}
              </div>
            ))}
          </div>
        </div>
      )}
    </Shell>
  );
}

function Shell({ children, theme }) {
  return (
    <div className="qd-root" data-theme={theme}>
      <style>{CSS}</style>
      {children}
    </div>
  );
}

function ThemeToggle({ theme, setTheme }) {
  const isDark = theme === "dark";
  return (
    <button
      className="qd-theme-toggle"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      aria-label={isDark ? "Switch to ward mode (light)" : "Switch to monitor mode (dark)"}
      title={isDark ? "Ward mode" : "Monitor mode"}
    >
      <span className={`qd-theme-icon ${!isDark ? "qd-theme-icon--active" : ""}`}><Sun size={13} /></span>
      <span className="qd-theme-track"><span className={`qd-theme-knob ${isDark ? "qd-theme-knob--dark" : ""}`} /></span>
      <span className={`qd-theme-icon ${isDark ? "qd-theme-icon--active" : ""}`}><Moon size={13} /></span>
    </button>
  );
}

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

.qd-root[data-theme="dark"] {
  --bg: #0c1117;
  --panel: #131a23;
  --panel-2: #1a2330;
  --border: #263140;
  --text: #e7edf3;
  --muted: #8393a6;
  --accent: #8b7cf6;
  --accent-dim: #4a4272;
  --accent-ink: #0c1117;
  --shadow-strength: 0;
  --bg-pattern: none;
  --danger-ink: #f2a5a7;
}
.qd-root[data-theme="light"] {
  --bg: #eef3f6;
  --panel: #ffffff;
  --panel-2: #f3f7f9;
  --border: #d7e1e8;
  --text: #0f1b24;
  --muted: #5c6d7a;
  --accent: #0d8a83;
  --accent-dim: #a9d9d4;
  --accent-ink: #ffffff;
  --shadow-strength: 0.1;
  --bg-pattern: radial-gradient(circle, #c9d8de 1px, transparent 1px);
  --danger-ink: #b3261e;
}
.qd-root {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  background-color: var(--bg);
  background-image: var(--bg-pattern);
  background-size: 18px 18px;
  color: var(--text);
  border-radius: 16px;
  padding: 28px;
  max-width: 900px;
  margin: 0 auto;
  box-sizing: border-box;
  transition: background-color .2s, color .2s;
}
.qd-root * { box-sizing: border-box; }
.qd-mono { font-family: 'IBM Plex Mono', monospace; }

.qd-theme-toggle {
  display: flex; align-items: center; gap: 7px; margin: 0 0 18px auto; width: fit-content;
  background: var(--panel); border: 1px solid var(--border); border-radius: 999px; padding: 5px 10px;
  cursor: pointer; color: var(--muted); font-family: inherit;
}
.qd-theme-toggle:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.qd-theme-icon { display: flex; opacity: 0.4; transition: opacity .15s, color .15s; }
.qd-theme-icon--active { opacity: 1; color: var(--accent); }
.qd-theme-track { width: 30px; height: 16px; background: var(--panel-2); border: 1px solid var(--border); border-radius: 999px; position: relative; }
.qd-theme-knob { position: absolute; top: 1px; left: 1px; width: 12px; height: 12px; border-radius: 50%; background: var(--accent); transition: transform .18s ease; }
.qd-theme-knob--dark { transform: translateX(14px); }

/* ── task picker ── */
.qd-hero { display: flex; flex-direction: column; gap: 4px; margin-bottom: 28px; }
.qd-pulse { width: 100%; height: 34px; overflow: visible; }
.qd-pulse polyline { fill: none; stroke: var(--accent); stroke-width: 1.6; stroke-linejoin: round; stroke-linecap: round; opacity: 0.85; }
.qd-eyebrow { display: flex; align-items: center; gap: 6px; color: var(--accent); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600; margin-top: 10px; }
.qd-hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 34px; font-weight: 700; margin: 4px 0 8px; letter-spacing: -0.01em; }
.qd-hero p { color: var(--muted); line-height: 1.55; margin: 0; max-width: 62ch; font-size: 14.5px; }

.qd-task-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
.qd-task-card {
  text-align: left; background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 18px; cursor: pointer; color: var(--text); font-family: inherit; display: flex; flex-direction: column; gap: 8px;
  transition: border-color .15s, transform .15s, box-shadow .15s;
  box-shadow: 0 1px 2px rgba(15,27,36,var(--shadow-strength));
}
.qd-task-card:hover { border-color: var(--accent-dim); transform: translateY(-2px); box-shadow: 0 6px 16px -6px rgba(15,27,36,calc(var(--shadow-strength) + 0.08)); }
.qd-task-card:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.qd-task-diff { font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); width: fit-content; padding: 2px 8px; border: 1px solid var(--accent-dim); border-radius: 20px; }
.qd-task-diff[data-diff="Hard"] { color: #e5484d; border-color: #5a2a2c; }
.qd-task-diff[data-diff="Medium"] { color: #e0b039; border-color: #5a4a2a; }
.qd-task-card h3 { font-family: 'Space Grotesk', sans-serif; font-size: 17px; margin: 2px 0 0; }
.qd-task-card p { color: var(--muted); font-size: 13px; line-height: 1.5; margin: 0; flex: 1; }
.qd-task-stats { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11.5px; color: var(--muted); font-family: 'IBM Plex Mono', monospace; }
.qd-task-stats span { display: flex; align-items: center; gap: 4px; }
.qd-task-start { display: flex; align-items: center; gap: 2px; font-size: 12.5px; color: var(--accent); font-weight: 600; margin-top: 4px; }
.qd-footnote { display: flex; align-items: flex-start; gap: 6px; color: var(--muted); font-size: 11.5px; line-height: 1.5; margin-top: 22px; padding-top: 16px; border-top: 1px solid var(--border); }

/* ── game ── */
.qd-topbar { display: flex; flex-wrap: wrap; gap: 18px; align-items: center; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 12px 16px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(15,27,36,var(--shadow-strength)); }
.qd-topbar-item { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 12.5px; }
.qd-topbar-item--warn { color: #e5484d; }
.qd-label { text-transform: uppercase; letter-spacing: 0.06em; font-size: 10px; color: var(--muted); }
.qd-value { color: var(--text); font-weight: 600; font-size: 13px; }
.qd-up { color: #3ecf8e; } .qd-down { color: #e5484d; }

.qd-advisory { display: flex; flex-wrap: wrap; gap: 16px; background: rgba(229,72,77,0.08); border: 1px solid rgba(229,72,77,0.25); color: var(--danger-ink); padding: 8px 14px; border-radius: 10px; font-size: 12.5px; margin-bottom: 14px; }
.qd-advisory span { display: flex; align-items: center; gap: 6px; }

.qd-queue { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px; margin-bottom: 16px; }
.qd-empty { grid-column: 1/-1; color: var(--muted); text-align: center; padding: 30px 10px; border: 1px dashed var(--border); border-radius: 12px; font-size: 13px; }

.qd-card {
  position: relative; text-align: left; border-radius: 12px; padding: 12px; cursor: pointer; font-family: inherit;
  background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--sev); color: var(--text);
  display: flex; flex-direction: column; gap: 6px; transition: transform .12s, box-shadow .12s;
  box-shadow: 0 1px 2px rgba(15,27,36,var(--shadow-strength));
}
.qd-card--live:hover { transform: translateY(-2px); box-shadow: 0 0 0 1px var(--sev), 0 8px 18px -8px var(--glow), 0 2px 8px rgba(15,27,36,var(--shadow-strength,0.9)); }
.qd-card--live:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.qd-card--blocked { opacity: 0.55; cursor: not-allowed; }
.qd-card-top { display: flex; justify-content: space-between; align-items: center; }
.qd-tag { font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: 0.06em; color: var(--sev); }
.qd-id { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--muted); }
.qd-card-name { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 13.5px; }
.qd-card-meta { display: flex; flex-wrap: wrap; gap: 8px; font-size: 11px; color: var(--muted); align-items: center; }
.qd-card-meta span { display: flex; align-items: center; gap: 3px; }
.qd-chip { background: var(--panel-2); border: 1px solid var(--border); border-radius: 6px; padding: 1px 6px; font-size: 10px; font-family: 'IBM Plex Mono', monospace; }
.qd-warn { display: flex; align-items: center; gap: 4px; font-size: 10.5px; color: #f0952d; }
.qd-blocked-reason { font-size: 10.5px; color: var(--muted); font-style: italic; }
.qd-serve-hint { display: flex; align-items: center; font-size: 11px; color: var(--accent); font-weight: 600; margin-top: 2px; }

.qd-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
.qd-served-count { color: var(--muted); font-family: 'IBM Plex Mono', monospace; font-size: 12px; }

.qd-btn {
  display: inline-flex; align-items: center; gap: 7px; background: var(--panel-2); color: var(--text);
  border: 1px solid var(--border); border-radius: 10px; padding: 10px 16px; font-size: 13.5px; font-weight: 600;
  cursor: pointer; font-family: inherit; transition: border-color .15s, background .15s;
}
.qd-btn:hover { border-color: var(--accent-dim); }
.qd-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.qd-btn--wait { color: var(--muted); }
.qd-btn--primary { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
.qd-btn--primary:hover { filter: brightness(1.1); }

.qd-log { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; max-height: 170px; overflow-y: auto; display: flex; flex-direction: column-reverse; gap: 3px; box-shadow: 0 1px 2px rgba(15,27,36,var(--shadow-strength)); }
.qd-log-row { font-size: 12px; color: var(--muted); line-height: 1.5; display: flex; gap: 8px; }
.qd-log-step { font-family: 'IBM Plex Mono', monospace; color: var(--accent); flex-shrink: 0; }
.qd-log--good { color: #3ecf8e; } .qd-log--good .qd-log-step { color: #3ecf8e; }
.qd-log--bad { color: #e5484d; } .qd-log--bad .qd-log-step { color: #e5484d; }
.qd-log--alarm { color: #f0952d; } .qd-log--alarm .qd-log-step { color: #f0952d; }
.qd-log--blocked { color: #6a7686; font-style: italic; }

/* ── result ── */
.qd-result { text-align: center; padding: 10px 0 0; }
.qd-result-score { position: relative; width: 140px; height: 140px; margin: 0 auto 10px; }
.qd-result-score svg { width: 100%; height: 100%; transform: rotate(-90deg); }
.qd-ring-bg { fill: none; stroke: var(--panel-2); stroke-width: 9; }
.qd-ring-fg { fill: none; stroke: var(--accent); stroke-width: 9; stroke-linecap: round; }
.qd-result-score-num { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.qd-big { font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; }
.qd-small { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.qd-result h2 { font-family: 'Space Grotesk', sans-serif; font-size: 20px; margin: 4px 0 6px; }
.qd-result-compare { color: var(--muted); font-size: 13.5px; max-width: 46ch; margin: 0 auto 20px; line-height: 1.5; }
.qd-breakdown { text-align: left; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(15,27,36,var(--shadow-strength)); }
.qd-breakdown-row { display: flex; justify-content: space-between; font-size: 13px; padding: 6px 0; border-bottom: 1px solid var(--border); font-family: 'IBM Plex Mono', monospace; }
.qd-breakdown-row:last-child { border-bottom: none; }
.qd-result-actions { display: flex; gap: 10px; justify-content: center; }

@media (max-width: 560px) {
  .qd-root { padding: 18px; border-radius: 0; }
  .qd-hero h1 { font-size: 26px; }
}
`;
