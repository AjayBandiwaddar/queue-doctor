"""
Inference Script — Queue Doctor
=================================
MANDATORY env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=queue_doctor model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

The agent makes genuine sequential decisions each step:
  - Reads current queue state (patients, severities, wait times, resources)
  - Decides which patient to serve (or whether to wait)
  - Executes action — environment state changes in response
  - Repeats until episode ends
"""

import json
import os
import sys

from openai import OpenAI

from client import QueueDoctorEnv

# ── Configuration ──────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
ENV_NAME     = "queue_doctor"
MAX_STEPS_GUARD = 40  # Safety cap per task

TASK_IDS = ["task_1_easy", "task_2_medium", "task_3_hard"]

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── Agent System Prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI hospital triage coordinator managing an emergency department queue.

TRIAGE LEVELS (severity 1 = most critical):
  1 = IMMEDIATE    — Life-threatening. Serve NOW. Every step of delay is catastrophic.
  2 = VERY_URGENT  — Serious. Serve within 1-2 steps. Reward decays quickly.
  3 = URGENT       — Significant. Serve within ~6 steps.
  4 = LESS_URGENT  — Minor-moderate. Serve when possible.
  5 = NON_URGENT   — Minor. Serve if time allows.

DECISION RULES:
1. ALWAYS prioritize severity-1 patients first — never leave an IMMEDIATE patient waiting.
2. Among equal severity, prefer the patient with the longest wait_time (fairness).
3. Watch deterioration_countdown — if a patient shows this field, they will worsen
   to a higher severity if not served before countdown reaches 0. Prioritize them.
4. For Task 3 with ICU beds: if ICU beds are nearly full and mass casualty is approaching,
   consider serving non-ICU patients first to conserve capacity.
5. Use wait() ONLY if the queue is empty. Never wait with patients waiting.

OUTPUT FORMAT — respond with ONLY a valid JSON object, no other text:
{"action": "serve_patient", "patient_id": "P001", "reasoning": "Emergency patient — immediate action required"}
OR
{"action": "wait", "reasoning": "Queue is empty"}"""


def call_llm(queue_state: dict) -> dict:
    """Call LLM with current queue state. Returns parsed action decision."""
    queue = queue_state.get("queue", [])

    # Build queue summary for LLM
    lines = []
    for p in queue:
        line = (
            f"  {p['patient_id']}: severity={p['severity']} ({p.get('severity_name','?')}), "
            f"waited={p['wait_time']} steps"
        )
        if p.get("deterioration_countdown"):
            line += f" ⚠️ DETERIORATES IN {p['deterioration_countdown']} STEPS"
        if p.get("requires_icu"):
            line += " [needs ICU]"
        if p.get("requires_specialist"):
            line += " [needs 2 doctors]"
        lines.append(line)

    icu_line = ""
    if queue_state.get("available_icu_beds") is not None:
        icu_line = f"\nAvailable ICU beds: {queue_state['available_icu_beds']}/{queue_state.get('total_icu_beds', '?')}"

    prompt = f"""Step {queue_state['step']}/{queue_state['max_steps']} | Steps remaining: {queue_state['steps_remaining']}
Available doctors: {queue_state['available_doctors']}{icu_line}
Patients served: {queue_state['patients_served']} | Missed emergencies: {queue_state['missed_emergencies']}
Triage advisory: {queue_state.get('triage_advisory', 'N/A')}

CURRENT QUEUE ({queue_state['queue_length']} patients):
{chr(10).join(lines) if lines else '  [Queue is empty]'}

Decide your next action."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=120,
    )
    raw = response.choices[0].message.content.strip()

    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        # Fallback: serve highest-priority patient or wait
        if queue:
            return {"action": "serve_patient", "patient_id": queue[0]["patient_id"],
                    "reasoning": "fallback — highest priority patient"}
        return {"action": "wait", "reasoning": "fallback — empty queue"}


# ── Episode Runner ─────────────────────────────────────────────────────────

def run_task(env, task_id: str) -> dict:
    """Run one complete task episode. Returns result dict."""
    step_num  = 0
    rewards   = []
    success   = False
    error_msg = None
    task_name = task_id

    try:
        # ── Start task ────────────────────────────────────────────────────
        step_num += 1
        raw = env.call_tool("start_task", task_id=task_id)
        task_data = json.loads(raw) if isinstance(raw, str) else raw
        task_name = task_data.get("task_name", task_id)
        max_steps = task_data.get("max_steps", 20)

        print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)
        print(
            f"[STEP] step={step_num} action=start_task('{task_id}') "
            f"reward=0.00 done=false error=null",
            flush=True,
        )

        # ── Main episode loop ─────────────────────────────────────────────
        done = False
        episode_step = 0

        while not done and episode_step < MAX_STEPS_GUARD:
            # Get observable state (no time advance)
            step_num += 1
            raw_state = env.call_tool("get_queue_state")
            queue_state = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
            done = queue_state.get("done", False)

            if done:
                print(
                    f"[STEP] step={step_num} action=get_queue_state() "
                    f"reward=0.00 done=true error=null",
                    flush=True,
                )
                break

            # LLM decides action
            decision   = call_llm(queue_state)
            action     = decision.get("action", "wait")
            patient_id = decision.get("patient_id", "")

            # Execute action (advances time by 1 step)
            step_num += 1
            if action == "serve_patient" and patient_id:
                raw_result  = env.call_tool("serve_patient", patient_id=patient_id)
                action_str  = f"serve_patient('{patient_id}')"
            else:
                raw_result  = env.call_tool("wait")
                action_str  = "wait()"

            result       = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
            step_reward  = float(result.get("step_reward", 0.0))
            done         = result.get("done", False)
            err          = result.get("error")

            rewards.append(step_reward)
            episode_step += 1

            done_str  = "true" if done else "false"
            error_str = err if err else "null"
            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={step_reward:.2f} done={done_str} error={error_str}",
                flush=True,
            )

        # ── Finalize episode ──────────────────────────────────────────────
        step_num += 1
        raw_final    = env.call_tool("finalize_episode")
        final        = json.loads(raw_final) if isinstance(raw_final, str) else raw_final
        final_score  = float(final.get("score", 0.0))
        rewards.append(final_score)
        success      = final_score >= 0.35

        print(
            f"[STEP] step={step_num} action=finalize_episode() "
            f"reward={final_score:.2f} done=true error=null",
            flush=True,
        )

    except Exception as e:
        error_msg = str(e)
        rewards.append(0.0)
        print(
            f"[STEP] step={step_num + 1} action=error "
            f"reward=0.00 done=true error={error_msg}",
            flush=True,
        )

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={step_num} rewards={rewards_str}", flush=True)

    return {
        "task_id":   task_id,
        "task_name": task_name,
        "score":     rewards[-1] if rewards else 0.0,
        "steps":     step_num,
        "success":   success,
    }


# ── Entry Point ────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}", flush=True)
    print(f"Queue Doctor — Inference", flush=True)
    print(f"Model : {MODEL_NAME}", flush=True)
    print(f"Server: {ENV_URL}", flush=True)
    print(f"{'='*60}\n", flush=True)

    all_results = []

    with QueueDoctorEnv(base_url=ENV_URL).sync() as env:
        env.reset()
        for task_id in TASK_IDS:
            print(f"\n--- Running {task_id} ---", flush=True)
            result = run_task(env, task_id)
            all_results.append(result)

    print(f"\n{'='*60}", flush=True)
    print("FINAL SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    total = 0.0
    for r in all_results:
        print(f"  {r['task_name']:<40} {r['score']:.4f}", flush=True)
        total += r["score"]
    avg = total / len(all_results) if all_results else 0.0
    print(f"\n  Average score: {avg:.4f}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()