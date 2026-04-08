"""
Inference Script - Queue Doctor
=================================
MANDATORY env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=queue_doctor model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Step counting:
    Only meaningful actions count as steps:
      - start_task       -> step 1
      - serve_patient()  -> step N
      - wait()           -> step N
      - finalize_episode -> final step
    get_queue_state() calls are observations and do NOT count as steps.

Notes:
    - A random seed is passed to start_task() each run to produce genuine
      episode variance across runs (required by Phase 2 score variance check).
    - Triage advisory is excluded from the LLM prompt -- agent must reason
      from raw queue data, not pre-computed hints.
    - Resource errors do not advance time. Agent automatically retries with
      the highest-priority servable patient.
    - Agent never waits when servable patients are present.
"""

import json
import os
import random
import time

from openai import OpenAI
from client import QueueDoctorEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL    = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY         = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME      = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL         = os.getenv("ENV_URL", "http://localhost:7860")
ENV_NAME        = "queue_doctor"
MAX_STEPS_GUARD = 60

TASK_IDS = ["task_1_easy", "task_2_medium", "task_3_hard"]

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Agent System Prompt
# ---------------------------------------------------------------------------
# Triage advisory intentionally excluded -- agent must reason from raw data.

SYSTEM_PROMPT = """You are an AI hospital triage coordinator managing an emergency department queue.

MANCHESTER TRIAGE SYSTEM severity levels (1 = most critical):
  1 = IMMEDIATE    -- Life-threatening. Treat NOW. Every step of delay is catastrophic.
  2 = VERY_URGENT  -- Serious. Treat within 1-2 steps. Reward decays at 0.125/step.
  3 = URGENT       -- Significant. Treat within ~6 steps.
  4 = LESS_URGENT  -- Minor-moderate. Treat when possible.
  5 = NON_URGENT   -- Minor. Treat if time allows.

DECISION RULES -- follow in strict order:
1. Serve severity-1 patients IMMEDIATELY. No exception.
2. Among equal severity, prefer the patient with the longest wait_time.
3. If a patient shows deterioration_countdown, treat before countdown reaches 0.
4. Check can_serve_now=true before choosing -- if blocked, choose next best.
5. NEVER output wait() if any patient has can_serve_now=true.
   Only use wait() if queue is empty or ALL patients are resource-blocked.

OUTPUT FORMAT -- respond with ONLY a valid JSON object, no other text:
{"action": "serve_patient", "patient_id": "P001", "reasoning": "Severity 1 immediate"}
OR
{"action": "wait", "reasoning": "Queue empty"}"""


def call_llm(queue_state: dict) -> dict:
    """
    Call LLM with current queue state. Returns parsed action decision.
    Falls back to greedy if API fails (rate limit, credits, parse error).
    Triage advisory excluded -- agent must reason from raw patient data.
    """
    queue = queue_state.get("queue", [])

    def greedy_fallback():
        servable = [p for p in queue if p.get("can_serve_now", True)]
        if servable:
            best = sorted(servable, key=lambda p: (p["severity"], -p["wait_time"]))[0]
            return {"action": "serve_patient", "patient_id": best["patient_id"],
                    "reasoning": "greedy fallback"}
        return {"action": "wait", "reasoning": "no servable patients"}

    if not queue:
        return {"action": "wait", "reasoning": "empty queue"}

    try:
        lines = []
        for p in queue:
            servable = "CAN SERVE" if p.get("can_serve_now", True) else \
                f"BLOCKED ({p.get('cannot_serve_reason', 'resource unavailable')})"
            line = (
                f"  {p['patient_id']}: severity={p['severity']} "
                f"({p.get('severity_name', '?')})"
                f", waited={p['wait_time']} steps, {servable}"
            )
            if p.get("deterioration_countdown"):
                line += f" DETERIORATES IN {p['deterioration_countdown']} STEPS"
            if p.get("requires_icu"):
                line += " [needs ICU]"
            if p.get("requires_specialist"):
                line += " [needs 2 doctors]"
            lines.append(line)

        resource_lines = [f"Available doctors: {queue_state['available_doctors']}"]
        if queue_state.get("available_icu_beds") is not None:
            resource_lines.append(
                f"Available ICU beds: {queue_state['available_icu_beds']}"
                f"/{queue_state.get('total_icu_beds', '?')}"
            )

        prompt = (
            f"Step {queue_state['step']}/{queue_state['max_steps']} "
            f"| Steps remaining: {queue_state['steps_remaining']}\n"
            f"{' | '.join(resource_lines)}\n"
            f"Patients served: {queue_state['patients_served']} "
            f"| Missed emergencies: {queue_state['missed_emergencies']}\n\n"
            f"CURRENT QUEUE ({queue_state['queue_length']} patients):\n"
            f"{chr(10).join(lines)}\n\n"
            f"Choose your action. NEVER wait if any patient has can_serve_now=true."
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        time.sleep(2)

        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        return json.loads(raw)

    except Exception:
        time.sleep(1)
        return greedy_fallback()


def _best_servable(queue: list) -> str | None:
    servable = [p for p in queue if p.get("can_serve_now", True)]
    return servable[0]["patient_id"] if servable else None


# ---------------------------------------------------------------------------
# Episode Runner
# ---------------------------------------------------------------------------

def run_task(env, task_id: str, episode_seed: int) -> dict:
    """
    Run one complete task episode with correct step counting.
    episode_seed: passed to start_task for reproducible variance across runs.
    """
    step_num    = 0
    rewards     = []
    final_score = 0.0
    task_name   = task_id

    try:
        # Step 1: start task with seed for episode variance
        step_num += 1
        raw = env.call_tool("start_task", task_id=task_id, seed=episode_seed)
        task_data = json.loads(raw) if isinstance(raw, str) else raw
        task_name = task_data.get("task_name", task_id)

        print(
            f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}",
            flush=True,
        )
        print(
            f"[STEP] step={step_num} action=start_task('{task_id}') "
            f"reward=0.00 done=false error=null",
            flush=True,
        )

        done         = False
        episode_step = 0

        while not done and episode_step < MAX_STEPS_GUARD:

            # Observe -- NOT a step
            raw_state   = env.call_tool("get_queue_state")
            queue_state = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
            done        = queue_state.get("done", False)
            if done:
                break

            queue = queue_state.get("queue", [])

            # LLM decision
            decision   = call_llm(queue_state)
            action     = decision.get("action", "wait")
            patient_id = decision.get("patient_id", "")

            # Safety override: never wait when servable patients exist
            if action == "wait" and queue:
                best = _best_servable(queue)
                if best:
                    action     = "serve_patient"
                    patient_id = best

            # Execute action -- this IS a step
            step_num     += 1
            episode_step += 1

            if action == "serve_patient" and patient_id:
                raw_result = env.call_tool("serve_patient", patient_id=patient_id)
                action_str = f"serve_patient('{patient_id}')"
            else:
                raw_result = env.call_tool("wait")
                action_str = "wait()"

            result      = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
            step_reward = float(result.get("step_reward", 0.0))
            done        = result.get("done", False)
            events      = result.get("events", [])

            # Resource error retry (time did NOT advance)
            resource_error = any(
                "Cannot" in str(e) or "no ICU" in str(e).lower()
                for e in events
            )
            if resource_error and action == "serve_patient":
                step_num     -= 1
                episode_step -= 1
                best = _best_servable(queue)
                if best and best != patient_id:
                    raw_result  = env.call_tool("serve_patient", patient_id=best)
                    action_str  = f"serve_patient('{best}')"
                else:
                    raw_result  = env.call_tool("wait")
                    action_str  = "wait()"
                result      = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
                step_reward = float(result.get("step_reward", 0.0))
                done        = result.get("done", False)
                step_num     += 1
                episode_step += 1

            rewards.append(step_reward)
            done_str = "true" if done else "false"
            print(
                f"[STEP] step={step_num} action={action_str} "
                f"reward={step_reward:.2f} done={done_str} error=null",
                flush=True,
            )

        # Final step: finalize
        step_num += 1
        raw_final   = env.call_tool("finalize_episode")
        final       = json.loads(raw_final) if isinstance(raw_final, str) else raw_final
        final_score = float(final.get("score", 0.0))
        rewards.append(final_score)

        print(
            f"[STEP] step={step_num} action=finalize_episode() "
            f"reward={final_score:.2f} done=true error=null",
            flush=True,
        )

    except Exception as exc:
        rewards.append(0.0)
        print(
            f"[STEP] step={step_num + 1} action=error "
            f"reward=0.00 done=true error={str(exc)}",
            flush=True,
        )

    success_str = "true" if final_score >= 0.35 else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={step_num} "
        f"score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return {
        "task_id":   task_id,
        "task_name": task_name,
        "score":     final_score,
        "steps":     step_num,
        "success":   final_score >= 0.35,
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    # Generate a single random seed for this run -- all three tasks use it.
    # This produces genuine episode variance across runs while keeping
    # intra-run consistency (same perturbations for all tasks in one run).
    episode_seed = random.randint(1, 9999)

    print(f"\n{'='*60}", flush=True)
    print(f"Queue Doctor - Inference", flush=True)
    print(f"Model  : {MODEL_NAME}", flush=True)
    print(f"Server : {ENV_URL}", flush=True)
    print(f"Seed   : {episode_seed}", flush=True)
    print(f"{'='*60}\n", flush=True)

    all_results = []

    with QueueDoctorEnv(base_url=ENV_URL).sync() as env:
        env.reset()
        for task_id in TASK_IDS:
            print(f"\n--- Running {task_id} ---", flush=True)
            result = run_task(env, task_id, episode_seed)
            all_results.append(result)

    print(f"\n{'='*60}", flush=True)
    print("FINAL SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    total = 0.0
    for r in all_results:
        print(f"  {r['task_name']:<42} {r['score']:.4f}", flush=True)
        total += r["score"]
    avg = total / len(all_results) if all_results else 0.0
    print(f"\n  Average score: {avg:.4f}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()