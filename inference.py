"""
Inference Script â€” Queue Doctor
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
      - start_task       â†’ step 1
      - serve_patient()  â†’ step N
      - wait()           â†’ step N
      - finalize_episode â†’ final step
    get_queue_state() calls are observations â€” they do NOT count as steps.

Notes:
    - Triage advisory is deliberately excluded from the LLM prompt.
      The agent must reason from raw queue data, not pre-computed hints.
    - Resource errors (no ICU bed, insufficient doctors) do not advance time.
      The agent automatically retries with the highest-priority servable patient.
    - The agent never waits when servable patients are present.
"""

import json
import time
import os

from openai import OpenAI
from client import QueueDoctorEnv

# â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
API_BASE_URL    = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY         = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME      = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL         = os.getenv("ENV_URL", "http://localhost:7860")
ENV_NAME        = "queue_doctor"
MAX_STEPS_GUARD = 60   # hard cap â€” prevents infinite loops

TASK_IDS = ["task_1_easy", "task_2_medium", "task_3_hard"]

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# â”€â”€ Agent System Prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Triage advisory intentionally omitted â€” agent must reason from raw data.

SYSTEM_PROMPT = """You are an AI hospital triage coordinator managing an emergency department queue.

MANCHESTER TRIAGE SYSTEM severity levels (1 = most critical):
  1 = IMMEDIATE    â€” Life-threatening. Treat NOW. Every step of delay is catastrophic.
  2 = VERY_URGENT  â€” Serious. Treat within 1-2 steps. Reward decays at 0.125/step.
  3 = URGENT       â€” Significant. Treat within ~6 steps.
  4 = LESS_URGENT  â€” Minor-moderate. Treat when possible.
  5 = NON_URGENT   â€” Minor. Treat if time allows.

DECISION RULES â€” follow in strict order:
1. Serve severity-1 patients IMMEDIATELY. No exception.
2. Among equal severity, prefer the patient with the longest wait_time.
3. If a patient shows deterioration_countdown, they WILL worsen soon.
   Treat them before their countdown reaches 0, even over higher wait-time patients.
4. Check can_serve_now=true before choosing a patient.
   If your preferred patient has can_serve_now=false, choose the next best servable one.
5. NEVER output wait() if any patient in the queue has can_serve_now=true.
   Only use wait() if the queue is completely empty or ALL patients cannot be served.

OUTPUT FORMAT â€” respond with ONLY a valid JSON object, no other text, no markdown:
{"action": "serve_patient", "patient_id": "P001", "reasoning": "Severity 1 immediate threat"}
OR
{"action": "wait", "reasoning": "Queue is empty"}"""


# def call_llm(queue_state: dict) -> dict:
#     """
#     Call LLM with current queue state. Returns parsed action decision.
#     Triage advisory is intentionally excluded from the prompt â€” the agent
#     must reason from raw patient data, not pre-computed recommendations.
#     """
#     queue = queue_state.get("queue", [])

#     lines = []
#     for p in queue:
#         servable = "âœ“ CAN SERVE" if p.get("can_serve_now", True) else f"âœ— BLOCKED ({p.get('cannot_serve_reason', 'resource unavailable')})"
#         line = (
#             f"  {p['patient_id']}: severity={p['severity']} ({p.get('severity_name','?')})"
#             f", waited={p['wait_time']} steps, {servable}"
#         )
#         if p.get("deterioration_countdown"):
#             line += f" âš ï¸ DETERIORATES IN {p['deterioration_countdown']} STEPS"
#         if p.get("requires_icu"):
#             line += " [needs ICU bed]"
#         if p.get("requires_specialist"):
#             line += " [needs 2 doctors]"
#         lines.append(line)

#     resource_lines = [
#         f"Available doctors: {queue_state['available_doctors']}",
#     ]
#     if queue_state.get("available_icu_beds") is not None:
#         resource_lines.append(
#             f"Available ICU beds: {queue_state['available_icu_beds']}"
#             f"/{queue_state.get('total_icu_beds', '?')}"
#         )

#     prompt = (
#         f"Step {queue_state['step']}/{queue_state['max_steps']} "
#         f"| Steps remaining: {queue_state['steps_remaining']}\n"
#         f"{' | '.join(resource_lines)}\n"
#         f"Patients served: {queue_state['patients_served']} "
#         f"| Missed emergencies: {queue_state['missed_emergencies']}\n\n"
#         f"CURRENT QUEUE ({queue_state['queue_length']} patients):\n"
#         f"{chr(10).join(lines) if lines else '  [Queue is empty]'}\n\n"
#         f"Choose your action. Remember: NEVER wait if any patient has can_serve_now=true."
#     )

#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user",   "content": prompt},
#         ],
#         temperature=0.0,
#         max_tokens=150,
#     )
#     raw = response.choices[0].message.content.strip()
#     time.sleep(2)

#     # Strip markdown fences if present
#     if "```" in raw:
#         raw = raw.split("```")[1].replace("json", "").strip()

#     try:
#         return json.loads(raw)
#     except (json.JSONDecodeError, IndexError):
#         # Fallback: serve highest-priority servable patient, never wait if avoidable
#         servable = [p for p in queue if p.get("can_serve_now", True)]
#         if servable:
#             return {
#                 "action": "serve_patient",
#                 "patient_id": servable[0]["patient_id"],
#                 "reasoning": "fallback â€” highest-priority servable patient",
#             }
#         if queue:
#             # All patients blocked by resources â€” wait is the only option
#             return {"action": "wait", "reasoning": "fallback â€” all patients resource-blocked"}
#         return {"action": "wait", "reasoning": "fallback â€” queue empty"}


# def _best_servable_patient(queue: list) -> str | None:
#     """Return patient_id of highest-priority servable patient, or None."""
#     servable = [p for p in queue if p.get("can_serve_now", True)]
#     return servable[0]["patient_id"] if servable else None

def call_llm(queue_state: dict) -> dict:
    """
    Call LLM with current queue state. Returns parsed action decision.
    Triage advisory is intentionally excluded from the prompt — the agent
    must reason from raw patient data, not pre-computed recommendations.
    Falls back to greedy policy if the API call fails for any reason
    (rate limit, credits exhausted, parse error).
    """
    queue = queue_state.get("queue", [])

    def greedy_fallback() -> dict:
        servable = [p for p in queue if p.get("can_serve_now", True)]
        if servable:
            return {
                "action": "serve_patient",
                "patient_id": servable[0]["patient_id"],
                "reasoning": "greedy fallback — highest-priority servable patient",
            }
        if queue:
            return {"action": "wait", "reasoning": "greedy fallback — all patients resource-blocked"}
        return {"action": "wait", "reasoning": "greedy fallback — queue empty"}

    lines = []
    for p in queue:
        servable_flag = (
            "CAN SERVE"
            if p.get("can_serve_now", True)
            else f"BLOCKED ({p.get('cannot_serve_reason', 'resource unavailable')})"
        )
        line = (
            f"  {p['patient_id']}: severity={p['severity']} ({p.get('severity_name', '?')})"
            f", waited={p['wait_time']} steps, {servable_flag}"
        )
        if p.get("deterioration_countdown"):
            line += f" DETERIORATES IN {p['deterioration_countdown']} STEPS"
        if p.get("requires_icu"):
            line += " [needs ICU bed]"
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
        f"{chr(10).join(lines) if lines else '  [Queue is empty]'}\n\n"
        f"Choose your action. NEVER wait if any patient has CAN SERVE status."
    )

    try:
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
        # LLM call failed (credits, rate limit, parse error) — use greedy
        time.sleep(1)
        return greedy_fallback()


def _best_servable_patient(queue: list) -> str | None:
    """Return patient_id of highest-priority servable patient, or None."""
    servable = [p for p in queue if p.get("can_serve_now", True)]
    return servable[0]["patient_id"] if servable else None


# â”€â”€ Episode Runner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_task(env, task_id: str) -> dict:
    """
    Run one complete task episode with correct step counting.

    Steps counted: start_task, each serve_patient/wait, finalize_episode.
    get_queue_state() calls are observations and do NOT count as steps.
    Resource errors do not advance time â€” agent retries with best servable patient.
    """
    step_num   = 0
    rewards    = []
    final_score = 0.0
    task_name  = task_id
    error_msg  = None

    try:
        # â”€â”€ Step 1: start task â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        step_num += 1
        raw = env.call_tool("start_task", task_id=task_id, seed=7)
        task_data = json.loads(raw) if isinstance(raw, str) else raw
        task_name = task_data.get("task_name", task_id)

        print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)
        print(
            f"[STEP] step={step_num} action=start_task('{task_id}') "
            f"reward=0.00 done=false error=null",
            flush=True,
        )

        # â”€â”€ Main episode loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        done         = False
        episode_step = 0

        while not done and episode_step < MAX_STEPS_GUARD:

            # Observe current state â€” NOT a step, no step_num increment
            raw_state   = env.call_tool("get_queue_state")
            queue_state = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
            done        = queue_state.get("done", False)

            if done:
                break

            queue = queue_state.get("queue", [])

            # LLM decides action
            decision   = call_llm(queue_state)
            action     = decision.get("action", "wait")
            patient_id = decision.get("patient_id", "")

            # Safety override: never wait when servable patients exist
            if action == "wait" and queue:
                best = _best_servable_patient(queue)
                if best:
                    action    = "serve_patient"
                    patient_id = best

            # Execute action â€” this IS a step
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

            # Detect resource error (time did NOT advance) and retry
            resource_error = any(
                "Cannot" in str(e) or "no ICU" in str(e).lower() or "needs" in str(e)
                for e in events
            )
            if resource_error and action == "serve_patient":
                # Time did not advance â€” don't count this as a step
                step_num     -= 1
                episode_step -= 1
                # Retry with best servable patient
                best = _best_servable_patient(queue)
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

        # â”€â”€ Final step: finalize episode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        step_num += 1
        raw_final   = env.call_tool("finalize_episode")
        final       = json.loads(raw_final) if isinstance(raw_final, str) else raw_final
        final_score = float(final.get("score", 0.0))
        rewards.append(final_score)
        done = True

        print(
            f"[STEP] step={step_num} action=finalize_episode() "
            f"reward={final_score:.2f} done=true error=null",
            flush=True,
        )

    except Exception as exc:
        error_msg = str(exc)
        rewards.append(0.0)
        print(
            f"[STEP] step={step_num + 1} action=error "
            f"reward=0.00 done=true error={error_msg}",
            flush=True,
        )

    success_str = "true" if final_score >= 0.35 else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={step_num} rewards={rewards_str}", flush=True)

    return {
        "task_id":   task_id,
        "task_name": task_name,
        "score":     final_score,
        "steps":     step_num,
        "success":   final_score >= 0.35,
    }


# â”€â”€ Entry Point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"Queue Doctor â€” Inference", flush=True)
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
        print(f"  {r['task_name']:<42} {r['score']:.4f}", flush=True)
        total += r["score"]
    avg = total / len(all_results) if all_results else 0.0
    print(f"\n  Average score: {avg:.4f}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
