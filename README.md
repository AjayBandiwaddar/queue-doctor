---
title: Queue Doctor
emoji: 🏥
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
tags:
  - openenv
  - rl
  - reinforcement-learning
  - healthcare
  - triage
  - agent
---

# 🏥 Queue Doctor — Hospital Emergency Triage RL Environment

> A genuine multi-step reinforcement learning environment for emergency department triage.
> Built for the Meta PyTorch OpenEnv Hackathon.

---

## What Is This?

Queue Doctor places an AI agent in charge of a hospital emergency department. At every step, the agent observes the queue of waiting patients and decides who to treat next. The environment state changes meaningfully after every decision — new patients arrive on a fixed schedule, wait times accumulate, patient conditions deteriorate — making this a true **Markov Decision Process**.

**A better policy produces measurably better outcomes.** A random agent scores ~0.40 on Task 1. An optimal agent scores ~0.98. That gap is exactly what RL training is designed to close.

---

## Why This Is Genuine RL

Unlike document classification or single-step benchmarks, every action in Queue Doctor has consequences that persist into future steps:

- Serving Patient A now means Patient B's wait time increases by 1 step, reducing the reward available for serving them later
- Ignoring a deteriorating patient means their severity worsens at step N+3, making the queue harder to manage going forward
- Conserving ICU beds in steps 1–14 (Task 3) is the only way to survive the mass casualty surge at step 15

The agent must discover non-obvious strategies: serve deteriorating patients before higher-severity stable ones, reserve specialist capacity before a predicted surge, and balance urgency with fairness across 20–30 step episodes.

---

## Tasks

### Task 1 — Basic Triage `easy`

| Property | Value |
|---|---|
| Patients | 6 (all present at step 0) |
| Doctors | 1 |
| Steps | 10 |
| Grader | Normalized cumulative reward |

**Challenge:** Discover the optimal service order. The reward function follows Manchester Triage System decay rates — serving a severity-1 patient at wait=2 already loses 80% of the available reward. The agent must internalize that every step of delay is costly, and that the penalty compounds differently per severity level.

**Baseline scores (meta-llama/Llama-3.1-8B-Instruct)**

| Seed | Score |
|---|---|
| 42 | 0.9241 |
| 7 | 0.7903 |
| 99 | 0.9955 |
| 5015 | 0.8149 |

---

### Task 2 — Dynamic Queue Management `medium`

| Property | Value |
|---|---|
| Patients | 17 (arriving in 6 waves across the episode) |
| Doctors | 2 |
| Steps | 20 |
| Grader | Throughput (75%) + MTS Timeliness Fairness (25%) |

**Challenge:** Emergency patients arrive unexpectedly at steps 4 and 10, requiring rapid reprioritization. A pure greedy policy neglects lower-priority patients indefinitely, which the timeliness fairness component penalizes. The agent must balance urgency with equity.

**Grader weight rationale:** Jones SS et al. (2009), *J Biomed Inform* — empirical ED throughput/fairness tradeoff analysis.

**Baseline scores**

| Seed | Score |
|---|---|
| 42 | 0.7784 |
| 7 | 0.7755 |
| 99 | 0.7670 |
| 5015 | 0.5972 |

---

### Task 3 — Mass Casualty Resource Management `hard`

| Property | Value |
|---|---|
| Patients | 20 (across 7 arrival waves) |
| Doctors | 3 |
| ICU beds | 2 (consumed permanently when used) |
| Steps | 22 |
| Grader | Survival (40%) + Time-to-Treatment (25%) + Timeliness (20%) + Resource Efficiency (15%) |

**Complexity elements:**

- **Patient deterioration:** P005 and P006 worsen if untreated within 4 and 3 steps respectively. Missing the countdown means their severity increases and they become harder to treat.
- **ICU constraints:** 2 ICU beds total. Patients requiring ICU cannot be served when beds are full. Running out before step 15 is catastrophic.
- **Specialist care:** Some patients require 2 doctors simultaneously. Serving them when only 1 doctor is available returns an error without wasting a step.
- **Triage uncertainty:** P008 self-reports severity 2 but true severity is 3. The agent acts on reported severity — this reflects real clinical triage error.
- **Mass casualty event at step 15:** 5 patients arrive simultaneously (3 emergencies, 2 very urgent). An agent that has not conserved ICU and specialist capacity cannot handle the surge.

**Grader weight rationale:** WHO Emergency Care System Framework (2019). Survival weighted highest (40%) because failure here means patient death.

**Baseline scores**

| Seed | Score |
|---|---|
| 42 | 0.8152 |
| 7 | 0.8449 |
| 99 | 0.8230 |
| 5015 | 0.2890 |

---

## Reward Function

Derived from the Manchester Triage System (MTS). 1 step ≈ 10 minutes of real time.

| Severity | Name | Color | Reward at wait=0 | Decay |
|---|---|---|---|---|
| 1 | IMMEDIATE | 🔴 RED | 1.00 | Cliff: 0.60→0.20→0.00 |
| 2 | VERY_URGENT | 🟠 ORANGE | 1.00 | 0.125/step |
| 3 | URGENT | 🟡 YELLOW | 0.85 | 0.071/step |
| 4 | LESS_URGENT | 🟢 GREEN | 0.60 | 0.040/step |
| 5 | NON_URGENT | 🔵 BLUE | 0.40 | 0.020/step |

**Wait penalties** (applied when agent calls `wait()` with patients in queue):

| Condition | Penalty |
|---|---|
| Emergency (severity 1) in queue | -0.30 per patient |
| Urgent (severity 2-3) in queue | -0.10 |
| Any patient in queue | -0.05 |
| Empty queue | 0.00 |

Reference: Manchester Triage Group (2014). *Emergency Triage*, 3rd Edition.

---

## Action Space

| Tool | Description | Advances Time? |
|---|---|---|
| `list_tasks()` | List all tasks with metadata | No |
| `start_task(task_id)` | Initialize a task episode | No |
| `get_queue_state()` | Observe current queue (observation only) | No |
| `serve_patient(patient_id)` | Treat a patient — core action | **Yes (+1 step)** |
| `wait()` | Skip step, penalized if patients waiting | **Yes (+1 step)** |
| `finalize_episode()` | Compute final score via grader | No |
| `get_current_state()` | Environment metadata | No |

**Resource errors** (no ICU bed, insufficient doctors) do **not** advance time. The agent receives an error message and must choose a different patient. This prevents wasted steps on impossible actions.

---

## Observation Space

`get_queue_state()` returns:

```json
{
  "step": 4,
  "max_steps": 20,
  "steps_remaining": 16,
  "queue": [
    {
      "patient_id": "P007",
      "severity": 1,
      "severity_name": "IMMEDIATE",
      "triage_color": "RED",
      "wait_time": 0,
      "condition": "stable",
      "deterioration_countdown": 2,
      "requires_icu": false,
      "requires_specialist": false,
      "can_serve_now": true
    }
  ],
  "queue_length": 5,
  "available_doctors": 2,
  "available_icu_beds": 1,
  "total_icu_beds": 2,
  "patients_served": 3,
  "missed_emergencies": 0,
  "cumulative_reward": 2.375,
  "done": false
}
```

---

## Episode Workflow

```
list_tasks()
  → start_task(task_id)
  → get_queue_state()           # observe (no time cost)
  → loop until done:
      serve_patient(patient_id) # or wait()
      get_queue_state()         # observe again if needed
  → finalize_episode()          # returns normalized score in (0, 1)
```

---

## Policy Distinguishability

The environment produces meaningfully different scores for different policies — confirming genuine RL signal:

| Task | Optimal (greedy) | Random | Worst (reverse) | Gap |
|---|---|---|---|---|
| Basic Triage | 0.9955 | 0.8058 | 0.6979 | +0.30 |
| Dynamic Queue | 0.9706 | 0.7996 | 0.7335 | +0.24 |
| Mass Casualty | 0.8748 | 0.8187 | 0.7691 | +0.11 |

---

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t queue-doctor .
docker run -p 7860:7860 queue-doctor
```

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_URL="https://ajaybandiwaddar01-queue-doctor.hf.space"

python inference.py
```

### Use the Client Directly

```python
import json
from client import QueueDoctorEnv

with QueueDoctorEnv(base_url="http://localhost:7860").sync() as env:
    env.reset()
    env.call_tool("start_task", task_id="task_1_easy")

    while True:
        state = json.loads(env.call_tool("get_queue_state"))
        if state["done"] or not state["queue"]:
            break
        patient_id = state["queue"][0]["patient_id"]
        env.call_tool("serve_patient", patient_id=patient_id)

    result = json.loads(env.call_tool("finalize_episode"))
    print(f"Score: {result['score']}")
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns `{"status": "healthy"}` |
| `/reset` | POST | Reset episode |
| `/step` | POST | Execute action |
| `/state` | GET | Get current state |
| `/docs` | GET | Interactive Swagger UI |

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.1-8B-Instruct` |
| `HF_TOKEN` | HuggingFace API key | — |
| `ENV_URL` | Environment server URL | `http://localhost:7860` |

---

## Project Structure

```
queue-doctor/
├── Dockerfile
├── README.md
├── client.py                    # QueueDoctorEnv(MCPToolClient)
├── inference.py                 # Baseline LLM agent script
├── openenv.yaml                 # OpenEnv spec manifest
├── pyproject.toml
├── requirements.txt
└── server/
    ├── app.py                   # FastAPI via create_app()
    ├── graders.py               # Principled graders (academic citations)
    ├── models.py                # Patient dataclass (MTS-based)
    ├── queue_engine.py          # Deterministic MDP simulation engine
    ├── queue_environment.py     # MCPEnvironment with 7 MCP tools
    └── tasks.py                 # Deterministic patient arrival schedules
```

---

## Clinical References

- Manchester Triage Group (2014). *Emergency Triage*, 3rd Edition. Wiley-Blackwell.
- Jones SS et al. (2009). "A Multivariate Time Series Approach to Modeling and Forecasting Demand in the Emergency Department." *J Biomed Inform* 42(1):123–139.
- WHO (2019). "Emergency Care System Framework." World Health Organization Technical Report.
- Jain R, Chiu D, Hawe W (1984). "A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared Computer Systems." DEC Technical Report TR-301.