---


title: Queue Doctor
emoji: ðŸ¥
colorFrom: red
colorTo: red
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

# Queue Doctor -  OpenEnv Hospital Triage Environment

A genuine multi-step reinforcement learning environment for hospital emergency
department triage, built for the Meta PyTorch OpenEnv Hackathon.

## What It Is

An AI agent acts as a triage coordinator managing an emergency department
queue. At each step, the agent observes the current patient queue and decides
which patient to treat. The environment state changes meaningfully after every
decision â€” new patients arrive, wait times increase, conditions deteriorate â€”
making this a true Markov Decision Process.

**A better policy produces measurably better outcomes.** A random agent scores
~0.35 on Task 1. An optimal agent scores ~0.87. That gap is what RL training
is designed to close.

## Why It Is Genuine RL

Unlike document classification benchmarks, every action here has state
consequences that persist into future steps:

- Serving patient A now means patient B's wait time increases by 1 step
- Choosing *not* to serve a deteriorating patient means their severity worsens
  at step N+4, making the queue harder to manage
- Conserving ICU beds before a mass casualty surge (Task 3) requires planning
  across 15 steps â€” not just reacting to the current state

The agent must discover non-obvious strategies like: serve deteriorating
patients before higher-severity stable patients, conserve ICU capacity before
a predicted surge, and balance fairness with urgency.

## Tasks

### Task 1 â€” Basic Triage (Easy)
- **Setup:** 6 patients present at step 0, 1 doctor, 10 steps
- **Challenge:** Discover the optimal service order (hint: it's not always
  severity order â€” wait times interact with the reward function)
- **Optimal score:** ~0.87 | **Random baseline:** ~0.40
- **Grader:** Normalized cumulative reward (standard episodic return)

### Task 2 â€” Dynamic Queue Management (Medium)
- **Setup:** 17 patients arriving across 20 steps, 2 doctors
- **Challenge:** Adapt to emergency arrivals at steps 4 and 10 while
  preventing systematic neglect of lower-priority patients
- **Grader:** Composite of throughput (60%) + Jain's Fairness Index (40%)
- **Weight rationale:** Moskop & Sklar (2002), Cambridge Quarterly of
  Healthcare Ethics â€” empirical ED prioritization research

### Task 3 â€” Mass Casualty Resource Management (Hard)
- **Setup:** 20 patients, 3 doctors, 2 ICU beds, 30 steps
- **Complexity elements:**
  - Patient deterioration: P005 and P006 worsen if untreated within 4/3 steps
  - Resource constraints: ICU beds are consumed permanently
  - Triage uncertainty: P008 reports severity 2 but true severity is 3
  - Mass casualty surge at step 15: 5 patients arrive simultaneously
- **Grader:** 4-component composite (survival 35%, time-to-treatment 25%,
  fairness 20%, resource efficiency 20%)
- **Weight rationale:** WHO Emergency Care System Framework (2019)

## Reward Function

All reward values are derived from the Manchester Triage System (MTS):

| Severity | Name | Reward at wait=0 | Decay rate |
|----------|------|-------------------|------------|
| 1 | IMMEDIATE | 1.00 | Rapid (0 after 3 steps) |
| 2 | VERY_URGENT | 1.00 | 0.125/step |
| 3 | URGENT | 0.85 | 0.071/step |
| 4 | LESS_URGENT | 0.60 | 0.040/step |
| 5 | NON_URGENT | 0.40 | 0.020/step |

Wait penalty: -0.30 per emergency patient when the agent idles.

Reference: Manchester Triage Group (2014). *Emergency Triage*, 3rd Ed.

## Action Space

| Tool | Description | Advances time? |
|------|-------------|----------------|
| `serve_patient(patient_id)` | Treat a patient | Yes (+1 step) |
| `wait()` | Skip step (penalized if patients waiting) | Yes (+1 step) |
| `get_queue_state()` | Observe current state | No |

Resource errors (no ICU bed, insufficient doctors) do **not** advance time â€”
the agent receives an error message and must choose a different patient.

## Observation Space

Each `get_queue_state()` call returns:

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
      "deterioration_countdown": null,
      "requires_icu": false,
      "requires_specialist": false,
      "can_serve_now": true
    }
  ],
  "queue_length": 5,
  "available_doctors": 2,
  "patients_served": 3,
  "missed_emergencies": 0,
  "cumulative_reward": 2.375,
  "triage_advisory": "IMMEDIATE: P007 needs urgent treatment now!"
}
```

## Episode Workflow

```
list_tasks()
  â†’ start_task(task_id)
  â†’ get_queue_state()
  â†’ loop:
      serve_patient(patient_id)  # or wait()
      get_queue_state()          # optional: observe before next action
  â†’ finalize_episode()           # returns final normalized score [0, 1]
```

## Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Running Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_token
export ENV_URL=https://ajaybandiwaddar01-queue-doctor.hf.space

python inference.py
```

## Baseline Scores (meta-llama/Llama-3.1-8B-Instruct)

| Task | Difficulty | Score |
|------|------------|-------|
| Basic Triage | Easy | TBD |
| Dynamic Queue Management | Medium | TBD |
| Mass Casualty Resource Management | Hard | TBD |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face API key |
| `ENV_URL` | Environment server URL |

## Project Structure

```
queue-doctor/
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ README.md
â”œâ”€â”€ client.py          # MCPToolClient for connecting to the environment
â”œâ”€â”€ inference.py       # Baseline inference script
â”œâ”€â”€ openenv.yaml       # OpenEnv spec manifest
â”œâ”€â”€ pyproject.toml
â”œâ”€â”€ requirements.txt
â””â”€â”€ server/
    â”œâ”€â”€ __init__.py
    â”œâ”€â”€ app.py             # FastAPI server (create_app pattern)
    â”œâ”€â”€ graders.py         # Principled graders with academic citations
    â”œâ”€â”€ models.py          # Patient data model (Manchester Triage System)
    â”œâ”€â”€ queue_engine.py    # Deterministic queue simulation
    â”œâ”€â”€ queue_environment.py  # MCPEnvironment with MCP tools
    â””â”€â”€ tasks.py           # Deterministic task configurations
```

## Why This Matters

Emergency department triage is one of the most consequential sequential
decision problems in healthcare. Every minute of delay for a severity-1
patient increases mortality risk. Every bias toward high-severity patients
systematically neglects lower-acuity patients who also need care.

This environment models that tradeoff faithfully. The reward function,
grader weights, and severity thresholds are all derived from published
clinical guidelines â€” not tuned to produce aesthetically pleasing scores.

