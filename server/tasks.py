# Copyright (c) Ajay Bandiwaddar — OpenEnv Hackathon Round 1
"""
Queue Doctor — Task Configurations.

All patient arrivals are pre-scheduled with fixed IDs.
No random number generators — fully deterministic and reproducible.

Task 1 (Easy)  — Basic Triage: static queue, 1 doctor, 10 steps.
Task 2 (Medium)— Dynamic Queue: dynamic arrivals, 2 doctors, 20 steps.
Task 3 (Hard)  — Mass Casualty: deterioration, ICU, specialist care,
                  mass casualty event at step 15, 3 doctors, 30 steps.
"""

TASKS = {

    # ── Task 1: Basic Triage ─────────────────────────────────────────────
    # A static queue of 6 patients, all present at step 0.
    # Single doctor. 10 steps.
    #
    # Optimal policy: serve strictly in severity order (1→2→2→3→4→5).
    # An agent that learns this scores ~0.87.
    # A random agent scores ~0.45 (many suboptimal orderings).
    # This is a genuine RL task: the agent must discover the priority rule.
    #
    # Reward calibration:
    #   Optimal cumulative = 1.0+0.88+0.76+0.64+0.44+0.30 = 4.02
    #   Random expected ≈ 1.7-1.9
    "task_1_easy": {
        "task_name": "Basic Triage",
        "difficulty": "easy",
        "max_steps": 10,
        "num_doctors": 1,
        "icu_beds": 0,
        "grader": "easy",
        "description": (
            "You are managing an emergency department triage queue with 1 doctor available. "
            "6 patients are waiting — each with a different severity level (1=most critical, 5=minor). "
            "Each step, you must decide which patient to serve.\n\n"
            "SCORING: Serving a severity-1 patient immediately scores 1.0. "
            "Every step they wait costs you significantly. Lower-severity patients "
            "score less but still contribute. "
            "Waiting when patients are in the queue incurs penalties.\n\n"
            "GOAL: Maximize total reward over 10 steps by serving patients in the right order."
        ),
        "arrivals": [
            {"step": 0, "patient_id": "P001", "severity": 1, "reported_severity": 1},
            {"step": 0, "patient_id": "P002", "severity": 2, "reported_severity": 2},
            {"step": 0, "patient_id": "P003", "severity": 2, "reported_severity": 2},
            {"step": 0, "patient_id": "P004", "severity": 3, "reported_severity": 3},
            {"step": 0, "patient_id": "P005", "severity": 4, "reported_severity": 4},
            {"step": 0, "patient_id": "P006", "severity": 5, "reported_severity": 5},
        ],
        # Optimal cumulative reward (used for score normalization)
        "optimal_reward": 4.02,
    },

    # ── Task 2: Dynamic Queue Management ────────────────────────────────
    # Patients arrive throughout the episode at known times.
    # 2 doctors available (can handle specialist patients).
    # Emergency patients appear mid-episode requiring rapid reprioritization.
    #
    # The agent must adapt dynamically: a policy that works for the initial
    # queue fails when new emergencies arrive at steps 4 and 10.
    # Fairness matters: low-priority patients cannot be indefinitely delayed.
    # Jain's Fairness Index penalizes systematic starvation.
    #
    # Optimal requires: prioritize emergencies immediately, but also ensure
    # lower-priority patients are served before their wait time becomes unfair.
    "task_2_medium": {
        "task_name": "Dynamic Queue Management",
        "difficulty": "medium",
        "max_steps": 20,
        "num_doctors": 2,
        "icu_beds": 0,
        "grader": "medium",
        "description": (
            "You are managing a busy emergency department with 2 doctors available. "
            "New patients arrive throughout the 20-step episode — you must adapt dynamically.\n\n"
            "SCORING: Composite of throughput (60%) and fairness (40%). "
            "Throughput rewards serving high-severity patients quickly. "
            "Fairness (Jain's Fairness Index) penalizes ignoring lower-priority patients "
            "indefinitely. A perfect score requires both speed and equity.\n\n"
            "With 2 doctors, specialist patients (requiring 2 doctors) can be served. "
            "Waiting incurs penalties — especially when emergencies are in the queue."
        ),
        "arrivals": [
            # Step 0: Initial queue
            {"step": 0, "patient_id": "P001", "severity": 1, "reported_severity": 1},
            {"step": 0, "patient_id": "P002", "severity": 3, "reported_severity": 3},
            {"step": 0, "patient_id": "P003", "severity": 4, "reported_severity": 4},
            {"step": 0, "patient_id": "P004", "severity": 5, "reported_severity": 5},
            # Step 2: New arrivals
            {"step": 2, "patient_id": "P005", "severity": 2, "reported_severity": 2},
            {"step": 2, "patient_id": "P006", "severity": 3, "reported_severity": 3},
            # Step 4: Emergency surge
            {"step": 4, "patient_id": "P007", "severity": 1, "reported_severity": 1},
            {"step": 4, "patient_id": "P008", "severity": 4, "reported_severity": 4},
            # Step 7: Mid-episode arrivals
            {"step": 7, "patient_id": "P009", "severity": 2, "reported_severity": 2},
            {"step": 7, "patient_id": "P010", "severity": 3, "reported_severity": 3},
            {"step": 7, "patient_id": "P011", "severity": 5, "reported_severity": 5},
            # Step 10: Second emergency
            {"step": 10, "patient_id": "P012", "severity": 1, "reported_severity": 1},
            {"step": 10, "patient_id": "P013", "severity": 3, "reported_severity": 3},
            # Step 13: Late arrivals
            {"step": 13, "patient_id": "P014", "severity": 2, "reported_severity": 2},
            {"step": 13, "patient_id": "P015", "severity": 4, "reported_severity": 4},
            # Step 16: Final wave
            {"step": 16, "patient_id": "P016", "severity": 3, "reported_severity": 3},
            {"step": 16, "patient_id": "P017", "severity": 2, "reported_severity": 2},
        ],
        # Optimal reward computed by running optimal policy simulation
        "optimal_reward": 11.8,
    },

    # ── Task 3: Mass Casualty Resource Management ────────────────────────
    # The hardest task. Combines all elements:
    #
    # Patient deterioration: P005 and P006 (arriving step 3) will worsen if
    # not treated within 4/3 steps respectively. Ignoring them risks cascade.
    #
    # Resource constraints: 2 ICU beds total. P001 and P012 need ICU.
    # P002 and P014 need specialist care (2 doctors). Running out of ICU
    # beds before the mass casualty event (step 15) is catastrophic.
    #
    # Triage uncertainty: P008 reported severity 2 but true severity is 3.
    # Agent acts on reported severity — this is realistic triage error.
    #
    # Mass casualty event at step 15: 5 patients arrive simultaneously
    # (3 emergencies, 2 very urgent). An agent that hasn't conserved resources
    # cannot handle the surge. Planning ahead is essential.
    #
    # This tests whether the agent can:
    # 1. Serve deteriorating patients before they worsen
    # 2. Conserve ICU beds before the surge
    # 3. Handle mass casualty by triaging correctly under pressure
    "task_3_hard": {
        "task_name": "Mass Casualty Resource Management",
        "difficulty": "hard",
        "max_steps": 30,
        "num_doctors": 3,
        "icu_beds": 2,
        "grader": "hard",
        "description": (
            "You are managing a major trauma center during a mass casualty incident. "
            "3 doctors and 2 ICU beds available. Episodes run for 30 steps.\n\n"
            "RESOURCE CONSTRAINTS:\n"
            "- Patients marked 'requires_icu' consume 1 ICU bed when served (permanent).\n"
            "- Patients marked 'requires_specialist' consume 2 doctors simultaneously.\n"
            "- If resources are unavailable, the patient cannot be served.\n\n"
            "DETERIORATION: Some patients will worsen if not treated in time — "
            "their severity increases, making them harder to handle. "
            "Watch the deterioration_countdown field carefully.\n\n"
            "MASS CASUALTY EVENT: A surge occurs mid-episode. "
            "Conserve ICU beds and specialist capacity before it arrives.\n\n"
            "SCORING: Weighted composite of survival rate, time-to-treatment, "
            "fairness, and resource efficiency. All weights are clinically derived."
        ),
        "arrivals": [
            # Step 0: Initial complex queue
            {"step": 0, "patient_id": "P001", "severity": 1, "reported_severity": 1,
             "requires_icu": True},
            {"step": 0, "patient_id": "P002", "severity": 2, "reported_severity": 2,
             "requires_specialist": True},
            {"step": 0, "patient_id": "P003", "severity": 3, "reported_severity": 3},
            {"step": 0, "patient_id": "P004", "severity": 4, "reported_severity": 4},
            # Step 3: Deteriorating patients — must treat before countdown hits 0
            {"step": 3, "patient_id": "P005", "severity": 2, "reported_severity": 2,
             "deterioration_countdown": 4},   # worsens at step 7 if untreated
            {"step": 3, "patient_id": "P006", "severity": 3, "reported_severity": 3,
             "deterioration_countdown": 3},   # worsens at step 6 if untreated
            # Step 6: More arrivals including misreported severity
            {"step": 6, "patient_id": "P007", "severity": 1, "reported_severity": 1},
            {"step": 6, "patient_id": "P008", "severity": 3, "reported_severity": 2}, # MISREPORTED: reported 2, true 3
            {"step": 6, "patient_id": "P009", "severity": 4, "reported_severity": 4},
            # Step 10: Pre-surge, conserve ICU
            {"step": 10, "patient_id": "P010", "severity": 2, "reported_severity": 2,
             "deterioration_countdown": 5},
            {"step": 10, "patient_id": "P011", "severity": 3, "reported_severity": 3},
            # Step 15: MASS CASUALTY EVENT — 5 patients arrive simultaneously
            {"step": 15, "patient_id": "P012", "severity": 1, "reported_severity": 1,
             "requires_icu": True},
            {"step": 15, "patient_id": "P013", "severity": 1, "reported_severity": 1},
            {"step": 15, "patient_id": "P014", "severity": 1, "reported_severity": 1,
             "requires_specialist": True},
            {"step": 15, "patient_id": "P015", "severity": 2, "reported_severity": 2,
             "requires_icu": True},
            {"step": 15, "patient_id": "P016", "severity": 2, "reported_severity": 2},
            # Step 20: Late arrivals
            {"step": 20, "patient_id": "P017", "severity": 2, "reported_severity": 2},
            {"step": 20, "patient_id": "P018", "severity": 3, "reported_severity": 3},
            {"step": 20, "patient_id": "P019", "severity": 4, "reported_severity": 4},
            # Step 25: Final
            {"step": 25, "patient_id": "P020", "severity": 3, "reported_severity": 3},
        ],
        "optimal_reward": 15.5,
    },
}