# Robot Gripper × VLA — IROS 2027 Execution Plan

> **Working title:** *Same Brain, Different Fingers: Task-Aligned Gripper Morphology for Vision-Language-Action Manipulation*
>
> **Target:** IROS 2027
>
> **Target submission date:** 2027-03-01

This repository tracks a research project on whether **task-aligned gripper finger morphology can improve manipulation success under the same robot and shared policy**.

The detailed research protocol and experimental TODO remain in [`IROS_2027_TODO.md`](./IROS_2027_TODO.md). This README does **not** replace that document. Its purpose is to translate the research plan into a schedule that matches the actual amount of time available for physical experiments at the Shanghai Jiao Tong University lab.

---

## 1. Core research question

Under the following controlled conditions:

- same robot platform;
- same gripper actuator and interface;
- same sensors;
- same action space;
- same policy architecture and checkpoint;
- same test objects and initial-condition protocol;

can different finger morphologies produce a stable **task type × gripper morphology interaction effect**?

The main hypothesis is that:

- a **narrow precision finger** should be better for constrained, fine manipulation;
- a **wide stable finger** should be better for large-contact, transport-oriented manipulation;
- a neutral finger should serve as the baseline;
- a morphology-conditioned shared policy should handle all three finger types better than a policy that is unaware of morphology.

---

## 2. Frozen minimum experimental scope

### Gripper morphologies

- `G_P`: precision finger, target tip width ≈ 8 mm
- `G_N`: neutral baseline, target tip width ≈ 18 mm
- `G_W`: stable finger, target tip width ≈ 32 mm

For the first complete experiment, **only terminal finger width should change** as much as possible. Length, material, surface treatment, interface, TCP, maximum gripping force, and mass should remain controlled or documented.

### Main tasks

Precision-oriented:

- `P1`: insertion of an oriented connector
- `P2`: thin-object pick/place inside a narrow slot

Stability-oriented:

- `W1`: bottle grasp followed by lateral acceleration/deceleration transport
- `W2`: wide-box grasp, transport, and placement

### Main real-robot test matrix

```text
3 grippers × 4 tasks × 3 unseen objects × 20 paired trials = 720 trials
```

The 720-trial experiment is the **minimum primary evidence**. Additional tasks, selector models, automatic gripper exchange, or more complicated morphology learning are optional and must not block the main paper.

---

## 3. Real-world time constraints

The project schedule must be based on **physical robot access**, not calendar days.

### Semester schedule

- Semester begins: **2026-09-07**
- During normal weeks, most days are spent at university.
- In practice, roughly **one full weekend day per week** can be used at the SJTU lab.
- Travel time from university to the SJTU lab is about **1.5 hours one way**.

### Concentrated lab windows

- Mid-Autumn Festival: **3 days**
- National Day holiday: **7 days**
- New Year holiday: **3 days**
- Winter concentrated lab period: **2027-01-11 to 2027-01-24 at the latest**
- New semester begins: **2027-02-22**

### Planning assumption

The calendar may contain around forty theoretically usable lab days from September to late January, but the project should be planned around only **~32–35 reliable physical experiment days** after accounting for coursework, travel, equipment occupancy, failures, exams, and unexpected interruptions.

Therefore:

> **Every physical lab day must be treated as an execution day rather than a software-development day.**

---

## 4. School vs. SJTU lab division of work

### Work that should be completed at school / remotely

- literature review;
- experiment design;
- CAD modification before printing;
- policy code development;
- dataset tooling;
- training scripts;
- replay and offline evaluation;
- simulation;
- statistics code;
- plotting;
- paper writing;
- experiment configuration preparation;
- issue diagnosis from previous robot runs;
- model training and checkpoint comparison;
- Git / data / experiment-log organization.

### Work that should be reserved for the SJTU lab

- robot calibration;
- gripper installation and measurement;
- fixture installation;
- real-robot data collection;
- teleoperation demonstrations;
- closed-loop policy tests;
- mechanical interaction tests;
- force/slip/collision validation;
- formal trials;
- hardware ablations;
- final video evidence.

A successful weekly cycle should look like:

```text
School / remote days
    ↓
code + training + simulation + experiment preparation
    ↓
freeze the weekend experiment configuration
    ↓
SJTU lab day
    ↓
calibrate → execute trials → collect logs/video → backup
    ↓
return to school and analyze results
```

The goal is to avoid spending valuable robot time on package installation, large code rewrites, environment debugging, or exploratory architecture changes.

---

## 5. Hard execution milestones

| Date | Required state |
|---|---|
| **2026-09-06** | Three finger prototypes and four task prototypes substantially ready; basic logging pipeline available |
| **2026-09-27** | Gate 1: at least one precision task and one stability task show repeatable morphology-dependent trends |
| **2026-10-07** | National Day sprint complete: full robot-policy loop demonstrated on at least two tasks |
| **2026-11-01** | Gate 2: shared-policy pipeline stable enough for repeatable data collection |
| **2026-12-13** | Main training data largely collected; model candidates and experiment configuration ready to freeze |
| **2026-12-14** | Gate 3: hardware, tasks, metrics, object split, model-selection rule and formal protocol frozen |
| **2027-01-10** | Formal testing already underway; ideally 150–250 valid primary trials completed |
| **~2027-01-20** | Target date for completing the 720 primary trials |
| **2027-01-24** | **All required real-robot experiments finished** |
| **2027-02-07** | Full paper draft v1 |
| **2027-02-14** | Internal review version |
| **2027-02-21** | **Submission-ready paper** |
| **2027-03-01** | Target IROS submission deadline |

`2027-02-21` is treated as the **personal deadline**, because the new semester begins on `2027-02-22`. The final week before submission should only be used for formatting, reference checks, typo fixes, anonymous-material checks, PDF compilation, and other low-risk changes.

---

## 6. Phase strategy under limited robot access

### Phase 0 — Before semester start

**Now → 2026-09-06**

Primary goal: eliminate mechanical uncertainty while lab access is relatively concentrated.

Priority order:

1. finalize common gripper interface and equal-TCP geometry;
2. manufacture / print `G_P`, `G_N`, `G_W`;
3. measure real dimensions and mass;
4. build first versions of P1/P2/W1/W2 fixtures;
5. establish automated trial logging and video naming;
6. run scripted mechanical tests;
7. verify that the task difficulty is neither trivial nor impossible.

The desired state on September 7 is:

> **The gripper hardware is basically usable, and weekend lab visits are for experiments rather than continued gripper invention.**

---

### Phase 1 — Mechanical identifiability

**2026-09-07 → 2026-09-27**

Use weekend and Mid-Autumn lab access to establish whether morphology genuinely affects task mechanics.

Minimum target:

- three grippers physically usable;
- four task fixtures reproducible;
- at least 10 small-sample tests per useful gripper-task cell where practical;
- clear failure taxonomy;
- at least one precision task favors the precision morphology;
- at least one stability task favors the wider morphology.

#### Gate 1 — 2026-09-27

Continue the VLA main line only if a repeatable cross-task morphology effect exists.

If this effect is absent, only one controlled revision to task difficulty / fixture / width parameter is allowed before deciding whether the main hypothesis needs to be narrowed.

---

### Phase 2 — National Day system sprint

**2026-09-28 → 2026-10-07**

The seven-day National Day window is the first major system-integration sprint.

The most important output is not high success rate. The most important output is a complete closed loop:

```text
observation
    ↓
policy / VLA
    ↓
action
    ↓
Piper
    ↓
gripper
    ↓
task execution
    ↓
automatic logging
```

By the end of October 7, the system should ideally demonstrate:

- at least two tasks in closed loop;
- at least two or three finger morphologies supported by the same pipeline;
- fixed observation format;
- fixed action representation;
- stable camera and robot synchronization;
- complete experiment logs;
- repeatable model launch / evaluation scripts.

This window should also pull forward part of the data-collection work originally scheduled for November–December.

---

### Phase 3 — Shared policy and data loop

**2026-10-08 → 2026-11-01**

Main objective: turn the National Day prototype into a repeatable research pipeline.

Required comparisons:

1. shared policy without morphology input;
2. shared policy with discrete gripper ID;
3. shared policy with continuous morphology parameters.

A minimal morphology vector may include:

```text
[tip_width, finger_length, corner_radius, finger_mass]
```

All variants must preserve the intended experimental claim: the comparison should not quietly become three independent policies trained separately for three grippers.

#### Gate 2 — 2026-11-01

Continue only if:

- at least two tasks execute reliably enough for data collection;
- training is reproducible;
- logs are complete;
- morphology conditioning does not destroy the neutral baseline;
- the same policy pipeline can operate all target morphologies.

---

### Phase 4 — Full data collection and pre-experiments

**2026-11-02 → 2026-12-13**

Because normal semester weeks provide very little physical robot time, this phase must use weekends primarily for **data acquisition and pre-experiments**, not architecture exploration.

Goals:

- finish training demonstrations for all three grippers and four tasks;
- freeze train / validation / test object split by physical object instance;
- train all three morphology-conditioning variants;
- run multiple training seeds offline;
- verify real-robot stability on representative cases;
- estimate true time per formal trial;
- verify automatic success/failure logging;
- prepare all final plots and statistics scripts before formal data arrives.

By early December, most remaining uncertainty should be experimental variance rather than system engineering.

---

### Phase 5 — Formal experiment freeze

**2026-12-14**

Freeze:

- finger hardware;
- task definitions;
- fixtures;
- train/test objects;
- initial-condition sampling procedure;
- observation and action spaces;
- model checkpoints / selection rule;
- success criterion;
- failure taxonomy;
- primary metrics;
- statistical analysis plan.

After this point, formal results must not be used as justification for repeatedly redesigning the experiment.

---

### Phase 6 — Formal real-robot testing

**2026-12-15 → 2027-01-24**

Primary target:

```text
720 valid paired real-robot trials
```

The first 5% should be treated as smoke testing for:

- logging;
- calibration;
- initial-condition generation;
- success criteria;
- video capture;
- randomization.

Formal trial order should be randomized across:

- gripper;
- task;
- object;
- initial-condition seed.

#### Target throughput

If all 720 trials were concentrated into the January 11–24 period:

```text
720 / 14 ≈ 52 trials/day
```

This is feasible only if the system is already stable before January 11. Therefore the real goal is to begin formal testing in December and reach roughly **150–250 valid trials before January 11**.

Then the January concentrated period becomes a high-throughput evidence-collection sprint rather than a debugging sprint.

---

## 7. Winter lab sprint: 2027-01-11 → 2027-01-24

This is the final guaranteed concentrated real-robot window.

### Rule

> **No architecture-level development during this period unless the current system is completely unusable.**

Avoid:

- switching to a new VLA model;
- redesigning the gripper from scratch;
- changing the action space;
- introducing a new perception stack;
- changing the primary hypothesis after seeing results.

Recommended sequence:

### Jan 11–12

- calibration;
- smoke tests;
- verify all three grippers;
- verify all four tasks;
- confirm logging and randomization.

### Jan 13–18

- high-throughput primary trials.

### Jan 19–20

- finish remaining 720-trial matrix.

### Jan 21–22

- mandatory hardware-dependent ablations;
- scripted mechanical baseline;
- morphology-conditioning comparison where real-robot evidence is required;
- visual-occlusion ablation if it remains part of the paper.

### Jan 23

- rerun missing / invalid trials according to pre-defined rules;
- resolve device-failure gaps;
- final video capture.

### Jan 24

- final data audit;
- verify trial counts;
- verify all metadata;
- create multiple independent backups;
- freeze the real-robot dataset.

**All experiments requiring physical access to the SJTU robot should be completed by this date.**

---

## 8. Paper-only period

**2027-01-25 → 2027-02-21**

This period should require no physical robot access.

Primary work:

- data cleaning under pre-defined rules;
- mixed-effects logistic regression;
- confidence intervals;
- per-object analysis;
- `gripper × task_type` interaction analysis;
- `Delta_match` analysis;
- failure-mode analysis;
- tables and plots;
- paper writing;
- supplementary material;
- experiment video editing;
- code cleanup;
- independent number verification.

Suggested internal deadlines:

- **2027-02-07:** complete first full manuscript
- **2027-02-14:** internal review version
- **2027-02-18:** freeze main figures, tables, supplement and video
- **2027-02-21:** submission-ready package

---

## 9. Weekly project dashboard

Every week, update only a small set of operational numbers:

- completed / planned CAD and fixture work;
- number of valid demonstrations;
- number of valid formal or pre-experiment trials;
- P1/P2/W1/W2 closed-loop status;
- automatic logging completeness;
- dominant failure modes;
- current estimated date for reaching 720 trials;
- current largest blocker;
- unfinished items before the next Gate.

A simple weekly status format:

```text
Week:

Hardware:
- G_P:
- G_N:
- G_W:

Tasks:
- P1:
- P2:
- W1:
- W2:

Data:
- valid demonstrations:
- valid pre-trials:
- valid formal trials:
- logging completeness:

Models:
- no morphology input:
- discrete gripper ID:
- continuous morphology parameters:

Largest blocker:

Estimated 720-trial completion date:

Next Gate:

Decision:
- on track / at risk / fallback required
```

---

## 10. Lab-day operating rule

Because one SJTU lab visit costs about three hours of round-trip travel, every visit should have a pre-written experiment sheet before departure.

A good lab day should begin with:

- code already pulled and tested;
- required checkpoints already trained;
- exact task configuration known;
- exact trial count known;
- fixtures and printed parts ready;
- automatic logging enabled;
- backup destination prepared.

And end with:

- all experiment logs copied;
- videos indexed;
- invalid trials marked rather than silently deleted;
- calibration state recorded;
- next week's blockers written down.

---

## 11. Main risk hierarchy

The project risks are ordered approximately as follows:

1. **robot-policy pipeline is not stable by October/November;**
2. training-data collection consumes too many scarce lab days;
3. task difficulty does not create a measurable morphology interaction;
4. hardware / fixtures are not repeatable enough for paired testing;
5. automatic logs or trial metadata are incomplete;
6. formal testing starts too late;
7. too many optional ideas distract from the 720-trial core evidence;
8. paper writing is delayed until after all experiments are finished.

The project should therefore optimize for **early system closure**, not early sophistication.

---

## 12. Scope control

The following are optional and must not delay the minimum paper:

- automatic morphology selector;
- extra high-torque task category;
- six-task / 1080-trial expansion;
- automatic gripper exchange;
- morphology generation network;
- complex adaptive control conditioned on geometry;
- additional sensors that are not necessary for the main claim.

The preferred order is:

```text
prove morphology effect
    ↓
prove shared-policy feasibility
    ↓
finish 720-trial evidence
    ↓
finish morphology-conditioning ablation
    ↓
only then add optional contributions
```

---

## 13. Decision philosophy

The goal is not to force a paper into IROS at any cost.

If the required evidence is not ready at a Gate:

- narrow the claim;
- remove optional work;
- preserve experimental integrity;
- or move to a later RA-L / T-RL route rather than compromising the protocol.

A strong delayed paper is preferable to a rushed result whose experimental design cannot support the claim.

---

## 14. Current immediate priorities

Before the semester begins, the project should prioritize:

- [ ] freeze the three finger-interface sketches;
- [ ] manufacture the first three finger variants;
- [ ] measure real geometry and mass;
- [ ] prototype all four task fixtures;
- [ ] define exact success / failure / timeout conditions;
- [ ] build trial logging and video naming;
- [ ] run first scripted mechanical tests;
- [ ] prepare remote code / training / data synchronization;
- [ ] ensure weekend lab sessions can begin directly with robot execution.

---

## Repository documents

- [`README.md`](./README.md): execution plan under real lab-access constraints
- [`IROS_2027_TODO.md`](./IROS_2027_TODO.md): detailed research protocol, experimental design, gates, metrics, and original TODO

The README should evolve as the real project schedule changes, while the frozen experiment protocol should only change through explicit research decisions rather than convenience.