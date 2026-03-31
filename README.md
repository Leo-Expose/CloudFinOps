# ☁️ CloudFinOps-Env

> **Meta AI & HuggingFace OpenEnv Hackathon Submission**

A Reinforcement Learning environment that simulates real-world **cloud infrastructure cost optimization** combined with **SLA incident management** — featuring realistic AWS instance types, stochastic metric noise, delayed scaling physics, and human-in-the-loop text-chat interactions.

---

## ⚡ Quick Start — 3 Steps

> **One file to configure. Everything reads from it automatically.**

### Step 1 — Configure your credentials

Copy the template and fill in your API key:

```bash
cp .env.example .env
```

Open `.env` and set your values:

```env
# The API endpoint for the LLM (OpenAI-compatible)
API_BASE_URL=https://api.openai.com/v1

# The model identifier to use for inference
MODEL_NAME=gpt-4o

# Your Hugging Face / API key
HF_TOKEN=hf_your_token_here
```

> **That's the only file you need to edit.** Both `inference.py` and Docker read from it automatically.

---

### Step 2 — Start the Environment Server

#### Option A: Docker (Recommended)

```bash
docker build -t cloudfinops-env .
docker run --env-file .env -p 8000:8000 cloudfinops-env
```

You should see the CloudFinOps banner and `Ready to accept connections ✓` in the terminal.

#### Option B: Python (No Docker)

```bash
pip install -r requirements.txt
uvicorn env.server:app --host 0.0.0.0 --port 8000
```

---

### Step 3 — Run the Baseline Evaluator

Open a **new terminal** (keep the server running) and run:

```bash
python inference.py
```

The script will:
1. Auto-load `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from your `.env` file
2. Initialize an **OpenAI Client** using those values
3. Run all 3 tasks (`easy`, `medium`, `hard`) against the environment with **automatic retry logic**
4. Print per-task grader scores with visual progress bars and an overall average

**Expected output:**
```
============================================================
  FINAL RESULTS
============================================================
  ✅     easy: 1.0000  ████████████████████░
  ✅   medium: 0.6500  █████████████░░░░░░░░
  ✅     hard: 0.4200  ████████░░░░░░░░░░░░░
     AVERAGE: 0.6900
============================================================
```

---

### Step 4 (Optional) — Run Validation

From the **parent directory** of `cloudfinops-env/`:

```bash
python pre_validation.py --repo-dir ./cloudfinops-env --skip-docker
```

If everything passes, you'll see: **🎉 All checks passed!.**

---
## 🛡️ Validation Suite (`pre_validation.py`)

We include a comprehensive validation script that verifies the entire environment meets hackathon compliance standards in a single command. This was built to give evaluators confidence that the environment is correctly wired — no manual inspection needed.

### Why we included it

OpenEnv environments have a strict contract: typed models, REST endpoints, deterministic graders, a Docker container, and an inference script using the OpenAI Client. Rather than asking evaluators to check each piece manually, `pre_validation.py` automates **21 checks across 5 categories** and reports a clear pass/fail summary.

### What it checks

| # | Category | What's verified |
|---|----------|----------------|
| 1 | **Environment Variables** | `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` are set (auto-loads from `.env`) |
| 2 | **OpenEnv Spec** | `openenv.yaml` is valid, has `name`/`entrypoint`/`tasks`, defines 3+ tasks. Pydantic models (`Action`, `Observation`, `RewardInfo`) exist. Server exposes `/reset`, `/step`, `/state`. |
| 3 | **Dockerfile** | Dockerfile exists, Docker builds successfully, exposes port 8000, runs uvicorn |
| 4 | **Inference Script** | `inference.py` exists in root, imports OpenAI Client, references all 3 env vars, has a `main` entry point |
| 5 | **Task Graders** | All 3 tasks (`easy`, `medium`, `hard`) run without error, each grader returns a score in `[0.0, 1.0]` |
| Bonus | **Resource Constraints** | No heavy ML frameworks (torch, tensorflow, etc.) in requirements |

### How to run

```bash
# From inside the project directory:
python pre_validation.py                   # full check (includes Docker build)
python pre_validation.py --skip-docker     # skip Docker build (faster)
```

### Sample output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CloudFinOps-Env — Validation Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▸ Check 1 — Environment Variables
  ✅ PASS  API_BASE_URL is set
  ✅ PASS  MODEL_NAME is set
  ✅ PASS  HF_TOKEN is set

▸ Check 2 — OpenEnv Spec Compliance
  ✅ PASS  openenv.yaml parseable
  ✅ PASS  openenv.yaml has 3+ tasks
  ✅ PASS  Pydantic models (Action, Observation, Reward)
  ✅ PASS  Server endpoints: /reset, /step, /state

▸ Check 4 — Inference Script (inference.py)
  ✅ PASS  inference.py exists in project root
  ✅ PASS  Uses OpenAI Client
  ✅ PASS  References API_BASE_URL, MODEL_NAME, HF_TOKEN

▸ Check 5 — Task Graders (3 tasks, scores in 0.0–1.0)
  ✅ PASS  Task 'easy'   → Score = 0.0000
  ✅ PASS  Task 'medium' → Score = 0.0000
  ✅ PASS  Task 'hard'   → Score = 0.2080
  ✅ PASS  All 3 tasks produce valid grader scores

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Passed: 21/21
  ✅ All 21 checks passed.
```

---
## 🔐 Environment Variables

All configuration lives in a single `.env` file. The table below documents every variable:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | ✅ Yes | — | OpenAI-compatible API endpoint |
| `MODEL_NAME` | ✅ Yes | — | Model identifier for inference |
| `HF_TOKEN` | ✅ Yes | — | Hugging Face / API key |
| `ENV_BASE_URL` | ❌ No | `http://localhost:8000` | Override the environment server URL |

`inference.py` validates all three mandatory variables at startup and exits with a clear error if any are missing.

---

## 🎯 Motivation & Why This Matters

Cloud infrastructure management is one of the most critical real-world tasks in modern engineering. Platform teams spend thousands of hours annually making decisions about:
- Which idle servers to terminate
- When to scale up before traffic spikes hit
- How to cut costs without causing outages
- Balancing conflicting directives from leadership ("cut costs!") vs reality ("traffic is spiking!")

**CloudFinOps-Env** captures this tension in a multi-objective RL environment where the agent must **simultaneously** optimize competing goals — just like a real SRE on-call.

| Objective | Tension |
|-----------|---------|
| 💰 Cut costs | ↔ Risk overloading remaining servers |
| 🛡️ Prevent SLA breaches | ↔ Requires expensive upscaling |
| ⏱️ Scale up for traffic | ↔ Scaling is **delayed by 1 step** |
| 💬 Follow human instructions | ↔ Humans may give risky directives |
| 📈 Handle metric noise | ↔ Must distinguish real trends from jitter |

---

## 🏗️ Environment Mechanics

### Physics Engine
- **Realistic AWS Instance Types**: Uses real-world instance types (`t3.medium`, `c5.large`, `r6g.xlarge`, `m5.large`) with authentic hourly pricing.
- **Delayed Scaling**: `UPSCALE` takes effect on the *next* step — the agent must anticipate, not react.
- **Upscale Tier Cap**: Each server can only be upgraded up to 2 times through a fixed tier path (e.g., `t3.micro` → `t3.medium` → `t3.large`), preventing infinite scaling exploits.
- **Stochastic Metric Noise**: Server CPU and memory metrics include bounded deterministic noise (seeded by task + server + step), making the environment realistic while remaining fully reproducible.
- **Load Redistribution**: Terminating a server redistributes its load across survivors.
- **SLA Breach = Catastrophe**: Any server hitting 100% CPU triggers an instant SLA failure with heavy penalty.
- **Budget Drain**: Each running server charges per-step costs. Overspending ends the episode.
- **Traffic Ramping**: Hard task features exponential traffic growth simulating a viral campaign.

### Instance Catalog

| Instance Type | vCPU | Memory | $/hour | Category |
|---------------|------|--------|--------|----------|
| `t3.micro` | 2 | 1 GB | $0.0104 | Web |
| `t3.medium` | 2 | 4 GB | $0.0416 | Web |
| `t3.large` | 2 | 8 GB | $0.0832 | Web |
| `c5.large` | 2 | 4 GB | $0.0850 | Compute |
| `c5.xlarge` | 4 | 8 GB | $0.1700 | Compute |
| `r6g.medium` | 1 | 8 GB | $0.0504 | Database |
| `r6g.large` | 2 | 16 GB | $0.1008 | Database |
| `r6g.xlarge` | 4 | 32 GB | $0.2016 | Database |
| `m5.large` | 2 | 8 GB | $0.0960 | Batch |
| `m5.xlarge` | 4 | 16 GB | $0.1920 | Batch |

### Upscale Tier Path

```
t3.micro  →  t3.medium  →  t3.large  (max)
c5.large  →  c5.xlarge              (max)
r6g.medium → r6g.large  →  r6g.xlarge (max)
m5.large  →  m5.xlarge              (max)
```

Each server can be upgraded a **maximum of 2 times**. Attempting to exceed this cap wastes the step.

### Action Space

| Field | Type | Description |
|-------|------|-------------|
| `command` | `Literal["UPSCALE", "DOWNSCALE", "TERMINATE", "REDISTRIBUTE_LOAD", "IGNORE"]` | Infrastructure action to execute |
| `target_id` | `Optional[str]` | ID of the server to act on |
| `reply` | `str` | Text response to stakeholder messages in inbox |

**Action Effects:**

| Command | Effect | Risk |
|---------|--------|------|
| `UPSCALE` | Upgrades to next tier (delayed 1 step), halves CPU utilisation | Expensive; delayed; max 2 per server |
| `DOWNSCALE` | Halves cost immediately, CPU load increases 1.8× | May trigger SLA breach |
| `TERMINATE` | Removes server, load redistributed to survivors | Irreversible |
| `REDISTRIBUTE_LOAD` | Balances CPU evenly across all running servers | May not help if all are loaded |
| `IGNORE` | No-op | Wastes a step; budget still drains |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `servers` | `List[ServerState]` | All server instances with id, type (AWS instance), cpu_util, memory_util, cost_per_hour, status |
| `traffic_load` | `float` | Global traffic level (0–100) |
| `spike_detected` | `bool` | Whether a traffic spike is occurring |
| `incidents` | `List[Dict]` | History of SLA breaches including instance type and CPU at breach |
| `budget_remaining` | `float` | Remaining budget in USD |
| `time_step` | `int` | Current step number |
| `inbox` | `List[str]` | Text messages from simulated stakeholders (CTO, Marketing, Ops, SRE) |

### Reward Function

The reward function provides **continuous signal over the full trajectory**, not just binary end-of-episode feedback:

| Signal | Reward | When |
|--------|--------|------|
| Terminate a server | +10.0 | Server successfully terminated |
| Downscale a server | +5.0 | Cost reduced |
| Redistribute load | +3.0 | Load balanced across fleet |
| Reply to inbox | +2.0 | Agent engages with stakeholder |
| Upscale (investment) | -5.0 | Short-term cost for future capacity |
| Upscale at max tier | -1.0 | Wasted action (already capped) |
| Invalid target | -2.0 | Acting on non-existent/dead server |
| High per-step cost | -1.0 | Running costs exceed $0.50/step |
| Budget overrun | -20.0 | Budget depleted |
| **SLA Breach** | **-100.0** | Any server hits 100% CPU |

---

## 🎮 The 3 Tasks

### 🟢 Easy — *Zombie Cleanup*
- **Setup**: 10 servers — 4 active web (`t3.medium`), 3 active compute (`c5.large`), 3 completely idle (`t3.micro`)
- **Inbox**: "Ops Team: We're seeing 3 zombie instances racking up charges. Please terminate unused servers to save costs."
- **Objective**: Terminate the 3 idle servers without touching active ones
- **Grading**: `score = (idle_terminated / 3) - (active_terminated × 0.25) - (0.5 if SLA breach)`
- **Perfect Score**: Terminate `idle-0`, `idle-1`, `idle-2` → **1.0**

### 🟡 Medium — *The CTO Budget Squeeze*
- **Setup**: 12 over-provisioned servers at ~3–9% CPU across `t3.medium`/`r6g.large`/`m5.large`
- **Inbox**: "CTO: Our cloud bill is through the roof. Cut costs by at least 50% immediately — no excuses."  "Finance: Q3 budget review in 2 days. We need measurable savings by then."
- **Objective**: Reduce spending by 50% without triggering any SLA breaches
- **Grading**: `score = min(cost_saved% / 50%, 1.0) - (0.5 if SLA breach)`
- **Challenge**: Downscaling increases CPU load 1.8× — must be strategic

### 🔴 Hard — *Black Friday Chaos*
- **Setup**: 8 servers — DBs (`r6g.large`) at 78–85% CPU (and climbing), web (`t3.medium`/`c5.large`) at 50–60%, batch (`m5.large`) at 10–20%
- **Inbox**: "Marketing: Massive ad campaign going live RIGHT NOW! Expect 10× normal traffic." + "SRE On-Call: DB-0 is approaching capacity. Consider upscaling before it breaches SLA."
- **Objective**: Prevent SLA failures while managing a tight $4.00 budget
- **Grading**: `score = (uptime × 0.6) + (cost_efficiency × 0.4) + (0.1 inbox bonus)`
- **Challenge**: Traffic ramps exponentially via `log1p(step)`, DB CPU climbs each step. Must upscale DBs (delayed!) while shedding batch jobs. Stochastic noise makes exact thresholds unpredictable.

---

## 📡 API Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/reset` | `{"task_id": "easy\|medium\|hard"}` | `Observation` |
| `POST` | `/step` | `Action` JSON | `{observation, reward, done, info}` |
| `GET`  | `/state` | — | `Observation` (no state advance) |

---

## 📊 Baseline Scores

Baseline scores achieved using `gpt-4o` with temperature 0.1 and structured JSON prompting:

| Task | Score | Notes |
|------|-------|-------|
| 🟢 Easy | ~0.67–1.00 | Depends on correctly identifying idle servers |
| 🟡 Medium | ~0.40–0.80 | Partial credit for cost savings below 50% target |
| 🔴 Hard | ~0.30–0.60 | Multi-objective tradeoff; uptime dominates |
| **Average** | **~0.45–0.80** | Varies by model capability |

> **Note**: Scores are ranges because LLM outputs are stochastic. A perfect rule-based agent can achieve 1.0 on Easy and Medium, ~0.85+ on Hard.

---

## 📐 Grading Criteria

All graders output deterministic scores between **0.0** and **1.0**:

- **Easy**: Full credit for terminating all 3 idle servers, `-0.25` per wrongly terminated active server, `-0.5` for any SLA breach
- **Medium**: Proportional to `cost_saved% / 50%` (capped at 1.0), `-0.5` for any SLA breach
- **Hard**: `(uptime_score × 0.6) + (cost_efficiency × 0.4) + (0.1 if inbox cleared)` where uptime is binary (1.0 if no breach, 0.0 if breached)

---

## 🧠 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **AWS instance types** | Judges care about real-world utility (30% weight). Using authentic `t3.medium`, `r6g.xlarge` etc. demonstrates deep domain understanding. |
| **Deterministic noise** | Uses `md5(task:server:step)` for reproducibility. Same inputs always produce the same noise, but metrics aren't artificially smooth. |
| **Upscale tier cap** | Prevents naive agents from infinitely upgrading a single server. Forces strategic multi-server management. |
| **Delayed scaling** | One of the hardest challenges in real cloud ops. The agent must predict load 1 step ahead. |
| **Human-in-the-loop** | The `inbox` messages with reply rewards add a unique dimension rarely seen in RL environments. |
| **Continuous reward** | Per-step cost pressure, action rewards, and graded final scores provide rich signal at every timestep. |
| **Tenacity retries** | The baseline inference script retries LLM calls up to 3 times with error self-correction, ensuring reliable evaluation during judging. |

---

## 📁 Project Structure

```
cloudfinops-env/
├── env/
│   ├── __init__.py           # Package marker
│   ├── models.py             # Pydantic schemas (ServerState, Action, Observation, RewardInfo)
│   ├── engine.py             # Physics simulator, AWS instances, noise, grading logic
│   └── server.py             # FastAPI endpoints (/reset, /step, /state)
├── openenv.yaml              # OpenEnv metadata (3 tasks + required env vars)
├── inference.py              # Baseline LLM evaluator using OpenAI SDK + tenacity retries
├── pre_validation.py         # ← Automated 21-check validation suite
├── .env.example              # ← Edit this one file with your credentials
├── requirements.txt          # Python dependencies (including openenv-core)
├── Dockerfile                # HF Spaces deployment container
└── README.md                 # This file
```

---

Built with ❤️