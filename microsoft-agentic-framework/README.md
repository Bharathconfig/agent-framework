# Microsoft Agentic Framework – 25 Examples from Beginner to Advanced

A hands-on learning repository for the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
(Python package `agent-framework`). It contains **25 runnable, heavily commented examples** that take you from a
single "hello world" agent to a **guarded multi-agent system (MAS) with a judge agent**.

> **You only need to add two values: your Azure OpenAI API key and endpoint.**
> Everything else (client, model deployment, API version, tools, workflows, guardrails) is already configured.

---

## Table of contents

1. [What you will learn](#what-you-will-learn)
2. [Architecture at a glance](#architecture-at-a-glance)
3. [Prerequisites](#prerequisites)
4. [Get your Azure OpenAI key and endpoint](#get-your-azure-openai-key-and-endpoint)
5. [Quick start (5 minutes)](#quick-start-5-minutes)
6. [The 25 examples](#the-25-examples)
7. [Key concepts explained](#key-concepts-explained)
8. [Guardrails – defence in depth](#guardrails--defence-in-depth)
9. [Multi-agent systems and the judge pattern](#multi-agent-systems-and-the-judge-pattern)
10. [Configuration reference](#configuration-reference)
11. [Project structure](#project-structure)
12. [Troubleshooting](#troubleshooting)
13. [Publishing this as its own GitHub repository](#publishing-this-as-its-own-github-repository)
14. [Further reading](#further-reading)

---

## What you will learn

| Level | Topics |
|-------|--------|
| **Beginner** | Creating an agent, streaming, function tools, multi-turn sessions, structured (Pydantic) output, personas and chat options |
| **Intermediate** | Agent and function middleware, context providers (long-term memory), human-in-the-loop tool approval, agents as tools, saving and restoring sessions |
| **Guardrails** | Input guardrails (prompt injection, PII, blocked topics), output guardrails, tool guardrails (allow-lists, rate limits, hard blocks), LLM-based safety classifier |
| **Workflows** | Graph-based workflows with custom executors, shared state and conditional (switch-case) routing |
| **Multi-agent (MAS)** | Sequential pipelines, concurrent fan-out/fan-in, group-chat debate with an LLM moderator, handoff-based customer support with human input |
| **Judge agent** | Best-of-N selection with a judge, generator ⇄ judge reflection loop, and a capstone guarded MAS with a judge quality gate |

Every file starts with a docstring that explains **what** the example does, **why** the pattern matters, the
**key APIs** used and **how to run it**. Inline comments explain each step.

---

## Architecture at a glance

```
                       ┌──────────────────────────────────────────────┐
   .env  ───────────►  │ common/config.py                             │
 (API key + endpoint)  │  • loads .env, validates settings            │
                       │  • get_chat_client() → Azure OpenAI client   │
                       └──────────────────────┬───────────────────────┘
                                              │ used by every example
        ┌─────────────────────────────────────┼─────────────────────────────────────┐
        ▼                                     ▼                                     ▼
 ┌──────────────┐                    ┌──────────────────┐                  ┌──────────────────┐
 │   Agent      │  tools, sessions,  │   Middleware     │  guardrails.py   │   Workflows /    │
 │ (01 – 12)    │  context providers │  & Guardrails    │  (reusable PII,  │   Orchestrations │
 │              │                    │  (07, 08, 13–16) │  injection, ...) │   (17 – 25)      │
 └──────────────┘                    └──────────────────┘                  └──────────────────┘

 Capstone (25):
   user ─► [Input guardrail] ─► [Safety classifier] ─► Researcher ─► Writer ─► [Judge] ─► [Output guardrail] ─► user
                                       │ unsafe                                   │ fail
                                       └──► polite refusal                        └──► revise (bounded loop)
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | `python --version` |
| **An Azure subscription** | A free account works: <https://azure.microsoft.com/free> |
| **An Azure OpenAI (or Azure AI Foundry) resource** | With a chat model deployment that supports **tool calling** and **structured outputs** (e.g. `gpt-4o-mini`, `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini`) |

No other services, databases or SDKs are required.

---

## Get your Azure OpenAI key and endpoint

1. In the [Azure Portal](https://portal.azure.com) create an **Azure OpenAI** resource
   (or open an existing Azure AI Foundry project).
2. Open **Azure AI Foundry → Deployments → Deploy model** and deploy **`gpt-4o-mini`**.
   *If you choose another name for the deployment, put it in `AZURE_OPENAI_DEPLOYMENT` in `.env`.*
3. Go back to the resource in the Azure Portal → **Resource Management → Keys and Endpoint** and copy:
   * **KEY 1** → `AZURE_OPENAI_API_KEY`
   * **Endpoint** (looks like `https://my-resource.openai.azure.com/`) → `AZURE_OPENAI_ENDPOINT`

---

## Quick start (5 minutes)

```bash
# 1. Go into the project folder
cd microsoft-agentic-framework

# 2. Create and activate a virtual environment
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Install the dependencies (Microsoft Agent Framework + python-dotenv)
pip install -r requirements.txt

# 4. Create your .env file and paste in your key + endpoint
cp .env.example .env          # Windows: copy .env.example .env
#   then edit .env:
#   AZURE_OPENAI_API_KEY=<your key>
#   AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/

# 5. Run something!
python run_example.py --list  # list all 25 examples
python run_example.py 1       # run example 01
python run_example.py 25      # run the capstone
python run_example.py all     # run all 25 one after another
```

You can also run any file directly, from any folder:

```bash
python examples/01_beginner/01_hello_agent.py
```

If the key or endpoint is missing you get a friendly message telling you exactly what to fix. Nothing is
sent to Azure in that case.

---

## The 25 examples

### 🟢 01 – Beginner (`examples/01_beginner/`)

| # | File | What it shows | Key APIs |
|---|------|---------------|----------|
| 01 | `01_hello_agent.py` | The smallest possible agent: name + instructions + one question | `Agent`, `agent.run()` |
| 02 | `02_streaming_agent.py` | Print tokens as they arrive (typewriter effect) | `agent.run(stream=True)` |
| 03 | `03_function_tools.py` | Let the model call your Python functions (weather, time, math) | `@tool`, `Annotated`, `Field` |
| 04 | `04_multi_turn_session.py` | Conversations with memory of previous turns; separate sessions stay isolated | `agent.create_session()`, `session=` |
| 05 | `05_structured_output.py` | Get typed Pydantic objects back instead of free text | `response_format=`, `response.value` |
| 06 | `06_personas_and_options.py` | Many personas from one client, per-run overrides, `tool_choice`, parallel runs | `default_options`, `options=`, `asyncio.gather` |

### 🟡 02 – Intermediate (`examples/02_intermediate/`)

| # | File | What it shows | Key APIs |
|---|------|---------------|----------|
| 07 | `07_agent_middleware.py` | Wrap every agent run: timing, logging, request IDs | `AgentMiddleware`, `AgentContext` |
| 08 | `08_function_middleware.py` | Wrap every tool call: logging, caching, automatic retries | `FunctionMiddleware`, `FunctionInvocationContext` |
| 09 | `09_context_provider_memory.py` | Long-term "user profile" memory injected before every run | `ContextProvider`, `before_run` / `after_run` |
| 10 | `10_human_approval.py` | Sensitive tools need a human to approve before they run | `approval_mode="always_require"`, `user_input_requests` |
| 11 | `11_agent_as_tool.py` | A manager agent delegates to specialist agents exposed as tools | `agent.as_tool()` |
| 12 | `12_session_persistence.py` | Save a conversation to disk and resume it later | `session.to_dict()`, `AgentSession.from_dict()` |

### 🛡️ 03 – Guardrails (`examples/03_guardrails/`)

| # | File | What it shows | Key APIs |
|---|------|---------------|----------|
| 13 | `13_input_guardrails.py` | Block prompt injection and banned topics, redact PII **before** the LLM sees it | `InputGuardrailMiddleware` |
| 14 | `14_output_guardrails.py` | Redact PII / withhold banned content / cap length **after** the LLM answers | `OutputGuardrailMiddleware` |
| 15 | `15_tool_guardrails.py` | Allow-listed arguments, hard blocks, per-tool rate limits, max invocations | `FunctionMiddleware`, `MiddlewareFailure`, `max_invocations` |
| 16 | `16_llm_safety_classifier.py` | A second "guard" LLM classifies requests as safe / unsafe with a reason | Structured output + agent middleware |

### 🔀 04 – Workflows (`examples/04_workflows/`)

| # | File | What it shows | Key APIs |
|---|------|---------------|----------|
| 17 | `17_custom_executor_workflow.py` | A graph of plain-Python steps and agents with shared state | `WorkflowBuilder`, `Executor`, `@handler`, `@executor` |
| 18 | `18_conditional_routing.py` | An LLM classifier routes each ticket to the right specialist branch | `add_switch_case_edge_group`, `Case`, `Default` |

### 🤝 05 – Multi-agent systems (`examples/05_multi_agent/`)

| # | File | What it shows | Key APIs |
|---|------|---------------|----------|
| 19 | `19_sequential_pipeline.py` | Researcher → Writer → Editor assembly line | `SequentialBuilder` |
| 20 | `20_concurrent_experts.py` | Fan-out to several experts in parallel, merge with a custom aggregator | `ConcurrentBuilder`, `with_aggregator` |
| 21 | `21_group_chat_debate.py` | Optimist / Skeptic / Pragmatist debate, steered by an LLM moderator | `GroupChatBuilder`, `orchestrator_agent` |
| 22 | `22_handoff_support.py` | Triage agent hands off to billing / tech agents; asks the human when needed | `HandoffBuilder`, `request_info` events |

### ⚖️ 06 – Judge agents & capstone (`examples/06_judge/`)

| # | File | What it shows | Key APIs |
|---|------|---------------|----------|
| 23 | `23_best_of_n_judge.py` | N writers produce candidates concurrently; a judge scores them with a rubric and picks a winner | `ConcurrentBuilder` + custom aggregator judge, structured verdicts |
| 24 | `24_reflection_loop_judge.py` | Generator ⇄ judge loop that revises until the judge approves (bounded iterations) | Workflow with a back-edge, `max_iterations` |
| 25 | `25_capstone_guarded_mas.py` | **Everything together:** input guardrail → safety classifier → research/write MAS → judge quality gate → output guardrail | Workflow + middleware + structured output |

---

## Key concepts explained

| Concept | One-line explanation | First seen in |
|---------|----------------------|---------------|
| **Chat client** | The connection to the LLM. Built once in `common/config.py` from your `.env`. | all |
| **Agent** | An LLM + instructions + optional tools/middleware. Call `await agent.run("...")`. | 01 |
| **Tool** | A Python function the model may call. Type hints + `Field(description=...)` become the JSON schema the model sees. | 03 |
| **Session** | Holds conversation history so follow-up questions have context. Can be serialized. | 04, 12 |
| **Structured output** | Pass a Pydantic model as `response_format`; read the parsed object from `response.value`. | 05 |
| **Agent middleware** | Code that runs around a whole agent run (before and after). Can block, rewrite, or log. | 07 |
| **Function middleware** | Code that runs around each tool call. Ideal for caching, retries, and tool guardrails. | 08, 15 |
| **Context provider** | Injects extra instructions or memory before each run and learns from the result afterwards. | 09 |
| **Human-in-the-loop** | Tools marked `approval_mode="always_require"` pause the run until a human approves. | 10, 22 |
| **Workflow** | A directed graph of executors (Python steps or agents) connected by edges, optionally conditional. | 17, 18 |
| **Orchestrations** | Ready-made multi-agent patterns: Sequential, Concurrent, Group Chat, Handoff. | 19 – 22 |
| **Judge agent** | An agent that evaluates other agents' output against a rubric and returns a structured verdict. | 23 – 25 |

---

## Guardrails – defence in depth

The reusable building blocks live in [`common/guardrails.py`](common/guardrails.py):

| Layer | Where | Protects against | Example |
|-------|-------|------------------|---------|
| **Input** | Agent middleware before the LLM | Prompt injection / jailbreaks, banned topics, leaking PII (emails, cards, SSNs, phones, API keys) to the model | 13 |
| **Output** | Agent middleware after the LLM | PII in answers, banned content, overly long responses | 14 |
| **Tool** | Function middleware around each tool call | Dangerous arguments, runaway loops, rate-limit abuse | 15 |
| **Semantic** | A second LLM acting as a classifier | Harmful requests that simple regex rules miss | 16 |
| **Quality gate** | Judge agent | Low-quality, off-topic or ungrounded answers | 23 – 25 |

> ⚠️ The regex and keyword rules are deliberately simple so they are easy to learn from. For production, combine
> them with services such as **Azure AI Content Safety / Prompt Shields** and your organization's policies.

---

## Multi-agent systems and the judge pattern

* **Sequential (19):** each agent builds on the previous one's output. Good for draft → edit pipelines.
* **Concurrent (20):** independent experts work in parallel; an aggregator merges their answers.
* **Group chat (21):** agents take turns in a shared conversation; a moderator (LLM) picks who speaks next and
  when to stop.
* **Handoff (22):** agents transfer control to each other (triage → specialist) and can ask the human for input.
* **Judge (23 – 25):** a dedicated evaluator scores outputs with a rubric (structured output, so scores are
  machine-readable). It is used to **select** the best candidate (23), to **drive revisions** until quality is
  high enough (24), or as the **final quality gate** in a guarded MAS (25). Loops are always bounded by a maximum
  number of iterations, so the cost stays predictable.

---

## Configuration reference

All settings are read from `.env` (or real environment variables, which take precedence) by
[`common/config.py`](common/config.py).

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_API_KEY` | ✅ | – | Your Azure OpenAI key |
| `AZURE_OPENAI_ENDPOINT` | ✅ | – | e.g. `https://my-resource.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | ❌ | `gpt-4o-mini` | Name of **your** model deployment |
| `AZURE_OPENAI_API_VERSION` | ❌ | `2024-10-21` | Azure OpenAI REST API version |
| `AUTO_APPROVE` | ❌ | *(unset)* | Example 10: set to `yes` / `no` to answer approval prompts automatically (useful with `run_example.py all`) |
| `INTERACTIVE` | ❌ | *(unset)* | Example 22: set to `1` to type your own replies to the support agents instead of the scripted ones |

**Want to use a different provider?** Only `get_chat_client()` in `common/config.py` needs to change. Every
example asks that one function for its client.

---

## Project structure

```
microsoft-agentic-framework/
├── README.md                  ← you are here
├── requirements.txt           ← agent-framework + python-dotenv
├── .env.example               ← copy to .env and add your key + endpoint
├── .gitignore                 ← keeps .env, venvs and caches out of git
├── run_example.py             ← launcher: --list | <number> | all
├── common/
│   ├── __init__.py            ← re-exports the helpers below
│   ├── config.py              ← loads .env, builds the Azure OpenAI chat client
│   ├── guardrails.py          ← PII / injection / topic rules + guardrail middleware
│   └── utils.py               ← pretty-printing helpers for agents and workflows
└── examples/
    ├── 01_beginner/           ← 01 – 06
    ├── 02_intermediate/       ← 07 – 12
    ├── 03_guardrails/         ← 13 – 16
    ├── 04_workflows/          ← 17 – 18
    ├── 05_multi_agent/        ← 19 – 22
    └── 06_judge/              ← 23 – 25
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `[configuration error] Missing: AZURE_OPENAI_API_KEY ...` | `.env` doesn't exist or still has placeholder values. Copy `.env.example` to `.env` and fill it in. |
| `401 Unauthorized` / `Access denied` | Wrong key, or a key from a different resource than the endpoint. |
| `404 DeploymentNotFound` / `Resource not found` | `AZURE_OPENAI_DEPLOYMENT` must match the **deployment name** in Azure AI Foundry, not necessarily the model name. Also make sure the endpoint is the resource root (`https://<name>.openai.azure.com/`). |
| `response_format` / `json_schema` not supported | Use a model and API version that support structured outputs (`gpt-4o-mini` / `gpt-4o` with `2024-10-21` or newer). |
| `Unsupported parameter: 'temperature'` | Reasoning models (o1/o3/o4-mini, gpt-5 family) don't accept temperature. Use `gpt-4o-mini` or remove the `temperature` option (example 02 / 06). |
| `429 Too Many Requests` | Your deployment's tokens-per-minute quota is too low. Raise it in Azure AI Foundry, or run examples one at a time. |
| Log lines like `Function failed. Error: ...` | **Expected** in examples 08 and 15. They demonstrate retries and tool guardrails, and the framework logs each failed tool call. |
| `ModuleNotFoundError: agent_framework` | Activate your virtual environment and run `pip install -r requirements.txt`. |

---

## Publishing this as its own GitHub repository

This folder is self-contained. To turn it into a stand-alone repo called **"Microsoft Agentic Framework"**:

```bash
# 1. Create an empty repo on GitHub, e.g. "microsoft-agentic-framework" (no README / .gitignore)
# 2. Copy this folder somewhere and push it
cp -r microsoft-agentic-framework ~/microsoft-agentic-framework
cd ~/microsoft-agentic-framework
git init -b main
git add .
git commit -m "Microsoft Agentic Framework: 25 examples"
git remote add origin https://github.com/<your-user>/microsoft-agentic-framework.git
git push -u origin main
```

`.env` is in `.gitignore`, so your key is never pushed.

---

## Further reading

* Microsoft Agent Framework on GitHub: <https://github.com/microsoft/agent-framework>
* Documentation: <https://learn.microsoft.com/agent-framework/>
* Azure OpenAI quickstart: <https://learn.microsoft.com/azure/ai-services/openai/quickstart>
* Azure AI Content Safety (production guardrails): <https://learn.microsoft.com/azure/ai-services/content-safety/>
