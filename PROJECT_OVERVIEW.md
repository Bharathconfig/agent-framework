# PROJECT_OVERVIEW

## Project Summary
Microsoft Agent Framework (Python) is a modular framework for building, orchestrating, hosting, and integrating AI agents across multiple model providers and runtime environments. The Python workspace is organized as a multi-package monorepo, where each package contributes a capability such as core agent abstractions, orchestration patterns, model connectors, hosting adapters, memory providers, or user-facing tooling.

## Architecture Overview
The architecture follows a layered, composable model:

1. **Core abstractions** define agent contracts, messages, tool invocation patterns, and execution context.
2. **Declarative and orchestration layers** compose core abstractions into workflows and multi-agent topologies.
3. **Model/provider adapters** connect the framework to external LLM services.
4. **Hosting adapters** expose agents through transport and runtime environments.
5. **Memory/storage adapters** provide state, context retention, and retrieval support.
6. **UI and developer tooling** improve local development, diagnostics, and interaction.

## Package Structure

### Core packages
- **core**: foundational primitives, runtime contracts, messaging, and execution helpers.
- **declarative**: declarative configuration and composition patterns for agents/workflows.
- **orchestrations**: workflow orchestration building blocks for multi-step and multi-agent systems.
- **tools**: tool integration primitives and reusable tool abstractions.

### AI Model Integration packages
- **openai**: integration with OpenAI models/services.
- **anthropic**: integration with Anthropic model APIs.
- **claude**: Claude-specific integration surface.
- **gemini**: integration with Google Gemini models.
- **bedrock**: AWS Bedrock model provider integration.
- **mistral**: Mistral model integration.
- **ollama**: local/hosted Ollama model integration.
- **azure-ai-search**: Azure AI Search integration support for retrieval scenarios.

### Hosting packages
- **hosting**: base hosting/runtime integrations.
- **hosting-a2a**: agent-to-agent hosting and communication adapters.
- **hosting-mcp**: Model Context Protocol hosting integration.
- **hosting-telegram**: Telegram channel hosting adapter.
- **hosting-responses**: response-oriented hosting flows.
- **foundry_hosting**: hosting capabilities specific to Foundry scenarios.

### Memory & Storage packages
- **redis**: Redis-backed memory/state components.
- **azure-cosmos**: Azure Cosmos DB integration.
- **mem0**: memory adapter integration for memory-centric agent behavior.
- **azure-cosmos-memory**: Cosmos-backed memory specialization.

### UI & Development packages
- **ag-ui**: agent user interface integration surface.
- **devui**: local developer UI and diagnostics support.
- **chatkit**: chat-oriented interaction components.

### Other specialized packages
- **foundry**: Azure AI Foundry related integrations and abstractions.
- **foundry_local**: local Foundry development/runtime support.
- **hyperlight**: specialized runtime and execution support.
- **lab**: experimental and prototyping features.
- **monty**: package-specific extension utilities.
- **a2a**: agent-to-agent communication primitives.
- **copilotstudio**: Copilot Studio ecosystem integration.
- **github_copilot**: GitHub Copilot interaction/integration support.
- **purview**: governance/compliance related integrations.
- **azure-contentunderstanding**: Azure content understanding integration.

## Key Components
Common component patterns across the repository include:

- **Agent interfaces and base classes** for implementing custom agents.
- **Message and context models** to pass user/system/tool state through execution pipelines.
- **Tool contracts** that expose callable capabilities to agents.
- **Orchestration coordinators** that manage sequencing, branching, and multi-agent collaboration.
- **Provider clients/adapters** that translate framework-native requests to model-vendor APIs.
- **Hosting endpoints/runtimes** that package and serve agents through protocols/channels.
- **Memory providers** for persistent or session-scoped context.

## How It Works
1. Define an agent using core abstractions.
2. Configure model provider bindings (for example OpenAI, Gemini, Claude, Bedrock, or Mistral).
3. Attach tools and memory providers as needed.
4. Compose behavior with orchestration/declarative packages.
5. Expose the agent through a hosting package (MCP, Telegram, A2A, or default hosting).
6. Validate behavior with tests/samples and iterate using dev tooling.
7. Deploy the resulting runtime in the target environment.

## Integration Points
- **Core ↔ Model Adapters**: provider packages consume core message/context contracts.
- **Core/Tools ↔ Orchestrations**: orchestrators invoke tool and agent actions through core contracts.
- **Orchestrations ↔ Hosting**: hosted runtimes call orchestrated flows as request handlers.
- **Memory Providers ↔ Core/Orchestrations**: memory layers enrich context and persist interaction history.
- **UI/Dev packages ↔ Hosting/Core**: debugging and interactive surfaces connect to running agents.

## Usage Patterns
- **Single-agent chat flow** with one model adapter and optional toolset.
- **Tool-augmented agent** where external tools are invoked as part of reasoning.
- **Multi-agent orchestration** for decomposition, routing, and collaboration.
- **Retrieval-augmented generation** via search/storage integrations.
- **Hosted agent applications** exposed through MCP, Telegram, or other channels.

## Directory Structure

```text
python/
├── AGENTS.md                        # Agent development instructions and guidance
├── README.md                        # Python workspace overview and quickstart
├── CHANGELOG.md                     # Change history across Python workspace
├── DEV_SETUP.md                     # Local development environment setup
├── CODING_STANDARD.md               # Coding conventions and standards
├── pyproject.toml                   # Python project/workspace metadata
├── uv.lock                          # Locked dependency graph for reproducibility
├── packages/                        # Monorepo package collection
│   ├── core/                        # Core runtime abstractions
│   ├── declarative/                 # Declarative composition layer
│   ├── orchestrations/              # Workflow and orchestration primitives
│   ├── tools/                       # Tool interfaces and implementations
│   ├── openai/ anthropic/ claude/
│   │   gemini/ bedrock/ mistral/
│   │   ollama/                      # Model provider integrations
│   ├── hosting/ hosting-a2a/
│   │   hosting-mcp/ hosting-telegram/
│   │   hosting-responses/           # Hosting adapters
│   ├── redis/ azure-cosmos/
│   │   mem0/ azure-cosmos-memory/   # Memory and storage providers
│   ├── ag-ui/ devui/ chatkit/       # UI and developer tooling
│   └── foundry/ hyperlight/ lab/
│       monty/ a2a/ copilotstudio/
│       github_copilot/ purview/
│       azure-contentunderstanding/  # Specialized integrations
├── samples/                         # End-to-end and focused usage samples
├── tests/                           # Test suites and sample validations
├── scripts/                         # Build/test/dev helper scripts
└── agent_framework_meta/            # Workspace metadata and package-level config
```

## Notes on This Repository Version
- The `python/` directory has been imported while preserving upstream organization.
- Python source files have been augmented with module and callable docstrings in Google-style format to improve readability and onboarding.
