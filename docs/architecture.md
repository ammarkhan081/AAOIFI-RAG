# Architecture and phase map

This structure maps one-to-one to the plan's six layers without pre-implementing any of them.

| Layer | Future source location | Configuration and retained artifacts | Primary phase |
| --- | --- | --- | --- |
| Data | `src/aaoifi_rag/data/` | `data/manifests/`, `data/private/`, `schemas/` | 1 - legal/data setup |
| Retrieval | `src/aaoifi_rag/retrieval/` | `configs/retrieval/` | 2 - retrieval prototype |
| Generation | `src/aaoifi_rag/generation/` | `configs/generation/`, `prompts/` | 3 - generation and routing |
| Reliability | `src/aaoifi_rag/reliability/` | `configs/reliability/`, `schemas/` | 3 - generation and routing |
| Orchestration | `src/aaoifi_rag/orchestration/` | `configs/orchestration/`; plain explicit Python state machine | 3 - generation and routing |
| Reporting | `src/aaoifi_rag/reporting/` | `reports/`, `runs/`, `configs/reporting/` | 4 - evaluation and reporting |

## Repository tree

```text
.
|- README.md                         # project boundary and current status
|- pyproject.toml                    # Python compatibility and project metadata
|- requirements.txt                  # pinned direct research dependencies
|- .python-version                   # local interpreter target
|- .gitignore                        # prevents accidental public corpus/trace commits
|- configs/                          # future, versioned experiment configuration
|- data/
|  |- manifests/                     # source, license, and hash metadata (tracked)
|  |- private/                       # licensed corpus, Git-ignored
|  `- evaluation/                    # future approved labels/metadata
|- docs/
|  |- architecture.md
|  |- governance/                    # licensing, security, data handling
|  |- decisions/                     # explicit architectural/research decisions
|  `- annotation/                    # future scholar-approved guides
|- prompts/                          # future versioned, context-only templates
|- schemas/                          # future clause, trace, and annotation schemas
|- src/aaoifi_rag/
|  |- data/
|  |- retrieval/
|  |- generation/
|  |- reliability/
|  |- orchestration/
|  `- reporting/
|- tests/                            # layer-aligned tests, added with implementation
|- runs/                             # complete local traces, Git-ignored
|- artifacts/                        # local indexes and model outputs, Git-ignored
`- reports/                          # curated technical notes and error analyses
```

The separation of `data/manifests/` from `data/private/` is deliberate: it permits reproducibility through source provenance without publishing licensed text. `runs/` is local-only so full trace logging can be retained without exposing protected clauses in public logs.
