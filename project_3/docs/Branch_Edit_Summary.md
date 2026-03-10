# Branch Edit Summary
## Branch: experiments/EXP-002-Refine-DAG
- DAG Refinement: Replaced hardcoded dependency nodes in `DAG.py` with dynamic AST parsing logic. The script now reads the Python files in the directory automatically, detects file contents including top-level classes, functions, variables, and cross-references their imports to programmatically generate an accurate component diagram.
- DAG Visual Polish: Implemented dynamic container heights, color styling, nested element boxes, custom layout parameters, and detailed legend for the new module graph format. The output is now saved dynamically as `dag.png` in the project root.
- Branching Workflow: Created `docs/branching_workflow.md` to define a systematic, step-by-step approach to running git-based experimental iterations on the project.
