# Can Coding Agents Optimize Algorithms Autonomously?

[Tengxiao Liu](https://tengxiaoliu.github.io/about/) &nbsp; [Yuqing Yang](https://ayyyq.github.io/) &nbsp; [Xi Ye](https://xiye17.github.io/) &nbsp; [Danqi Chen](https://www.cs.princeton.edu/~danqic/)

**Blog post**: [https://tengxiaoliu.github.io/autoevolver/](https://tengxiaoliu.github.io/autoevolver/)

## Repository Structure

### Tasks

`tasks/` contains the initial setup files we provided to Claude Code for each problem. Each task directory includes:

- `initial.py` — the seed program the agent starts from
- `evaluate.py` — evaluator
- `prompt.py` — task description
- `INSTRUCTION.md` — instructions given to the agent

```
tasks/
├── ac1/                  # First Autocorrelation Inequality
├── circle_packing/       # Circle Packing (n=26)
└── erdos_min_overlap/    # Erdos Minimum Overlap
```

### Results

`results/` contains the best solutions produced by the agent for each problem.

```
results/
├── ac1/result.json
├── circle_packing/result.json
└── erdos/result.json
```

## Reproducing Evaluation

Each evaluator supports two modes:

**1. Run from a program file** (execute `initial.py` or any evolved variant):

```bash
cd tasks/circle_packing
python evaluate.py --program_path initial.py --results_dir results
```

**2. Validate a pre-computed result** (verify solutions in `results/`):

```bash
cd tasks/circle_packing
python evaluate.py --from-result ../../results/circle_packing/result.json --results_dir results
```

Both modes produce `metrics.json` and `correct.json` in the specified `--results_dir`.

## Acknowledgements

Our task definitions and evaluation scripts were adapted from [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) and [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve).

Conversation trajectories were captured and exported using [DataClaw](https://github.com/peteromallet/dataclaw).

All experiments were conducted using [Claude Code](https://code.claude.com/docs/en/overview).
