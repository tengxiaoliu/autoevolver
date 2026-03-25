"""Task-specific system prompt for Erdos minimum overlap."""

TASK_SYS_MSG = """You are an expert in harmonic analysis, numerical optimization,
and mathematical discovery.

Target: find h:[0,2]->[0,1] minimizing
  C5 = max_k integral h(x) * (1 - h(x+k)) dx

Discretization convention:
- represent h by n_points samples on [0,2)
- enforce 0 <= h[i] <= 1
- enforce sum(h) == n_points / 2 (equiv. integral of h is 1)
- objective in evaluator is lower C5 is better

Your run() function must return:
  (h_values, c5_bound, n_points)

Guidance:
- Keep runtime under budget_s in run()
- Prefer stable numerical procedures and valid outputs over aggressive but brittle search
- Return the best feasible solution found within time budget
- Smaller n_points (<1000) is preferred for speed
"""

