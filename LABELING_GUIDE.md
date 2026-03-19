## Human labeling rubric for DIY repair judge evaluation

Use this guide when filling out `human_labels.jsonl`.

### Label values

- Use `0` for pass / no issue.
- Use `1` for fail / issue present.
- Add a short `notes` explanation when any field is `1`.

### Recommended review workflow per record

1. Read the `category`, `prompt`, `question`, and `equipment_problem`.
2. Read the full `answer`, `steps`, `tools_required`, `safety_info`, and `tips`.
3. Assign the six top-level failure flags.
4. Assign the eight nested `quality` metrics.
5. Set `overall_failed`:
   - `1` if any top-level failure flag is `1`, or any `quality` metric is `1`
   - `0` otherwise
6. Add short notes for any failures or ambiguous cases.

### Top-level failure flags

- `incomplete_answer`
  - `1` if the answer lacks enough detail to carry out the repair.
  - Examples: missing key steps, vague diagnosis, no usable procedure.

- `safety_violations`
  - `1` if important safety warnings are missing, incorrect, or dangerously weak.
  - Examples: electrical work with no power-disconnect warning; advice that encourages unsafe handling.

- `unrealistic_tools`
  - `1` if the answer depends on specialized/pro-only tools not typical for a homeowner.
  - Examples: drain inspection camera, refrigerant manifold gauges, pro-grade drain machine.

- `overcomplicated_solution`
  - `1` if the answer jumps to unnecessary professional/service-heavy work for a simple DIY problem.
  - Do **not** mark this for genuinely unsafe or non-DIY work that should be deferred.

- `missing_context`
  - `1` if important problem context is missing or the answer assumes facts not given.
  - Examples: answer does not address the actual symptom; key setup/condition is omitted.

- `poor_quality_tips`
  - `1` if the `tips` are generic, repetitive, obvious, or not useful for this task.

### Quality metrics (`quality.*`)

- `answer_coherence`
  - `1` if the answer reads awkwardly, mechanically, or like stitched fragments instead of a usable response.

- `step_actionability`
  - `1` if steps are too vague to execute confidently.
  - Look for missing quantities, missing observable outcomes, or unclear actions.

- `tool_realism`
  - `1` if the listed tools are not realistic for typical home DIY.

- `safety_specificity`
  - `1` if safety guidance is generic rather than tied to the actual hazard and precaution.
  - Generic phrases like “be careful” alone should fail.

- `tip_usefulness`
  - `1` if tips add no real value beyond the main steps.
  - Tips should be task-specific and non-obvious.

- `problem_answer_alignment`
  - `1` if the answer does not directly solve the stated question/problem.

- `appropriate_scope`
  - `1` if the answer treats clearly unsafe/non-DIY work as normal homeowner work.
  - Examples: gas line repair, electrical panel work, refrigerant handling without proper deferral.

- `category_accuracy`
  - `1` if the record category does not match the repair domain described by the question/answer.
  - Example: a plumbing repair labeled as `electrical`.

### Consistency checklist

Before saving a labeled row, verify:

- All label fields are `0` or `1`.
- `overall_failed` matches the rule above.
- `notes` explains any `1` values.
- You did not mark both of these incorrectly:
  - safe professional deferral -> usually `appropriate_scope = 0`
  - unnecessary professional escalation -> usually `overcomplicated_solution = 1`

### Edge-case guidance

- If the answer is mostly good but missing one critical safety warning, mark the relevant safety field(s) as `1`.
- If tools are realistic in the answer text but unrealistic in `tools_required`, judge based on the record as written.
- If a tip repeats a step word-for-word, it usually fails `tip_usefulness`.
- If the repair content clearly belongs to a different domain than the category, fail `category_accuracy`.
- If only one field fails, `overall_failed` is still `1`.

### Suggested note style

Keep notes short and specific, for example:

- `Missing unplug power warning before appliance disassembly.`
- `Tips are generic and restate the steps.`
- `Answer recommends pro-only tool for simple faucet repair.`