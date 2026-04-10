# Memory DB Template

Use one entry per major step.

```markdown
## <timestamp> | <task-id> | <phase>

Decision:
- <what was decided>

Evidence:
- <metric/log/checkpoint>

Artifacts:
- <path-or-id-1>
- <path-or-id-2>

Blockers:
- <none or blocker details>

Next:
- <single next action>
```

## Compression Rules

- Keep each entry <= 12 lines.
- Prefer IDs/paths over verbose prose.
- Link to logs instead of pasting large outputs.
