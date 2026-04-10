# Memory DB Template

Use one compact entry per major step:

```markdown
## <timestamp> | <trial_id> | <phase>
Decision:
- <what was decided>
Evidence:
- <metric/log/checkpoint>
Artifacts:
- <path-or-id>
Blockers:
- <none or blocker>
Next:
- <single next action>
```

Compression rules:
- <= 12 lines per entry
- prefer IDs/paths over prose
