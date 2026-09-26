# Contributing

Use a feature branch and include a focused test with every logic change. Do not tune against one well alone. Any threshold change must document its physical meaning, units, evidence, and regression results on all available validation wells. Do not commit proprietary raw data, credentials, or generated multi-megabyte dashboards.

Run before opening a pull request:

```bash
python -m compileall -q src
pytest -q
```

