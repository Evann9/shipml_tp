# ShipML Web Map

Run from the repository root:

```powershell
C:\Users\green\anaconda3\envs\myproject\python.exe -m flask --app shipml.web.app:app run --host 127.0.0.1 --port 5000 --no-reload
```

Then open:

```text
http://127.0.0.1:5000
```

The web app reads these generated files when available:

- `shipml/route_anal/outputs/route_predictions_with_types.csv`
- `shipml/route_anal/outputs/future_position_forecast.csv`
- `shipml/type_anal/outputs/ship_type_classifier_group_split_metrics.json`
- `shipml/type_anal/outputs/ship_type_classifier_class_metrics.csv`
- `shipml/type_anal/outputs/ship_type_classifier_confusion_pairs.csv`

Large `.joblib` model files are not tracked by Git. Download them from the GitHub Release before rerunning model-dependent scripts:

```powershell
.\scripts\download_model_artifacts_from_github_release.ps1
```
