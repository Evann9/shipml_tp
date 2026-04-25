param(
  [string]$Repo = "Evann9/Team-Project-ML-",
  [string]$Tag = "shipml-model-artifacts-v1",
  [string]$Token = $env:GITHUB_TOKEN
)

$ErrorActionPreference = "Stop"

$headers = @{
  Accept = "application/vnd.github+json"
  "X-GitHub-Api-Version" = "2022-11-28"
  "User-Agent" = "ShipML-Release-Downloader"
}
if ($Token) {
  $headers.Authorization = "Bearer $Token"
}

$assetMap = @{
  "ship_type_classifier_group_split.joblib" = "shipml/type_anal/outputs/ship_type_classifier_group_split.joblib"
  "ship_type_classifier_tuned_group_split.joblib" = "shipml/type_anal/outputs/ship_type_classifier_tuned_group_split.joblib"
  "ship_type_classifier_row_split_legacy.joblib" = "shipml/type_anal/outputs/ship_type_classifier_row_split_legacy.joblib"
  "future_position_regressor.joblib" = "shipml/route_anal/outputs/future_position_regressor.joblib"
  "ship_anal_model.joblib" = "shipml/route_anal/ship_anal_model.joblib"
}

$release = Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/$Repo/releases/tags/$Tag" -Headers $headers
$assets = Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/$Repo/releases/$($release.id)/assets?per_page=100" -Headers $headers

foreach ($assetName in $assetMap.Keys) {
  $asset = $assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1
  if (-not $asset) {
    Write-Warning "Release asset not found: $assetName"
    continue
  }

  $target = $assetMap[$assetName]
  $targetDir = Split-Path -Parent $target
  if ($targetDir -and -not (Test-Path -LiteralPath $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir | Out-Null
  }
  Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $target
  Write-Host "Downloaded: $target"
}
