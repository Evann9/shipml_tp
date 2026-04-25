param(
  [string]$Repo = "Evann9/Team-Project-ML-",
  [string]$Tag = "shipml-model-artifacts-v1",
  [string]$ReleaseName = "ShipML Model Artifacts v1",
  [string]$Token = $env:GITHUB_TOKEN
)

$ErrorActionPreference = "Stop"

if (-not $Token) {
  throw "Set GITHUB_TOKEN first. Example: `$env:GITHUB_TOKEN='ghp_...'"
}

$assets = @(
  "shipml/type_anal/outputs/ship_type_classifier_group_split.joblib",
  "shipml/type_anal/outputs/ship_type_classifier_tuned_group_split.joblib",
  "shipml/type_anal/outputs/ship_type_classifier_row_split_legacy.joblib",
  "shipml/route_anal/outputs/future_position_regressor.joblib",
  "shipml/route_anal/ship_anal_model.joblib"
)

$headers = @{
  Authorization = "Bearer $Token"
  Accept = "application/vnd.github+json"
  "X-GitHub-Api-Version" = "2022-11-28"
  "User-Agent" = "ShipML-Release-Uploader"
}

try {
  $release = Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/$Repo/releases/tags/$Tag" -Headers $headers
  Write-Host "Release exists: $($release.html_url)"
} catch {
  $body = @{
    tag_name = $Tag
    target_commitish = "main"
    name = $ReleaseName
    body = "Large ShipML model artifacts. Download these files into the same local paths when running the project."
    draft = $false
    prerelease = $false
  } | ConvertTo-Json
  $release = Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$Repo/releases" -Headers $headers -ContentType "application/json" -Body $body
  Write-Host "Release created: $($release.html_url)"
}

$uploadBase = $release.upload_url -replace "\{.*$", ""
$existingAssets = Invoke-RestMethod -Method Get -Uri "https://api.github.com/repos/$Repo/releases/$($release.id)/assets?per_page=100" -Headers $headers

foreach ($path in $assets) {
  if (-not (Test-Path -LiteralPath $path)) {
    Write-Warning "Missing asset: $path"
    continue
  }

  $name = Split-Path -Leaf $path
  $old = $existingAssets | Where-Object { $_.name -eq $name } | Select-Object -First 1
  if ($old) {
    Invoke-RestMethod -Method Delete -Uri "https://api.github.com/repos/$Repo/releases/assets/$($old.id)" -Headers $headers | Out-Null
    Write-Host "Replaced existing asset: $name"
  }

  $uri = "$uploadBase?name=$([uri]::EscapeDataString($name))"
  $curlArgs = @(
    "-sS", "-L", "-X", "POST",
    "-H", "Authorization: Bearer $Token",
    "-H", "Accept: application/vnd.github+json",
    "-H", "Content-Type: application/octet-stream",
    "--data-binary", "@$path",
    $uri
  )
  $result = & curl.exe @curlArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Upload failed: $name"
  }
  if ($result -notmatch '"state"\s*:\s*"uploaded"') {
    throw "GitHub did not confirm upload for $name. Response: $result"
  }
  Write-Host "Uploaded: $name"
}

Write-Host "Release URL: $($release.html_url)"
