param([switch]$SkipGitHub, [string]$PythonVersion = '3.13')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$sources = Get-Content -LiteralPath 'config/sources.json' -Raw | ConvertFrom-Json
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & py "-$PythonVersion" -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python venv creation failed; Python 3.11+ is required.' }
}
if (-not $SkipGitHub) {
    New-Item -ItemType Directory -Path external -Force | Out-Null
    if (-not (Test-Path -LiteralPath 'external/download-steam-reviews')) {
        & git clone --depth 1 https://github.com/woctezuma/download-steam-reviews.git external/download-steam-reviews
        if ($LASTEXITCODE -ne 0) { throw 'GitHub clone failed.' }
    }
    $installedCommit = (& git -C external/download-steam-reviews rev-parse HEAD).Trim()
    if ($installedCommit -ne $sources.github.commit) {
        & git -C external/download-steam-reviews fetch --depth 1 origin $sources.github.commit
        if ($LASTEXITCODE -ne 0) { throw 'Pinned GitHub revision fetch failed.' }
        & git -C external/download-steam-reviews checkout --detach $sources.github.commit
        if ($LASTEXITCODE -ne 0) { throw 'Pinned GitHub revision checkout failed.' }
    }
    & ./.venv/Scripts/python.exe -m pip install ./external/download-steam-reviews
    if ($LASTEXITCODE -ne 0) { throw 'steamreviews installation failed.' }
    & git -C external/download-steam-reviews rev-parse HEAD | Set-Content -Encoding utf8 external/steamreviews-commit.txt
    $frozenPackages = & ./.venv/Scripts/python.exe -m pip freeze
    $frozenPackages -replace '^steamreviews @.*$', 'steamreviews==0.9.6.1' | Set-Content -Encoding utf8 requirements-lock.txt
}
Write-Output 'Setup complete. Run ./.venv/Scripts/python.exe scripts/steam_data.py collect'
