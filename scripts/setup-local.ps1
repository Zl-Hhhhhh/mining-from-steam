param(
    [string]$PythonExecutable,
    [string]$PythonVersion = '3.13'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    $bootstrapArgs = @()
    if ($PythonExecutable) {
        $bootstrap = (Get-Command -Name $PythonExecutable -ErrorAction Stop).Source
    } else {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        $localLauncher = Join-Path $env:LOCALAPPDATA 'Programs\Python\Launcher\py.exe'
        $localPython = Join-Path $env:LOCALAPPDATA ('Programs\Python\Python' + $PythonVersion.Replace('.', '') + '\python.exe')
        if ($launcher) {
            $bootstrap = $launcher.Source
            $bootstrapArgs = @("-$PythonVersion")
        } elseif (Test-Path -LiteralPath $localPython) {
            $bootstrap = $localPython
        } elseif (Test-Path -LiteralPath $localLauncher) {
            $bootstrap = $localLauncher
            $bootstrapArgs = @("-$PythonVersion")
        } else {
            $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if (-not $pythonCommand -or $pythonCommand.Source -like '*\WindowsApps\*') {
                throw 'Python was not found. Install Python 3.13 or pass -PythonExecutable with its full path.'
            }
            $bootstrap = $pythonCommand.Source
        }
    }
    & $bootstrap @bootstrapArgs -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ is required'"
    if ($LASTEXITCODE -ne 0) { throw 'Python version check failed.' }
    & $bootstrap @bootstrapArgs -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}

& $venvPython -c "import site, sys; assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE, 'An isolated virtual environment is required'; print(sys.executable); print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw 'Virtual environment isolation check failed.' }
& $venvPython -m pip install --disable-pip-version-check --cache-dir (Join-Path $projectRoot '.cache\pip') -r (Join-Path $projectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Project dependency check failed.' }
Write-Output 'Environment ready. Use .\.venv\Scripts\python.exe from the project root.'
