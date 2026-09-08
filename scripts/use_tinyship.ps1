# Optional: put the `tinyship` conda env on PATH for this PowerShell session.
# Usage (from repo root):  . .\scripts\use_tinyship.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$candidates = @(
    (Join-Path $env:USERPROFILE ".conda\envs\tinyship"),
    (Join-Path $env:USERPROFILE "miniconda3\envs\tinyship"),
    (Join-Path $env:USERPROFILE "anaconda3\envs\tinyship")
)
if ($env:CONDA_PREFIX) {
    $candidates += (Join-Path (Split-Path $env:CONDA_PREFIX -Parent) "tinyship")
}

$Tinyship = $candidates | Where-Object { Test-Path (Join-Path $_ "python.exe") } | Select-Object -First 1
if (-not $Tinyship) {
    Write-Host "Could not find conda env 'tinyship'. Create it with:"
    Write-Host "  conda env create -f environment.yml"
    Write-Host "Then: conda activate tinyship"
    Set-Location $RepoRoot
    return
}

$env:Path = "$Tinyship;$Tinyship\Scripts;" + $env:Path
$env:CONDA_DEFAULT_ENV = "tinyship"
$env:CONDA_PREFIX = $Tinyship
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
Set-Location $RepoRoot
Write-Host "tinyship ready:" (& "$Tinyship\python.exe" --version)
Write-Host "cwd: $(Get-Location)"
