# Launch Jupyter Lab with the tinyship env (bypasses broken conda activate on some Windows setups).
$Tinyship = "C:\Users\Arbi\.conda\envs\tinyship"
$env:Path = "$Tinyship;$Tinyship\Scripts;$Tinyship\Library\bin;" + $env:Path
Set-Location "D:\TinyShip-Finger"
Write-Host "Python:" (& "$Tinyship\python.exe" --version)
Write-Host "Starting Jupyter Lab..."
& "$Tinyship\Scripts\jupyter.exe" lab notebooks
