# Register tinyship as a Jupyter kernel (run once).
$Py = "C:\Users\Arbi\.conda\envs\tinyship\python.exe"
& $Py -m ipykernel install --user --name tinyship --display-name "Python (tinyship)"
Write-Host "Kernel registered: Python (tinyship)"
