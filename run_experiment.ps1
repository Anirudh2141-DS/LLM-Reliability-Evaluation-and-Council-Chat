# RLRGF - Run Evaluation Experiment
# This script runs the full reliability evaluation pipeline using synthetic data.

Write-Host "Starting RLRGF Reliability Evaluation..." -ForegroundColor Cyan

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path "./output"

# Run the Python experiment runner
# Set environment variables
$env:PYTHONPATH = ".;./python;C:\Users\aniru\AppData\Roaming\Python\Python312\site-packages"
$env:PYTHONIOENCODING = "utf-8"
python -m rlrgf.run_experiment --experiment-id local_test_001 --output-dir ./output --n-normal 20 --n-adversarial 10

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nVisualization Results:" -ForegroundColor Green
    Get-ChildItem "./output/visualizations" | Select-Object Name
    
    Write-Host "`nMetrics Summary:" -ForegroundColor Green
    Get-Content "./output/experiment_metrics.json" | ConvertFrom-Json | Format-List
    
    Write-Host "`nExperiment complete. Check the ./output directory for full reports." -ForegroundColor Green
}
else {
    Write-Host "`nExperiment failed. Check for missing dependencies." -ForegroundColor Red
}
