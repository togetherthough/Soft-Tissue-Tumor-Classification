# PowerShell script to copy results from cluster to local machine
# Usage: .\scripts\copy_results_from_cluster.ps1

$ClusterUser = "r112276"
$ClusterHost = "rad-hpc-master-001"
$ClusterPath = "~/Med3Tab-PFN/results"
$LocalPath = "C:\Users\cahel\Desktop\Med3Tab-PFN\results"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Copying Results from Cluster to Local Machine" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Create local results directories if they don't exist
Write-Host "Creating local directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "$LocalPath\pooling_comparison" | Out-Null
New-Item -ItemType Directory -Force -Path "$LocalPath\preprocessing_comparison" | Out-Null
New-Item -ItemType Directory -Force -Path "$LocalPath\classification_head" | Out-Null
New-Item -ItemType Directory -Force -Path "$LocalPath\classification_head_roi" | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green
Write-Host ""

# Copy pooling comparison results
Write-Host "Copying pooling comparison results..." -ForegroundColor Yellow
scp -r "${ClusterUser}@${ClusterHost}:${ClusterPath}/pooling_comparison/*" "$LocalPath\pooling_comparison\"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Pooling comparison results copied" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to copy pooling comparison results" -ForegroundColor Red
}
Write-Host ""

# Copy preprocessing comparison results
Write-Host "Copying preprocessing comparison results..." -ForegroundColor Yellow
scp -r "${ClusterUser}@${ClusterHost}:${ClusterPath}/preprocessing_comparison/*" "$LocalPath\preprocessing_comparison\"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Preprocessing comparison results copied" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to copy preprocessing comparison results" -ForegroundColor Red
}
Write-Host ""

# Copy classification head results (exp3)
Write-Host "Copying classification head results..." -ForegroundColor Yellow
scp -r "${ClusterUser}@${ClusterHost}:${ClusterPath}/classification_head/*" "$LocalPath\classification_head\"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Classification head results copied" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to copy classification head results (might not exist)" -ForegroundColor Yellow
}
Write-Host ""

# Copy classification head ROI results
Write-Host "Copying classification head ROI results..." -ForegroundColor Yellow
scp -r "${ClusterUser}@${ClusterHost}:${ClusterPath}/classification_head_roi/*" "$LocalPath\classification_head_roi\"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Classification head ROI results copied" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to copy classification head ROI results (might not exist)" -ForegroundColor Yellow
}
Write-Host ""

# List copied files
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Files Copied:" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Get-ChildItem -Path $LocalPath -Recurse -File | ForEach-Object {
    $relativePath = $_.FullName.Substring($LocalPath.Length)
    Write-Host $relativePath
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "✓ Copy Complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Open the visualization notebook:" -ForegroundColor White
Write-Host "   notebooks\visualization\Pooling_Comparison_Results.ipynb" -ForegroundColor Cyan
Write-Host "2. Run all cells to generate plots" -ForegroundColor White
Write-Host ""
