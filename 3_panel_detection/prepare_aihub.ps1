# AI-Hub 데이터 원클릭 압축 해제 및 COCO 변환 스크립트 (PowerShell)
param (
    [string]$OutputDir = "",
    [switch]$SkipUnzip
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$AihubDir = (Resolve-Path "../data/aihub").Path
$ZipsDir = Join-Path $AihubDir "zips"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $AihubDir "coco_256"
}

function Expand-ArchiveSafe($zipPath, $targetDir, $label) {
    if (-not (Test-Path $zipPath)) {
        Write-Host "[!] $label not found at $zipPath (skipping)" -ForegroundColor Yellow
        return
    }

    if ((Test-Path $targetDir) -and (Get-ChildItem $targetDir -Force | Measure-Object).Count -gt 0) {
        Write-Host "[✓] $label already extracted to $targetDir" -ForegroundColor Green
    } else {
        Write-Host "[*] Extracting $label..." -ForegroundColor Cyan
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        Expand-Archive -Path $zipPath -DestinationPath $targetDir -Force
        Write-Host "    Extraction complete." -ForegroundColor Green
    }
}

if (-not $SkipUnzip) {
    Write-Host "=== Step 1: Extracting AI-Hub zip archives ===" -ForegroundColor Magenta
    Expand-ArchiveSafe (Join-Path $ZipsDir "train_images.zip") (Join-Path $AihubDir "train_data") "Train Images (TIFF)"
    Expand-ArchiveSafe (Join-Path $ZipsDir "train_labels.zip") (Join-Path $AihubDir "train_label") "Train Labels (JSON)"
    Expand-ArchiveSafe (Join-Path $ZipsDir "train_meta.zip")   (Join-Path $AihubDir "train_meta")  "Train Metadata"
    Expand-ArchiveSafe (Join-Path $ZipsDir "valid_images.zip") (Join-Path $AihubDir "valid_data") "Validation Images (TIFF)"
    Expand-ArchiveSafe (Join-Path $ZipsDir "valid_labels.zip") (Join-Path $AihubDir "valid_label") "Validation Labels (JSON)"
    Expand-ArchiveSafe (Join-Path $ZipsDir "valid_meta.zip")   (Join-Path $AihubDir "valid_meta")  "Validation Metadata"
    Write-Host ""
}

Write-Host "=== Step 2: Running prepare_aihub (COCO 256px conversion) ===" -ForegroundColor Magenta
python -u -m src.prepare_aihub --aihub-root $AihubDir --output $OutputDir
exit $LASTEXITCODE
