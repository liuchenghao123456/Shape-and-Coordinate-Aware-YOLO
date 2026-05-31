$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = "python"
$ultraRoot = & $python -c "import pathlib, ultralytics; print(pathlib.Path(ultralytics.__file__).resolve().parent)"

$copies = @(
    @{ Source = Join-Path $repoRoot "patches\ultralytics\nn\modules\block.py"; Target = Join-Path $ultraRoot "nn\modules\block.py" },
    @{ Source = Join-Path $repoRoot "patches\ultralytics\nn\modules\__init__.py"; Target = Join-Path $ultraRoot "nn\modules\__init__.py" },
    @{ Source = Join-Path $repoRoot "patches\ultralytics\nn\tasks.py"; Target = Join-Path $ultraRoot "nn\tasks.py" },
    @{ Source = Join-Path $repoRoot "patches\ultralytics\utils\loss.py"; Target = Join-Path $ultraRoot "utils\loss.py" },
    @{ Source = Join-Path $repoRoot "patches\ultralytics\utils\metrics.py"; Target = Join-Path $ultraRoot "utils\metrics.py" }
)

foreach ($item in $copies) {
    Copy-Item -Force $item.Source $item.Target
    Write-Host "Copied $($item.Source) -> $($item.Target)"
}

Write-Host "Ultralytics patch applied successfully."
