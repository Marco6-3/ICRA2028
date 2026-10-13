param(
    [ValidateSet('pilot', 'coordinates', 'analyze')]
    [string]$Mode = 'pilot',
    [string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONUTF8 = '1'
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:PYTHONUNBUFFERED = '1'
$python = 'C:\isaacsim_env\Scripts\python.exe'
if (!$OutputDirectory) {
    if ($Mode -eq 'analyze') { throw 'Specify an existing OutputDirectory for analyze mode.' }
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutputDirectory = Join-Path $PSScriptRoot "outputs\${Mode}_$stamp"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
if ($Mode -ne 'analyze') {
    if (Test-Path -LiteralPath $OutputDirectory) { throw 'Output already exists; preserve the previous run.' }
    $parent = Split-Path -Parent $OutputDirectory
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $runArguments = @(
        (Join-Path $PSScriptRoot 'run_pilot.py'), '--out', $OutputDirectory,
        '--headless', '--device', 'cpu',
        '--kit_args=--/exts/omni.kit.registry.nucleus/cacheCreateLinks=false'
    )
    if ($Mode -eq 'pilot') {
        $runArguments += @('--load-frame', 'reference', '--reference-from-finger', '2')
    } else {
        & $python (Join-Path $PSScriptRoot 'make_coordinate_matrix.py')
        if ($LASTEXITCODE -ne 0) { throw 'Matrix generation failed.' }
        $runArguments += @('--matrix', (Join-Path $PSScriptRoot 'coordinate_matrix.json'),
            '--reference-wxyz', '.7071067811865476', '0', '0', '.7071067811865476')
    }
    $logPath = $OutputDirectory + '.log'
    if (Test-Path -LiteralPath $logPath) { throw 'Log already exists.' }
    Write-Output "Running E1-S; log: $logPath"
    & $python @runArguments *> $logPath
}
$manifestPath = Join-Path $OutputDirectory 'manifest.json'
if (!(Test-Path -LiteralPath $manifestPath)) { throw 'Simulation did not save a manifest.' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($manifest.status -ne 'completed') { throw "Simulation did not complete: $($manifest.errors)" }
& $python (Join-Path $PSScriptRoot 'analyze_pilot.py') $OutputDirectory
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed.' }
$analysis = Get-Content -LiteralPath (Join-Path $OutputDirectory 'analysis.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ($analysis.status -ne 'validated') { throw 'Data checks did not pass.' }
Write-Output (Join-Path $OutputDirectory 'REPORT.md')
