param([string]$OutputRoot,[string]$FixtureNpz)
$ErrorActionPreference='Stop'
if (!$OutputRoot) { $OutputRoot=Join-Path $PSScriptRoot ('outputs\contact_'+(Get-Date -Format 'yyyyMMdd_HHmmss')) }
if (Test-Path -LiteralPath $OutputRoot) { throw 'Preserve old output; use a new directory.' }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$python='C:\isaacsim_env\Scripts\python.exe'
$env:OMNI_KIT_ACCEPT_EULA='YES'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONUTF8='1'
& $python (Join-Path $PSScriptRoot 'make_contact_matrix.py')
if ($LASTEXITCODE -ne 0) { throw 'Matrix generation failed.' }
foreach ($support in @('table','removed')) {
    foreach ($preload in @('open','proxy','normal-force')) {
        $name="${support}_${preload}"
        $out=Join-Path $OutputRoot $name
        $arguments=@((Join-Path $PSScriptRoot 'contact_pilot.py'),'--out',$out,
            '--matrix',(Join-Path $PSScriptRoot 'contact_matrix.json'),
            '--support',$support,'--preload-mode',$preload,
            '--load-frame','reference','--reference-from-finger','2',
            '--headless','--device','cpu','--kit_args=--/exts/omni.kit.registry.nucleus/cacheCreateLinks=false')
        if ($FixtureNpz) { $arguments+=@('--fixture-npz',$FixtureNpz) }
        & $python @arguments *> (Join-Path $OutputRoot ($name+'.log'))
        $manifestPath=Join-Path $out 'manifest.json'
        if (!(Test-Path -LiteralPath $manifestPath)) { throw "No manifest: $name" }
        $manifest=Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
        Write-Output "$name status=$($manifest.status) loaded=$($manifest.completed_trials.Count) rejected=$($manifest.rejected_initializations.Count)"
        if ($manifest.status -ne 'completed') { throw "Simulation failure: $name" }
    }
}
