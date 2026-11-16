param(
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Debug',
    [switch]$LegacyPresent,
    [float]$RenderScale = 1.0
)
$ErrorActionPreference = 'Stop'
$templateRoot = Split-Path -Parent $PSScriptRoot
Push-Location $templateRoot
try {
    cmake --preset windows
    if ($LASTEXITCODE) { throw 'CMake configure failed' }
    cmake --build out/windows --config $Configuration --parallel 4
    if ($LASTEXITCODE) { throw 'Native build failed' }
    ctest --test-dir out/windows -C $Configuration --output-on-failure
    if ($LASTEXITCODE) { throw 'CPU regression tests failed' }
    $templateExe = Join-Path $templateRoot "out/windows/$Configuration/sandbox_windows.exe"
    $templateLog = Join-Path $templateRoot "out/windows/smoke-$Configuration.log"
    $templateArgs = @('--hidden', '--smoke-test', '--stress', '--frames', '240', '--render-scale', $RenderScale.ToString([Globalization.CultureInfo]::InvariantCulture))
    if ($LegacyPresent) { $templateArgs += '--legacy-present' }
    # A different cwd verifies that runtime assets are resolved by the adapter.
    Push-Location (Join-Path $templateRoot 'out')
    try {
        & $templateExe @templateArgs 2>&1 | Tee-Object -FilePath $templateLog
        if ($LASTEXITCODE) { throw "GPU smoke test failed; see $templateLog" }
    } finally { Pop-Location }
} finally { Pop-Location }
