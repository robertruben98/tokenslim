# install.ps1 — install the TokenSlim CLI and library (Windows PowerShell / pwsh).
#
# Usage:
#   .\install.ps1               # install tokenslim
#   .\install.ps1 -WithExtras   # install tokenslim[tokenizers,images,semantic]
#
# Behavior mirrors install.sh:
#   * Uses pipx when available, otherwise `pip install --user`.
#   * Idempotent: re-running upgrades/reinstalls the same package.
#   * Verifies the install by running `tokenslim doctor`.

param(
    [switch]$WithExtras
)

$ErrorActionPreference = 'Stop'

$Spec = if ($WithExtras) { 'tokenslim[tokenizers,images,semantic]' } else { 'tokenslim' }

function Find-Python {
    # Prefer the py launcher: the bare 'python3'/'python' names may resolve to
    # the Microsoft Store app-execution-alias stub on a fresh Windows install.
    foreach ($candidate in @('py', 'python', 'python3')) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) {
            return $candidate
        }
    }
    return $null
}

Write-Host "Installing $Spec ..."

if (Get-Command pipx -ErrorAction SilentlyContinue) {
    # --force makes re-runs idempotent (reinstall/upgrade instead of erroring).
    pipx install --force $Spec
    if ($LASTEXITCODE -ne 0) { throw "pipx install failed with exit code $LASTEXITCODE" }
    $Installer = 'pipx'
} else {
    $Python = Find-Python
    if (-not $Python) {
        throw 'No python found on PATH. Install Python 3.10+ first.'
    }
    & $Python -m pip install --user --upgrade $Spec
    if ($LASTEXITCODE -ne 0) { throw "pip install failed with exit code $LASTEXITCODE" }
    $Installer = 'pip --user'
}

Write-Host "Installed via $Installer. Verifying with 'tokenslim doctor' ..."

if (Get-Command tokenslim -ErrorAction SilentlyContinue) {
    tokenslim doctor
    if ($LASTEXITCODE -ne 0) { throw "'tokenslim doctor' failed with exit code $LASTEXITCODE" }
} else {
    $Python = Find-Python
    if (-not $Python) {
        throw "Verification failed - 'tokenslim' is not on PATH and no python was found."
    }
    & $Python -m tokenslim.cli doctor
    if ($LASTEXITCODE -ne 0) {
        throw "Verification failed - 'tokenslim doctor' did not run. If you installed with pip --user, add the user Scripts directory to PATH and retry."
    }
    Write-Warning "'tokenslim' is not on PATH yet (check your user Scripts directory)."
}

Write-Host 'TokenSlim is ready. Try: tokenslim --help'
