# A private, checksum-verified runtime leaves existing Python/Conda installs alone.
[CmdletBinding()]
param(
    [string]$Prefix = (Join-Path $env:LOCALAPPDATA 'NeuroPyGuiN'),
    [string]$Source = '',
    [switch]$Update,
    [switch]$NoLaunch
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-Checked {
    # PowerShell 5 does not throw when a native process fails, so check every exit.
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'NeuroPyGuiN requires Windows x64.'
    }
    $Prefix = [IO.Path]::GetFullPath($Prefix)
    New-Item -ItemType Directory -Force -Path (Join-Path $Prefix 'bin') | Out-Null
    $Mamba = Join-Path $Prefix 'bin\micromamba.exe'
    $Expected = 'aa763f3c7fc7ef529d79f170d1d134569995f08fe54a8c545ef20e01e0dc7046'
    if (-not (Test-Path $Mamba) -or (Get-FileHash $Mamba -Algorithm SHA256).Hash -ne $Expected) {
        Write-Host 'Downloading the private environment manager...'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing 'https://github.com/mamba-org/micromamba-releases/releases/download/2.7.0-0/micromamba-win-64' -OutFile "$Mamba.download"
        if ((Get-FileHash "$Mamba.download" -Algorithm SHA256).Hash -ne $Expected) {
            Remove-Item "$Mamba.download"
            throw 'Download checksum mismatch.'
        }
        Move-Item -Force "$Mamba.download" $Mamba
    }
    $env:MAMBA_ROOT_PREFIX = Join-Path $Prefix 'mamba'
    $env:PYTHONNOUSERSITE = '1'
    Remove-Item Env:PYTHONHOME, Env:PYTHONPATH -ErrorAction SilentlyContinue
    $Bootstrap = Join-Path $Prefix 'bootstrap'
    $Python = Join-Path $Bootstrap 'python.exe'
    $Git = Join-Path $Bootstrap 'Library\bin\git.exe'
    if (-not (Test-Path $Python) -or -not (Test-Path $Git)) {
        Invoke-Checked $Mamba @('create', '-y', '--no-rc', '-p', $Bootstrap, '--override-channels', '-c', 'conda-forge', 'python=3.10', 'git')
    }
    $env:PATH = "$Bootstrap;$Bootstrap\Library\bin;$Bootstrap\Scripts;$env:PATH"
    if (-not $Source) {
        $Checkout = Split-Path -Parent $PSScriptRoot
        if (Test-Path (Join-Path $Checkout 'main.py')) { $Source = $Checkout }
        else { $Source = Join-Path $Prefix 'app' }
    }
    $Source = [IO.Path]::GetFullPath($Source)
    if (-not (Test-Path $Source)) {
        Invoke-Checked $Git @('clone', '--branch', 'main', 'https://github.com/BelloneLab/NeuroPyGuiN.git', $Source)
    }
    $Setup = Join-Path $Source 'install\setup.py'
    if (-not (Test-Path $Setup)) { throw "Not a NeuroPyGuiN checkout: $Source" }
    $SetupArgs = @($Setup, '--prefix', $Prefix, '--source', $Source)
    if ($Update) { $SetupArgs += '--update' }
    if ($NoLaunch) { $SetupArgs += '--no-launch' }
    Invoke-Checked $Python $SetupArgs
}
catch {
    Write-Error $_
    exit 1
}
