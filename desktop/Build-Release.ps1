param(
    [string]$OutputZip = ""
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtime = Join-Path $project "runtime\langswap"
$releaseRoot = Join-Path $project "work\desktop-release"
$stage = Join-Path $releaseRoot ("stage-" + [guid]::NewGuid().ToString("N"))
if (-not $OutputZip) {
    $OutputZip = Join-Path $releaseRoot "LangSwap-Windows-0.1.0.zip"
}
$compiler = Join-Path $env:WINDIR "Microsoft.NET\Framework/v4.0.30319/csc.exe"
if (-not (Test-Path -LiteralPath $compiler)) {
    throw ".NET Framework compiler is missing: $compiler"
}
New-Item -ItemType Directory -Path $stage -Force | Out-Null
$runtimeOut = Join-Path $stage "runtime\langswap"
New-Item -ItemType Directory -Path $runtimeOut -Force | Out-Null

$source = Join-Path $project "desktop\LangSwap.Desktop\Program.cs"
$exe = Join-Path $stage "LangSwap.exe"
& $compiler /nologo /target:winexe /platform:x64 /codepage:65001 /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.Core.dll ("/out:" + $exe) $source
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $exe)) {
    throw "Windows application compilation failed."
}

# An explicit source whitelist keeps local models, media, caches and secrets out.
$rootFiles = @(
    ".dockerignore", "Dockerfile.french", "Dockerfile.qwen", "LICENSE",
    "README.md", "pyproject.toml", "main.py", "gradio_demo.py",
    "desktop_worker.py", "qwen_generate.py"
)
foreach ($name in $rootFiles) {
    Copy-Item -LiteralPath (Join-Path $runtime $name) -Destination (Join-Path $runtimeOut $name)
}
$sourceRoot = Join-Path $runtime "langswap"
Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | Where-Object {
    $_.Extension -in @(".py", ".json", ".yaml") -or $_.Name -eq "__VERSION__"
} | ForEach-Object {
    $relative = $_.FullName.Substring($sourceRoot.Length).TrimStart("\")
    $destination = Join-Path (Join-Path $runtimeOut "langswap") $relative
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $_.FullName -Destination $destination
}
Copy-Item -LiteralPath (Join-Path $runtime "LICENSE") -Destination (Join-Path $stage "LICENSE")
Copy-Item -LiteralPath (Join-Path $project "NOTICE.md") -Destination (Join-Path $stage "NOTICE.md")
Copy-Item -LiteralPath (Join-Path $project "desktop\README-Windows.md") -Destination (Join-Path $stage "LISEZMOI.md")
$desktopSource = Join-Path $stage "desktop\LangSwap.Desktop"
New-Item -ItemType Directory -Path $desktopSource -Force | Out-Null
Copy-Item -LiteralPath $source -Destination (Join-Path $desktopSource "Program.cs")
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $stage "desktop\Build-Release.ps1")
Copy-Item -LiteralPath (Join-Path $project "desktop\README-Windows.md") -Destination (Join-Path $stage "desktop\README-Windows.md")

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipPath = [System.IO.Path]::GetFullPath($OutputZip)
New-Item -ItemType Directory -Path (Split-Path -Parent $zipPath) -Force | Out-Null
if (Test-Path -LiteralPath $zipPath) {
    throw "Release ZIP already exists: $zipPath"
}
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $stage, $zipPath, [System.IO.Compression.CompressionLevel]::Optimal, $false)
$artifact = Get-Item -LiteralPath $zipPath
$digest = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash
Write-Output ("RELEASE|" + $artifact.FullName + "|" + $artifact.Length + "|" + $digest)
