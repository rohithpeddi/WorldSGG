# Rasterise self-contained SVG figures to PNG with headless Edge (no extra
# dependencies on Windows).  Each SVG's width / height are read from its root
# element, so the screenshot is exactly the canvas.
#
#   powershell -ExecutionPolicy Bypass -File scripts/paper_figures/dark/rasterize.ps1 outputs/paper_figures/dark/12XD3/worldwise.svg ...
#   powershell -ExecutionPolicy Bypass -File scripts/paper_figures/dark/rasterize.ps1 -Dir outputs/paper_figures/dark   # every SVG below it
#
# Every render gets a fresh profile directory (left under $env:TEMP for the OS to
# clean) and a cache-busting query string, and the script waits for the Edge
# processes it started to exit before launching the next one: a lingering
# instance adopts a new launch and exits 0 without writing the screenshot.
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)] [string[]] $Svgs,
    [string] $Dir,
    [string] $Edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
)
if ($Dir) { $Svgs += @(Get-ChildItem -Path $Dir -Recurse -Filter *.svg | ForEach-Object { $_.FullName }) }
if (-not $Svgs) { Write-Error "no SVG given"; exit 1 }

function Render-One([string] $svg) {
    $head = Get-Content $svg -TotalCount 2 | Out-String
    if ($head -notmatch 'width="(\d+)" height="(\d+)"') { Write-Error "no width/height in $svg"; return }
    $w = [int]$Matches[1]
    $h = [int]$Matches[2]
    $png = [System.IO.Path]::ChangeExtension($svg, ".png")
    if (Test-Path $png) { Remove-Item $png -Force }
    $edgeProfile = Join-Path $env:TEMP ("edge-raster-" + [guid]::NewGuid().ToString("N"))
    $url = "file:///" + ($svg -replace '\\', '/') + "?v=" + [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $edgeArgs = @("--headless=new", "--disable-gpu", "--hide-scrollbars",
                  "--user-data-dir=$edgeProfile", "--window-size=$w,$h", "--screenshot=$png", $url)
    $t0 = Get-Date
    $proc = Start-Process -FilePath $Edge -ArgumentList $edgeArgs -Wait -PassThru
    # wait (up to 20 s) for the Edge processes of this launch to go away
    for ($i = 0; $i -lt 40; $i++) {
        $mine = @(Get-Process msedge -ErrorAction SilentlyContinue | Where-Object {
            try { $_.StartTime -ge $t0 } catch { $false } })
        if ($mine.Count -eq 0) { break }
        Start-Sleep -Milliseconds 500
    }
    if (Test-Path $png) {
        Write-Output ("wrote " + $png + " (" + $w + "x" + $h + ")")
    } else {
        Write-Error ("Edge (exit " + $proc.ExitCode + ") wrote nothing for " + $svg)
    }
}

foreach ($svgPath in $Svgs) {
    Render-One ((Resolve-Path $svgPath).Path)
}
