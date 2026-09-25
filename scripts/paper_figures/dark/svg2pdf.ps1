# Convert self-contained SVG figures to vector PDF with headless Edge's print
# engine (text stays text; embedded PNG panels stay raster).  For every SVG an
# HTML wrapper with a CSS @page of exactly the canvas size is written next to
# it, printed with --print-to-pdf, and removed again.
#
#   powershell -ExecutionPolicy Bypass -File scripts/paper_figures/dark/svg2pdf.ps1 a.svg b.svg
#   powershell -ExecutionPolicy Bypass -File scripts/paper_figures/dark/svg2pdf.ps1 -Dir outputs/paper_figures/dark
#   ... -OutDir <folder>     writes every PDF into <folder> (default: next to the SVG)
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)] [string[]] $Svgs,
    [string] $Dir,
    [string] $OutDir,
    [string] $Edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
)
if ($Dir) { $Svgs += @(Get-ChildItem -Path $Dir -Recurse -Filter *.svg | ForEach-Object { $_.FullName }) }
if (-not $Svgs) { Write-Error "no SVG given"; exit 1 }
if ($OutDir) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

function Convert-One([string] $svg) {
    $head = Get-Content $svg -TotalCount 2 | Out-String
    if ($head -notmatch 'width="(\d+)" height="(\d+)"') { Write-Error "no width/height in $svg"; return }
    $w = [int]$Matches[1]
    $h = [int]$Matches[2]
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($svg)
    $pdf = if ($OutDir) { Join-Path (Resolve-Path $OutDir).Path ($stem + ".pdf") } else { [System.IO.Path]::ChangeExtension($svg, ".pdf") }
    if (Test-Path $pdf) { Remove-Item $pdf -Force }
    $html = [System.IO.Path]::ChangeExtension($svg, ".print.html")
    $svgText = Get-Content $svg -Raw -Encoding UTF8
    $doc = "<!doctype html><html><head><meta charset='utf-8'><style>@page{size:" + $w + "px " + $h + "px;margin:0}" +
           "html,body{margin:0;padding:0;background:#fff}svg{display:block;width:" + $w + "px;height:" + $h + "px}</style></head><body>" +
           $svgText + "</body></html>"
    [System.IO.File]::WriteAllText($html, $doc, (New-Object System.Text.UTF8Encoding($false)))
    $edgeProfile = Join-Path $env:TEMP ("edge-pdf-" + [guid]::NewGuid().ToString("N"))
    $url = "file:///" + ($html -replace '\\', '/') + "?v=" + [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $edgeArgs = @("--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--user-data-dir=$edgeProfile",
                  "--print-to-pdf=$pdf", $url)
    $t0 = Get-Date
    $proc = Start-Process -FilePath $Edge -ArgumentList $edgeArgs -Wait -PassThru
    for ($i = 0; $i -lt 40; $i++) {
        $mine = @(Get-Process msedge -ErrorAction SilentlyContinue | Where-Object {
            try { $_.StartTime -ge $t0 } catch { $false } })
        if ($mine.Count -eq 0) { break }
        Start-Sleep -Milliseconds 500
    }
    Remove-Item $html -Force -ErrorAction SilentlyContinue
    if (Test-Path $pdf) {
        Write-Output ("wrote " + $pdf + " (" + $w + "x" + $h + " px page)")
    } else {
        Write-Error ("Edge (exit " + $proc.ExitCode + ") wrote nothing for " + $svg)
    }
}

foreach ($svgPath in $Svgs) {
    Convert-One ((Resolve-Path $svgPath).Path)
}
