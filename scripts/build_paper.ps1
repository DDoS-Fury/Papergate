<#
.SYNOPSIS
    Auto-detects available LaTeX engines, compiles the scientific paper, and cleans intermediate files.

.DESCRIPTION
    Searches for installed LaTeX compilation tools (latexmk, pdflatex, xelatex, lualatex, tectonic, docker),
    compiles the manuscript with bibliographies (BibTeX), verifies output PDF generation,
    and removes auxiliary/intermediate build artifacts.

.PARAMETER Engine
    Force a specific LaTeX engine: 'auto', 'latexmk', 'pdflatex', 'xelatex', 'lualatex', 'tectonic', 'docker'.

.PARAMETER KeepAux
    If specified, intermediate files (.aux, .log, .bbl, etc.) will NOT be deleted after compilation.

.PARAMETER CleanOnly
    If specified, only cleans intermediate files without compiling.

.EXAMPLE
    .\scripts\build_paper.ps1
    .\scripts\build_paper.ps1 -Engine pdflatex
    .\scripts\build_paper.ps1 -CleanOnly
#>

[CmdletBinding()]
param (
    [ValidateSet('auto', 'latexmk', 'pdflatex', 'xelatex', 'lualatex', 'tectonic', 'docker')]
    [string]$Engine = 'auto',

    [switch]$KeepAux,
    [switch]$CleanOnly
)

$ErrorActionPreference = 'Stop'

# Determine directories
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$PaperDir = Join-Path $RepoRoot "docs\paper"
$MainTex = "main.tex"
$MainBase = "main"

if (-not (Test-Path (Join-Path $PaperDir $MainTex))) {
    Write-Error "[!] Error: File '$MainTex' not found in '$PaperDir'."
    exit 1
}

# Intermediate file patterns to clean
$AuxExtensions = @(
    "*.aux", "*.log", "*.bbl", "*.blg", "*.out", "*.synctex.gz",
    "*.fdb_latexmk", "*.fls", "*.toc", "*.nav", "*.snm", "*.vrb",
    "*.bcf", "*.run.xml", "*.auxlock"
)

function Clean-AuxFiles {
    param ([string]$TargetDir)
    Write-Host "[*] Cleaning intermediate LaTeX build files in: $TargetDir" -ForegroundColor Cyan
    $Count = 0
    foreach ($pattern in $AuxExtensions) {
        $files = Get-ChildItem -Path $TargetDir -Filter $pattern -File -ErrorAction SilentlyContinue
        foreach ($file in $files) {
            Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
            $Count++
        }
    }
    Write-Host "[+] Removed $Count intermediate file(s)." -ForegroundColor Green
}

if ($CleanOnly) {
    Clean-AuxFiles -TargetDir $PaperDir
    exit 0
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Papergate — Automated LaTeX Paper Builder" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[*] Target Directory: $PaperDir"
Write-Host "[*] Main Document:    $MainTex"

function Test-CommandAvailable ([string]$CmdName) {
    return [bool](Get-Command $CmdName -ErrorAction SilentlyContinue)
}

$HasBibtex = Test-CommandAvailable "bibtex"
$HasPerl = Test-CommandAvailable "perl"

# Order of preference: prioritize pdflatex if latexmk lacks perl
if ($Engine -eq 'auto') {
    Write-Host "[*] Scanning system for available LaTeX engines..." -ForegroundColor Yellow
    
    $Candidates = @('pdflatex', 'latexmk', 'xelatex', 'lualatex', 'tectonic')
    $SelectedEngine = $null
    
    foreach ($cand in $Candidates) {
        if (Test-CommandAvailable $cand) {
            $cmdPath = (Get-Command $cand).Source
            if ($cand -eq 'latexmk' -and -not $HasPerl) {
                Write-Host "  [-] Found $cand, but 'perl' is missing (skipping $cand)" -ForegroundColor DarkGray
                continue
            }
            Write-Host "  [+] Found: $cand ($cmdPath)" -ForegroundColor Green
            if ($null -eq $SelectedEngine) {
                $SelectedEngine = $cand
            }
        } else {
            Write-Host "  [-] Not found: $cand" -ForegroundColor DarkGray
        }
    }

    if ($null -eq $SelectedEngine) {
        if (Test-CommandAvailable "docker") {
            Write-Host "  [+] Found Docker: using TeX Live container fallback." -ForegroundColor Green
            $SelectedEngine = 'docker'
        } else {
            Write-Error "[!] No usable LaTeX engine found in PATH. Please install MiKTeX or TeX Live."
            exit 1
        }
    }
} else {
    $SelectedEngine = $Engine
    if ($SelectedEngine -ne 'docker' -and -not (Test-CommandAvailable $SelectedEngine)) {
        Write-Error "[!] Specified engine '$SelectedEngine' was not found in PATH."
        exit 1
    }
}

Write-Host "[*] Selected Compilation Engine: $SelectedEngine" -ForegroundColor Magenta
Write-Host "------------------------------------------------------------"

Push-Location $PaperDir
try {
    $StartTime = Get-Date

    function Run-ThreePass ($Compiler) {
        Write-Host "[>] [Pass 1/3] Running $Compiler..." -ForegroundColor Yellow
        & $Compiler -interaction=nonstopmode $MainTex | Out-Null

        if ($HasBibtex) {
            Write-Host "[>] Running bibtex..." -ForegroundColor Yellow
            & bibtex $MainBase | Out-Null
        } else {
            Write-Warning "[!] bibtex not found in PATH. References may not be resolved."
        }

        Write-Host "[>] [Pass 2/3] Running $Compiler..." -ForegroundColor Yellow
        & $Compiler -interaction=nonstopmode $MainTex | Out-Null

        Write-Host "[>] [Pass 3/3] Running $Compiler (finalizing references)..." -ForegroundColor Yellow
        & $Compiler -interaction=nonstopmode $MainTex | Out-Null
    }

    switch ($SelectedEngine) {
        'latexmk' {
            Write-Host "[>] Executing latexmk (automated multi-pass & bibtex)..." -ForegroundColor Yellow
            & latexmk -pdf -interaction=nonstopmode $MainTex
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "[!] latexmk encountered an issue. Falling back to pdflatex..."
                if (Test-CommandAvailable "pdflatex") {
                    Run-ThreePass "pdflatex"
                }
            }
        }
        
        { $_ -in @('pdflatex', 'xelatex', 'lualatex') } {
            Run-ThreePass $SelectedEngine
        }

        'tectonic' {
            Write-Host "[>] Running tectonic..." -ForegroundColor Yellow
            & tectonic $MainTex
        }

        'docker' {
            Write-Host "[>] Running compilation inside Docker (texlive container)..." -ForegroundColor Yellow
            docker run --rm -v "${PaperDir}:/work" -w /work texlive/texlive:latest sh -c "pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex"
        }
    }

    $PdfPath = Join-Path $PaperDir "$MainBase.pdf"
    if (Test-Path $PdfPath) {
        $PdfItem = Get-Item $PdfPath
        $Duration = [math]::Round(((Get-Date) - $StartTime).TotalSeconds, 2)
        $SizeKB = [math]::Round($PdfItem.Length / 1KB, 1)

        Write-Host "============================================================" -ForegroundColor Green
        Write-Host "[SUCCESS] Paper compiled successfully in $Duration s!" -ForegroundColor Green
        Write-Host "  -> PDF File: $PdfPath ($SizeKB KB)" -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Green

        # Check log for unresolved warnings
        $LogPath = Join-Path $PaperDir "$MainBase.log"
        if (Test-Path $LogPath) {
            $LogContent = Get-Content $LogPath -Raw
            if ($LogContent -match 'Warning: (Reference|Citation).*undefined') {
                Write-Warning "[!] Warning: Some citations or references might be undefined. Check main.log."
            }
        }
    } else {
        Write-Error "[!] Build failed: PDF file '$PdfPath' was not generated. Check $MainBase.log for errors."
        exit 1
    }

} finally {
    Pop-Location
    if (-not $KeepAux) {
        Clean-AuxFiles -TargetDir $PaperDir
    } else {
        Write-Host "[*] -KeepAux specified: intermediate files retained." -ForegroundColor Gray
    }
}
