#!/usr/bin/env bash
# ==============================================================================
# Papergate — Automated LaTeX Paper Builder (Bash / Linux / macOS / WSL)
#
# Searches for available LaTeX compilation engines (latexmk, pdflatex, xelatex,
# lualatex, tectonic, or Docker), compiles docs/paper/main.tex with bibliographies,
# and cleans up intermediate auxiliary files.
#
# Usage:
#   ./scripts/build_paper.sh                 # Auto-detect engine & compile & clean
#   ./scripts/build_paper.sh --engine pdflatex
#   ./scripts/build_paper.sh --keep-aux      # Do not delete .aux, .log, .bbl
#   ./scripts/build_paper.sh --clean-only    # Only clean auxiliary files
# ==============================================================================

set -eo pipefail

# ANSI color codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Determine directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAPER_DIR="${REPO_ROOT}/docs/paper"
MAIN_TEX="main.tex"
MAIN_BASE="main"

ENGINE="auto"
KEEP_AUX=false
CLEAN_ONLY=false

# Parse command line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine|-e)
      ENGINE="$2"
      shift 2
      ;;
    --keep-aux|-k)
      KEEP_AUX=true
      shift
      ;;
    --clean-only|-c)
      CLEAN_ONLY=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --engine, -e <name>   Force engine: pdflatex, latexmk, xelatex, lualatex, tectonic, docker"
      echo "  --keep-aux, -k        Keep intermediate build files"
      echo "  --clean-only, -c      Only clean intermediate files and exit"
      echo "  --help, -h            Show this help message"
      exit 0
      ;;
    *)
      echo -e "${RED}[!] Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

if [[ ! -f "${PAPER_DIR}/${MAIN_TEX}" ]]; then
  echo -e "${RED}[!] Error: File '${MAIN_TEX}' not found in '${PAPER_DIR}'.${NC}"
  exit 1
fi

clean_aux_files() {
  echo -e "${CYAN}[*] Cleaning intermediate LaTeX build files in: ${PAPER_DIR}${NC}"
  local count=0
  local patterns=(
    "*.aux" "*.log" "*.bbl" "*.blg" "*.out" "*.synctex.gz"
    "*.fdb_latexmk" "*.fls" "*.toc" "*.nav" "*.snm" "*.vrb"
    "*.bcf" "*.run.xml" "*.auxlock"
  )
  for pat in "${patterns[@]}"; do
    for f in "${PAPER_DIR}"/${pat}; do
      if [[ -f "$f" ]]; then
        rm -f "$f"
        count=$((count + 1))
      fi
    done
  done
  echo -e "${GREEN}[+] Removed ${count} intermediate file(s).${NC}"
}

if [[ "$CLEAN_ONLY" == true ]]; then
  clean_aux_files
  exit 0
fi

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN} Papergate — Automated LaTeX Paper Builder${NC}"
echo -e "${CYAN}============================================================${NC}"
echo -e "[*] Target Directory: ${PAPER_DIR}"
echo -e "[*] Main Document:    ${MAIN_TEX}"

# Check command utility
has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

SELECTED_ENGINE=""
HAS_BIBTEX=false
if has_cmd bibtex; then
  HAS_BIBTEX=true
fi
HAS_PERL=false
if has_cmd perl; then
  HAS_PERL=true
fi

run_three_pass() {
  local compiler="$1"
  echo -e "${YELLOW}[>] [Pass 1/3] Running ${compiler}...${NC}"
  "$compiler" -interaction=nonstopmode "${MAIN_TEX}" >/dev/null 2>&1 || true

  if [[ "$HAS_BIBTEX" == true ]]; then
    echo -e "${YELLOW}[>] Running bibtex...${NC}"
    bibtex "${MAIN_BASE}" >/dev/null 2>&1 || true
  else
    echo -e "${YELLOW}[!] Warning: bibtex not found in PATH. References may not be resolved.${NC}"
  fi

  echo -e "${YELLOW}[>] [Pass 2/3] Running ${compiler}...${NC}"
  "$compiler" -interaction=nonstopmode "${MAIN_TEX}" >/dev/null 2>&1 || true

  echo -e "${YELLOW}[>] [Pass 3/3] Running ${compiler} (finalizing references)...${NC}"
  "$compiler" -interaction=nonstopmode "${MAIN_TEX}" >/dev/null 2>&1 || true
}

if [[ "$ENGINE" == "auto" ]]; then
  echo -e "${YELLOW}[*] Scanning system for available LaTeX engines...${NC}"
  
  CANDIDATES=("pdflatex" "latexmk" "xelatex" "lualatex" "tectonic")
  for cand in "${CANDIDATES[@]}"; do
    if has_cmd "$cand"; then
      cmd_path="$(command -v "$cand")"
      if [[ "$cand" == "latexmk" && "$HAS_PERL" == false ]]; then
        echo -e "  ${GRAY}[-] Found latexmk, but 'perl' is missing (skipping latexmk)${NC}"
        continue
      fi
      echo -e "  ${GREEN}[+] Found: ${cand} (${cmd_path})${NC}"
      if [[ -z "$SELECTED_ENGINE" ]]; then
        SELECTED_ENGINE="$cand"
      fi
    else
      echo -e "  ${GRAY}[-] Not found: ${cand}${NC}"
    fi
  done

  if [[ -z "$SELECTED_ENGINE" ]]; then
    if has_cmd docker; then
      echo -e "  ${GREEN}[+] Found Docker: will use TeX Live Docker container as fallback.${NC}"
      SELECTED_ENGINE="docker"
    else
      echo -e "${RED}[!] No usable LaTeX engine found in PATH. Please install TeX Live, MacTeX or MiKTeX.${NC}"
      exit 1
    fi
  fi
else
  SELECTED_ENGINE="$ENGINE"
  if [[ "$SELECTED_ENGINE" != "docker" ]] && ! has_cmd "$SELECTED_ENGINE"; then
    echo -e "${RED}[!] Specified engine '${SELECTED_ENGINE}' not found in PATH.${NC}"
    exit 1
  fi
fi

echo -e "[*] Selected Compilation Engine: ${MAGENTA}${SELECTED_ENGINE}${NC}"
echo "------------------------------------------------------------"

cd "${PAPER_DIR}"

trap '
  if [[ "$KEEP_AUX" == false ]]; then
    clean_aux_files
  else
    echo -e "${GRAY}[*] --keep-aux specified: intermediate files retained.${NC}"
  fi
' EXIT

START_TIME=$(date +%s)

case "$SELECTED_ENGINE" in
  latexmk)
    echo -e "${YELLOW}[>] Executing latexmk (automated multi-pass & bibtex)...${NC}"
    if ! latexmk -pdf -interaction=nonstopmode "${MAIN_TEX}"; then
      echo -e "${YELLOW}[!] latexmk encountered an issue. Falling back to pdflatex...${NC}"
      if has_cmd pdflatex; then
        run_three_pass "pdflatex"
      fi
    fi
    ;;
  pdflatex|xelatex|lualatex)
    run_three_pass "$SELECTED_ENGINE"
    ;;
  tectonic)
    echo -e "${YELLOW}[>] Running tectonic...${NC}"
    tectonic "${MAIN_TEX}"
    ;;
  docker)
    echo -e "${YELLOW}[>] Running compilation inside Docker (texlive container)...${NC}"
    docker run --rm -v "${PAPER_DIR}:/work" -w /work texlive/texlive:latest sh -c \
      "pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex"
    ;;
esac

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

PDF_FILE="${PAPER_DIR}/${MAIN_BASE}.pdf"
if [[ -f "$PDF_FILE" ]]; then
  SIZE_KB=$(du -k "$PDF_FILE" 2>/dev/null | cut -f1 || echo "unknown")
  echo -e "${GREEN}============================================================${NC}"
  echo -e "${GREEN}[SUCCESS] Paper compiled successfully in ${DURATION}s!${NC}"
  echo -e "${GREEN}  -> PDF File: ${PDF_FILE} (${SIZE_KB} KB)${NC}"
  echo -e "${GREEN}============================================================${NC}"

  LOG_FILE="${PAPER_DIR}/${MAIN_BASE}.log"
  if [[ -f "$LOG_FILE" ]] && grep -qE 'Warning: (Reference|Citation).*undefined' "$LOG_FILE"; then
    echo -e "${YELLOW}[!] Warning: Some citations or references might be undefined. Check ${MAIN_BASE}.log.${NC}"
  fi
else
  echo -e "${RED}[!] Build failed: PDF file '${PDF_FILE}' was not generated. Check ${MAIN_BASE}.log for errors.${NC}"
  exit 1
fi
