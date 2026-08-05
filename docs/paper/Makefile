# Build the paper. Requires tectonic (brew install tectonic) — it fetches IEEEtran and
# runs the LaTeX/BibTeX passes itself, so no TeX Live installation is needed.

TECTONIC ?= tectonic

.PHONY: all check clean

all: main.pdf

main.pdf: main.tex results.tex refs.bib $(wildcard sections/*.tex)
	$(TECTONIC) -X compile main.tex --keep-intermediates --keep-logs --synctex

# Guard for tasks/lessons.md L5: no literal three-decimal figure may appear outside
# results.tex, so regenerating the numbers is always a one-file edit.
check:
	@if grep -nE '[0-9]\.[0-9]{3}' sections/*.tex | grep -vE 'columnwidth|textwidth|linewidth'; then \
		echo "FAIL: literal figure outside results.tex (see the lines above)"; exit 1; \
	else echo "OK: no literal figures in sections/"; fi
	@test -f main.log || { echo "FAIL: main.log missing — run 'make' first"; exit 1; }
	@if grep -qE 'Warning: (Reference|Citation).*undefined' main.log; then \
		grep -E 'Warning: (Reference|Citation).*undefined' main.log; \
		echo "FAIL: undefined references or citations"; exit 1; \
	else echo "OK: all references and citations resolved"; fi

clean:
	rm -f main.pdf main.aux main.log main.bbl main.blg main.out main.synctex.gz
