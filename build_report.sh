#!/bin/bash
# Compile the report (run from the project folder: bash build_report.sh)
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode report.tex > latex_build.txt 2>&1
pdflatex -interaction=nonstopmode report.tex > latex_build.txt 2>&1
echo "LaTeX errors: $(grep -c '^!' latex_build.txt)  undefined refs: $(grep -c 'undefined' latex_build.txt)"
ls -l report.pdf
rm -f report.aux report.log report.out latex_build.txt
