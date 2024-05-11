#!/bin/bash

# Compile NN diagrams
for file in diagrams/*.tex; do
    pdflatex  -interaction=nonstopmode -halt-on-error -output-directory=diagrams $file
done

pdflatex -interaction=nonstopmode final.tex
bibtex final
pdflatex -interaction=nonstopmode final.tex
pdflatex -interaction=nonstopmode final.tex
