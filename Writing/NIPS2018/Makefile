SOURCES	= \
	main.tex

.PHONY: default all clean

default:
	pdflatex  -shell-escape main.tex

all:  
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex


clean:
	-rm -rf main.pdf main.synctex.gz main.out main.tpt main.aux main.bbl main.blg main.dvi main.log main.ps *~

sources:
	@echo  ${SOURCES}

