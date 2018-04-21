SOURCES	= \
	sd.tex

.PHONY: default all clean

default:
	pdflatex  -shell-escape sd.tex

all:  
	pdflatex  -shell-escape sd.tex
	-	bibtex sd
	pdflatex sd.tex
	pdflatex sd.tex
	thumbpdf sd.pdf
	pdflatex sd.tex

clean:
	-rm -rf sd.pdf sd.synctex.gz sd.out sd.tpt sd.aux sd.bbl sd.blg sd.dvi sd.log sd.ps *~

sources:
	@echo  ${SOURCES}
