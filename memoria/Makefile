all: doc.pdf

doc.pdf: doc.tex ribes19bib.bib
	latexmk -pdf doc.tex

cont:
	latexmk -pvc -pdf doc.tex

clean:
	latexmk -C
