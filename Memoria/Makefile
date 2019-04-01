all: main.pdf

main.pdf: main.tex ribes19bib.bib
	latexmk -pdf main.tex

cont:
	latexmk -pvc -pdf main.tex

clean:
	latexmk -C
