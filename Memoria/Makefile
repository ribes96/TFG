# chapternames=Ch1_intro Ch2_background Ch3_develop Ch4_expRes Ch5_conlusion Ch6_sustainability
# chapterprefix=Chapters/

# hithere=$(addsufix .tex, $(chapternames))
# $(addpredix Chapters/, $(chapternames))

all: main.pdf

main.pdf: main.tex ribes19bib.bib
	latexmk -pdf main.tex

cont:
	latexmk -pvc -pdf main.tex

clean:
	latexmk -C
