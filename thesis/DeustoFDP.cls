\NeedsTeXFormat{LaTeX2e}[1999/01/01]
\ProvidesClass{DeustoFDP}[2015/03/02 v0.2 Final Degree Project template for University of Deusto]

\LoadClass[a4paper,10pt]{book}          % Using the book template as the base

\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}

\RequirePackage[UKenglish]{babel}       % LaTeX in English
\RequirePackage{csquotes}               % Using british quotes
\RequirePackage[citestyle=numeric-comp, % Defining the style for the biblography
                bibstyle=numeric,
                %style=trad-plain,
                %backref=true,
                maxnames=99,
                backrefstyle=two,
                abbreviate=false,
                backend=biber]
                {biblatex}
                                        % Document margins
\RequirePackage[tmargin=3cm,bmargin=3cm,lmargin=3.5cm,rmargin=2.5cm]{geometry}
\RequirePackage{setspace}               % Line spacing set to one and a half
  \onehalfspacing
                                          % Space between paragraphs
  \setlength{\parskip}{1ex plus 0.5ex minus 0.2ex}
  \setlength\parindent{0pt}               % Without indentation

\RequirePackage{enumitem}
  \setitemize{noitemsep,topsep=0pt,parsep=0pt,partopsep=0pt}
  \setenumerate{noitemsep,topsep=0pt,parsep=0pt,partopsep=0pt}
  \setlist[description,2]{font=\normalfont\itshape}
  \setlist[description,3]{font=\normalfont\itshape}

\RequirePackage{fancyhdr}               % Package for defining page header and footer
  \fancypagestyle{plain}{               % Redefine the page style at the beginning of new chapters
    \fancyhf{}
    \fancyfoot[LE,RO]{\thepage}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
  }

\makeatletter                           % White pages on chapter change
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else%
\hbox{} \thispagestyle{empty} \newpage%
\if@twocolumn\hbox{}\newpage\fi\fi\fi}%
\makeatother

\fancypagestyle{phdthesis}{             % Redefine main page style
  \fancyhf{}
  \fancyhead[RO]{\itshape\MakeUppercase{MASTER'S THESIS}}
  \fancyhead[LE]{\sffamily\itshape\thechapter. \leftmark}
  \fancyfoot[LE,RO]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}
\pagestyle{phdthesis}
\renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{#1}}{}}

\makeatletter                           % First page of each chapter according to the style
\renewcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef{\thispagestyle{phdthesis}\@chapter}{\thispagestyle{plain}\@schapter}}
\makeatother

\RequirePackage{graphicx}               % Package for including images
\RequirePackage{booktabs}               % Package for doing professional tables
\RequirePackage{subcaption}             % Packages for doing sub-captions

\RequirePackage{etoolbox}
\RequirePackage{tocbibind}              % Package for including different indexes in the index
\RequirePackage[titles]{tocloft}        % Package for personalizing the index
\preto\section{\ifnum\value{section}=0\addtocontents{toc}{\vskip10pt}\fi}
\preto\figure{\ifnum\value{figure}=0\addtocontents{lof}{{\bfseries Chapter \thechapter\vskip10pt}}\fi}
\preto\table{\ifnum\value{table}=0\addtocontents{lot}{{\bfseries Chapter \thechapter\vskip10pt}}\fi}
\preto\lstlisting{\ifnum\value{lstlisting}=0\addtocontents{lol}{{\bfseries Chapter \thechapter\vskip10pt}}\fi}

\RequirePackage{titlesec}               % Package for personalizing chapters
                                        % Special formats for chapters and sections

\titleformat{\chapter}[display]{\sffamily\bfseries}{}{0pt}{\filleft\Large\thechapter.\quad\uppercase}[\titlerule\vspace{-2em}]
\titleformat{name=\chapter,numberless}{\sffamily\bfseries}{}{0pt}{\Large}[\vspace{-2em}]
\titleformat{\section}{\sffamily\bfseries}{}{0pt}{\Large\thesection. \uppercase}
\titleformat*{\subsection}{\large\bfseries\sffamily}
\titleformat*{\subsubsection}{\large\bfseries\sffamily}

\RequirePackage{hyperref}               % Package for introducing hyperlinks in the documentH
\hypersetup{
  unicode=true,
  bookmarksopen,
  breaklinks,
  plainpages=false,
  pdfborder={0 0 0},
}

\RequirePackage{microtype}              % Package that includes microtyping
\RequirePackage{flafter}                % Do not allow floating objects going backwards

\RequirePackage{multirow}
\RequirePackage{hhline}

\RequirePackage{amsmath}
\RequirePackage{amssymb}

\RequirePackage[table]{xcolor}
\RequirePackage{collcell}
\RequirePackage{pgf}