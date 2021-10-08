%===============================================================================
%
% Copyright (C) 2015 Jean-Sébastien Gosselin
%
% This work is free software: you can redistribute it and/or modify it under the terms of the CC BY-NC-SA 3.0 License as published by the Creative Commons nonprofit organization.
%
% This work is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the CC BY-NC-SA 3.0 License for more details at http://creativecommons.org/licenses/by-nc-sa/3.0/.
%
% Important notes:
% This template needs to be compiled with XeLaTeX and the bibliography, if
% used, needs to be compiled with biber rather than bibtex.
%
%===============================================================================

\ProvidesClass{fiches-rsesq}[2021/10/05 Fiches Signalétique RSESQ]
\NeedsTeXFormat{LaTeX2e}

\LoadClass[10pt, 5p, onecolumn]{report}

\pagestyle{empty}

\newcommand{\classleftmargin}{1cm}
\newcommand{\classrightmargin}{1cm}
\newcommand{\classtopmargin}{4.5cm}
\newcommand{\classbottommargin}{3.25cm}
\newcommand{\classpaperheight}{11in}
\newcommand{\classpaperwidth}{8.5in}

\RequirePackage[left=\classleftmargin, right=\classrightmargin, 
                top=\classtopmargin, bottom=\classbottommargin]{geometry}

%===============================================================================
% COLORS 
%===============================================================================
\RequirePackage[svgnames,table]{xcolor}
\definecolor{gray}{HTML}{4D4D4D}
\definecolor{white}{HTML}{FFFFFF}
\definecolor{blue}{HTML}{095797}

\colorlet{headercolor}{blue}
\colorlet{fillheader}{blue}
\colorlet{textheader}{white}


%===============================================================================
% FONTS 
%===============================================================================
%http://graphicdesign.stackexchange.com/questions/15462/

\RequirePackage[quiet]{fontspec}

\newfontfamily\thinfont[Path = fonts/]{Roboto-Regular}
\newfontfamily\bodyfont[Path = fonts/]{Roboto-Regular}
\newfontfamily\boldfont[Path = fonts/]{Roboto-Bold}
\newfontfamily\robotoitalic[Path = fonts/]{Roboto-Italic}
\newfontfamily\robotoblack[Path = fonts/]{Roboto-Black}

\setmainfont[Mapping=tex-text, Scale=MatchLowercase]{Roboto-Regular}

\newcommand{\notefont}{\fontsize{10pt}{12pt} %
            \selectfont\robotoitalic\color{black}}

\newcommand{\subtitlefont}{\fontsize{20}{24} %
            \selectfont\boldfont\color{blue}}

\newcommand{\boxtitlefont}{\fontsize{14}{16} %
            \selectfont\boldfont\color{white}}


%===============================================================================
% HEADER AND FOOTER
%===============================================================================
\RequirePackage{tikz}
\RequirePackage{tikzpagenodes}
\RequirePackage{lastpage}

% Clear the header, footer, and rules
%\RequirePackage{fancyhdr}
%\pagestyle{fancy}
%\fancyhf{}
%\renewcommand{\headrulewidth}{0pt}

\newcommand{\header}[2]{%
  \begin{tikzpicture}[remember picture, overlay]
    \node [rectangle, fill=fillheader, anchor=north, minimum width=\paperwidth, minimum height=2.5cm, yshift=-1cm] (box) at (current page.north){};
    \node [anchor=center, yshift=8pt] (name) at (box) {%
      \fontsize{40pt}{72pt}\color{textheader}%
      \bodyfont Station #1
    };
    \node [anchor=north] at (name.south) {%
      \fontsize{16pt}{24pt}\color{textheader}%
      \thinfont Municipalité de #2%
    };
  \end{tikzpicture}
  \vspace{-2em}
}

\newcommand{\footer}[0]{%
  \begin{tikzpicture}[remember picture, overlay]
    \node [rectangle, fill=fillheader, anchor=south west, minimum width=\paperwidth, minimum height=1.25cm, yshift=1cm] (box) at (current page.south west){};
    \node [anchor=west][xshift=1cm] (logo) at (box.west) {%
      {\includegraphics[height=0.9cm]{./signature-PIV.pdf}}
    };
    \node [align=left, anchor=west, text width=\paperwidth][xshift=0.5cm] (name1) at (logo.east) {%
      \fontsize{12}{14}\color{textheader}\bodyfont {Environnement et Lutte \\ contre les changements climatiques}
    };
    \node [anchor=east][xshift=-1cm] (page) at (box.east) {%
      \fontsize{14}{24}\color{textheader}%
      {\bodyfont \thepage~de~\pageref{LastPage}}
    };
    \node [anchor=north west, xshift=1cm] (note) at (box.south west) {%
      \notefont{Créé le 2021-04-28, MELCC, Direction de l'eau potable et des 
                eaux souterraines}
    };
  \end{tikzpicture}
}

\RequirePackage{atbegshi}
\newcommand{\createfooterheader}[2]{%
    \AtBeginShipout{\AtBeginShipoutAddToBox{%
      \header{#1}{#2}
      \footer
    }}
}
