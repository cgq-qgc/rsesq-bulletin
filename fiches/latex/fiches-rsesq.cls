% -----------------------------------------------------------------------------
% Copyright © Institut National de la Recherche Scientifique (INRS)
% https://github.com/cgq-qgc/rsesq-bulletin
%
% Created by Jean-Sébastien Gosselin
% jean-sebastien.gosselin@inrs.ca
%
% This work is licensed under the terms of the CC BY 4.0 License as published
% by the Creative Commons nonprofit organization. For more details, see the
% CC BY 4.0 License at https://creativecommons.org/licenses/by/4.0/.
% -----------------------------------------------------------------------------

\ProvidesClass{fiches-rsesq}[2021/10/05 Fiches Signalétique RSESQ]
\NeedsTeXFormat{LaTeX2e}

\LoadClass[10pt, 5p, onecolumn]{report}

\pagestyle{empty}

\newcommand{\classleftmargin}{1cm}
\newcommand{\classrightmargin}{1cm}
\newcommand{\classtopmargin}{4cm}
\newcommand{\classbottommargin}{3.75cm}
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

\newcommand{\normaltextfont}{\fontsize{10}{12} %
            \selectfont\bodyfont\color{black}}


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

\newcommand{\header}[1]{%
  \begin{tikzpicture}[remember picture, overlay]
    \node [rectangle, fill=fillheader, anchor=north, minimum 
    width=\paperwidth, minimum height=2cm, yshift=-1cm] (box) at (current 
    page.north){};
    \node [anchor=center, yshift=0pt] (name) at (box) {%
      \fontsize{40pt}{72pt}\color{textheader}%
      \bodyfont Station #1
    };
  \end{tikzpicture}
  \vspace{-2em}
}

\newcommand{\footer}[0]{%
  \begin{tikzpicture}[remember picture, overlay]
    \node [rectangle, fill=fillheader, anchor=south west, minimum 
    width=\paperwidth, minimum height=1.75cm, yshift=1cm] (box) at (current 
    page.south west){};
    \ifnum \thepage=1 {%
      \node [anchor=east][xshift=-1cm] (logo) at (box.east) {%
        {\includegraphics[height=0.9cm]{./signature-PIV.pdf}}
      };
    }\fi
    \ifnum \thepage=\pageref{LastPage} {%
      \node [anchor=west][xshift=1cm] (logo) at (box.west) {%
        {\includegraphics[height=0.9cm]{./signature-PIV.pdf}}
      };
    }\fi
    \node [anchor=center][xshift=0cm] (page) at (box.center) {%
      \fontsize{18}{24}\color{textheader}%
      {\bodyfont \thepage~de~\pageref{LastPage}}
    };
    \ifnum \thepage=1 {%
      \node [anchor=north west, xshift=1cm] (note) at (box.south west) {%
        \notefont{Créé le 2021-04-28, MELCC, Direction de l'eau potable et des 
                  eaux souterraines}
      };
    }\fi
  \end{tikzpicture}
}

\RequirePackage{atbegshi}
\newcommand{\createfooterheader}[1]{%
    \AtBeginShipout{\AtBeginShipoutAddToBox{%
      \header{#1}
      \footer
    }}
}
