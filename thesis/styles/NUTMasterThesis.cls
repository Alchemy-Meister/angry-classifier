\NeedsTeXFormat{LaTeX2e}[1999/01/01]
\ProvidesClass{NUTMasterThesis}[2016/10/11 v0.1 Master Thesis Template for Nagaoka University of Technology]

% Package to include images.
\RequirePackage{graphicx}

\RequirePackage{fontspec}
  % Use standard TeX ligature logic
  \defaultfontfeatures{Ligatures=TeX}

% Enable use of XeTeX CJK fonts for Japanese text if needed
\RequirePackage[indentfirst=false]{xeCJK}

% Load base template.
\LoadClass[a4paper, 12pt]{report}

% Defines document's default margins.
\RequirePackage[tmargin=3cm,bmargin=3cm,lmargin=3.5cm,rmargin=2.5cm]{geometry}

\RequirePackage{placeins}