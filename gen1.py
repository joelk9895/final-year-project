import os

content = r"""\documentclass{beamer}
\usetheme{Madrid}
\usepackage{graphicx}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{comment}
\usepackage{amsmath}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{longtable}
\usetikzlibrary{patterns, shapes.geometric, arrows, positioning}

\setbeamertemplate{footline}{}
\setbeamertemplate{navigation symbols}{}

\title[EV Charging Station]{Edge-Driven Semi-Autonomous EV Charging Station Using Quantized Vision Models and Real-Time License Plate Recognition}
\author{
  Izz Al Din Noufel Mukthar \quad MDL22CSBS026\\
  Joel K George \quad MDL22CSBS029\\
  Sarah Ann Pothen \quad MDL22CSBS050\\
  Shahma Fathima \quad MDL22CSBS053
}
\institute{Department of Computer Engineering\\Model Engineering College}
\date{April 2026}

\begin{document}

% FRAME 1
\begin{frame}
    \titlepage
\end{frame}

% FRAME 2
\begin{frame}{Contents (1/3)}
  \tableofcontents[sections={1-4}]
\end{frame}

% FRAME 3
\begin{frame}{Contents (2/3)}
  \tableofcontents[sections={5-9}]
\end{frame}

% FRAME 4
\begin{frame}{Contents (3/3)}
  \tableofcontents[sections={10-14}]
\end{frame}

\section{Introduction}

% FRAME 5
\begin{frame}
    \centering
    \Huge Introduction
\end{frame}

% FRAME 6
\begin{frame}{Global EV Adoption Surge}
\begin{itemize}
    \item Unprecedented growth in Electric Vehicle (EV) adoption worldwide.
    \item Global EV sales surpassed 14 million units in 2023.
    \item Represents a massive year-over-year increase of 35\%.
    \item Cumulative number of EVs on roads crossed 40-million globally.
\end{itemize}
\end{frame}

% FRAME 7
\begin{frame}{Market Projections}
\begin{itemize}
    \item Projections: EVs will be $>60\%$ of new car sales by 2030 in leading markets.
    \item Key markets include China, Europe, and North America.
    \item This rapid expansion places enormous pressure on charging infrastructure.
\end{itemize}
\end{frame}

% FRAME 8
\begin{frame}{Infrastructure Strain}
\begin{itemize}
    \item The infrastructure must scale in capacity to meet demand.
    \item Crucially, it must also scale in usability and seamlessness.
    \item Current systems suffer from complex user-interaction paradigms.
\end{itemize}
\end{frame}

% FRAME 9
\begin{frame}{The Charging Experience Bottleneck}
\begin{itemize}
    \item High-power hardware (Level 2 AC, DC fast chargers) is deploying rapidly.
    \item However, the user experience at public stations remains a major bottleneck.
    \item The transition from petrol stations to EV charging stations introduces significant new friction.
\end{itemize}
\end{frame}

\section{The Indian Context}

% FRAME 10
\begin{frame}
    \centering
    \Huge The Indian Context
\end{frame}

% FRAME 11
\begin{frame}{Accelerated Adoption in India}
\begin{itemize}
    \item India has witnessed a dramatic acceleration in EV adoption.
    \item Propelled by the FAME II scheme (Faster Adoption and Manufacturing of EVs).
    \item EV sales surged from $\sim$5,000 units in 2018 to $>1.5$ million in 2023.
\end{itemize}
\end{frame}

% FRAME 12
\begin{frame}{Infrastructure Goals in India}
\begin{itemize}
    \item Government target: 30\% EV penetration by 2030.
    \item Current network: $\sim$12,000 public charging stations.
    \item Required expansion: An estimated 400,000 stations nationwide.
    \item Streamlining the charging process is essential for this scale.
\end{itemize}
\end{frame}

\section{Friction in Current Systems}

% FRAME 13
\begin{frame}
    \centering
    \Huge Current Friction Points
\end{frame}

% FRAME 14
\begin{frame}{The App-Centric Model}
\begin{itemize}
    \item The prevailing interaction model is entirely smartphone-centric.
    \item Drivers must engage with operator-specific smartphone applications.
    \item Requires downloading, registering, authenticating, and initiating sessions.
\end{itemize}
\end{frame}

% FRAME 15
\begin{frame}{Network Fragmentation}
\begin{itemize}
    \item The network is highly fragmented among competing operators.
    \item Operators like Tata Power, ChargeZone, and Ather Grid use distinct ecosystems.
    \item A driver visiting five networks must maintain five separate apps and wallets.
\end{itemize}
\end{frame}

\section{Motivation}

% FRAME 16
\begin{frame}
    \centering
    \Huge Motivation
\end{frame}

% FRAME 17
\begin{frame}{Motivation: App Fatigue}
\begin{itemize}
    \item Research: 67\% of EV drivers express frustration with multiple apps.
    \item Average public session involves 4--7 discrete digital interaction steps.
    \item Contrasts sharply with a traditional petrol pump "swipe and go" gesture.
\end{itemize}
\end{frame}

% FRAME 18
\begin{frame}{Motivation: Connectivity Dependency}
\begin{itemize}
    \item Cloud authentication inherently depends on stable cellular networks.
    \item Chargers are often in areas with poor cellular coverage:
    \begin{itemize}
        \item Underground parking garages.
        \item Remote highway rest stops.
    \end{itemize}
    \item Connection failures prevent session initiation entirely.
\end{itemize}
\end{frame}

% FRAME 19
\begin{frame}{Motivation: Accessibility Gaps}
\begin{itemize}
    \item App-centric models disproportionately exclude certain demographics.
    \item Elderly users or individuals with limited smartphone proficiency face barriers.
    \item Tourists without local SIM cards cannot access regional networks.
\end{itemize}
\end{frame}

% FRAME 20
\begin{frame}{Motivation: Edge Computing Maturity}
\begin{itemize}
    \item Edge computing brings intelligence directly to the point of service.
    \item Affordable single-board computers (Raspberry Pi 4B) now provide sufficient power.
    \item Quantization enables massive models to run efficiently on resource-constrained devices.
\end{itemize}
\end{frame}

\section{Objectives}

% FRAME 21
\begin{frame}
    \centering
    \Huge Objectives
\end{frame}

% FRAME 22
\begin{frame}{Primary Objectives}
\begin{itemize}
    \item Develop an Automatic License Plate Recognition (ALPR) system for edge deployment.
    \item Quantize a YOLOv11 object detection model from FP32 to INT8 precision.
    \item Achieve $>85\%$ model size reduction with $<2\%$ accuracy degradation.
\end{itemize}
\end{frame}

% FRAME 23
\begin{frame}{Technical Objectives}
\begin{itemize}
    \item Resolve INT8 operator incompatibility on ARM via custom ONNX graph patching.
    \item Implement an asynchronous WebSocket server for real-time frame processing.
    \item Build a native iOS client application for camera streaming with live overlays.
\end{itemize}
\end{frame}

\section{Scope}

% FRAME 24
\begin{frame}
    \centering
    \Huge Scope
\end{frame}

% FRAME 25
\begin{frame}{Scope \& Boundaries}
\begin{itemize}
    \item \textbf{In Scope:} Model fine-tuning, ONNX graph patching, Raspberry Pi edge server, iOS client app, SQLite DB integration.
    \item \textbf{Out of Scope:} Physical charger hardware integration (GPIO), production-grade payment gateways, non-Indian plate formats.
\end{itemize}
\end{frame}
"""
with open("presentation_80.tex", "w") as f:
    f.write(content)
