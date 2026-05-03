import os

content = r"""
\section{Problem Statement}

% FRAME 26
\begin{frame}
    \centering
    \Huge Problem Statement
\end{frame}

% FRAME 27
\begin{frame}{The Core Issue}
\begin{itemize}
    \item Current stations require multi-step digital workflows via smartphone apps.
    \item Creates significant friction, connectivity dependencies, and accessibility barriers.
    \item Identifying vehicles and processing payments is slow and cumbersome.
\end{itemize}
\end{frame}

% FRAME 28
\begin{frame}{Typical Charging Workflow (Steps 1-4)}
\begin{enumerate}
    \item Driver locates an available charger.
    \item Identifies the specific charging network operator.
    \item Downloads the operator's app (if not installed).
    \item Creates an account and adds a payment method.
\end{enumerate}
\end{frame}

% FRAME 29
\begin{frame}{Typical Charging Workflow (Steps 5-8)}
\begin{enumerate}
    \setcounter{enumi}{4}
    \item Opens app, scans QR code on charger.
    \item Selects charging plan and confirms payment via cloud.
    \item Charging session begins.
    \item Receives notification upon completion.
\end{enumerate}
\end{frame}

% FRAME 30
\begin{frame}{The Research Question}
\begin{block}{Question}
How can we develop a system that automatically identifies registered vehicles upon arrival and authorizes charging sessions without requiring any smartphone interaction, while deploying efficiently on resource-constrained edge hardware?
\end{block}
\end{frame}

\section{Proposed Solution}

% FRAME 31
\begin{frame}
    \centering
    \Huge Proposed Solution
\end{frame}

% FRAME 32
\begin{frame}{Our Edge-Driven Solution}
\begin{itemize}
    \item A semi-autonomous EV charging station powered by edge-based computer vision.
    \item Replaces the 8-step manual workflow with a single autonomous step.
    \item \textbf{The New Workflow:} Vehicle parks $\to$ Camera detects plate $\to$ Edge server authorizes charging.
\end{itemize}
\end{frame}

% FRAME 33
\begin{frame}{Solution Pipeline Overview}
\begin{itemize}
    \item Live camera feed captured by client (iOS/CSI).
    \item Transmitted via low-latency WebSocket to Raspberry Pi.
    \item YOLOv11 INT8 detects license plates.
    \item EasyOCR extracts alphanumeric characters.
    \item Regex validation and SQLite lookup finalize authorization.
\end{itemize}
\end{frame}

% FRAME 34
\begin{frame}{Hardware Deployment}
\begin{itemize}
    \item The entire processing pipeline runs on a \$55 Raspberry Pi 4B.
    \item No cloud server required for the critical authentication window.
    \item Operates securely within the station's Local Area Network.
\end{itemize}
\end{frame}

% FRAME 35
\begin{frame}{Zero-Interaction Experience}
\begin{itemize}
    \item True "Park and Plug" experience.
    \item No smartphones, RFID cards, or QR codes needed.
    \item High degree of privacy as visual data never leaves the edge device.
\end{itemize}
\end{frame}

\section{Key Innovations}

% FRAME 36
\begin{frame}
    \centering
    \Huge Key Innovations
\end{frame}

% FRAME 37
\begin{frame}{Key Innovations Highlighted}
\begin{itemize}
    \item \textbf{Aggressive Quantization:} Reducing YOLOv11 from 200 MB (FP32) to 25 MB (INT8) while maintaining $\sim$93\% accuracy.
    \item \textbf{ONNX Graph Patching:} Programmatically circumventing ARM CPU architecture limitations by shifting signed INT8 tensors to unsigned UINT8 directly within the protobuf graph.
\end{itemize}
\end{frame}

% FRAME 38
\begin{frame}{Comparison with Existing Approaches}
\begin{table}[]
\centering
\renewcommand{\arraystretch}{1.3}
\footnotesize
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Feature} & \textbf{App-Based} & \textbf{RFID} & \textbf{Cloud ALPR} & \textbf{Ours} \\
\hline
Smartphone Reqd. & Yes & No & No & \textbf{No} \\
Cloud Conn.      & Reqd. & Reqd. & Reqd. & \textbf{Not Reqd.} \\
User Steps       & 5--8 & 2--3 & 0 & \textbf{0} \\
Privacy          & Low & Low & Low & \textbf{High} \\
Latency          & 3--10s & 1--2s & 2--5s & \textbf{$<$1s} \\
Hardware Cost    & Low & Medium & High & \textbf{Low} \\
\hline
\end{tabular}
\end{table}
\end{frame}

\section{Literature Survey}

% FRAME 39
\begin{frame}
    \centering
    \Huge Literature Survey
\end{frame}

% FRAME 40
\begin{frame}{Literature Survey (1/4)}
\begin{itemize}
    \item \textbf{Knowledge Distillation:} Acharya et al. (2024) and Cantini et al. (2024) demonstrated transferring complex logic from massive teacher models to smaller student models.
    \item \textit{Relevance:} Validates that aggressive model compression can retain task-critical reasoning, supporting our INT8 quantization strategy.
\end{itemize}
\end{frame}

% FRAME 41
\begin{frame}{Literature Survey (2/4)}
\begin{itemize}
    \item \textbf{Quantization \& Pruning:} Kim (2023) and Lang et al. (2023) provided hardware-aware surveys of PTQ vs. QAT and uniform quantization.
    \item \textit{Relevance:} Dictated our choice of Post-Training Static Quantization (PTQ) to leverage ARM NEON SIMD optimizations without retraining.
\end{itemize}
\end{frame}

% FRAME 42
\begin{frame}{Literature Survey (3/4)}
\begin{itemize}
    \item \textbf{Edge Computing Paradigm:} Shi et al. (2016) and Yi et al. (2025) highlighted latency, bandwidth, and privacy advantages of processing at the edge.
    \item \textit{Relevance:} Forms the architectural foundation of our zero-cloud-dependency authentication pipeline.
\end{itemize}
\end{frame}

% FRAME 43
\begin{frame}{Literature Survey (4/4)}
\begin{itemize}
    \item \textbf{ALPR \& EV Behavior:} Laroca et al. (2021) identified YOLO as optimal for ALPR. Hardman et al. (2018) identified "app fragmentation" as a primary pain point for EV users.
    \item \textit{Relevance:} Directly motivates the project's core user-experience overhaul and algorithm selection.
\end{itemize}
\end{frame}

\section{System Architecture}

% FRAME 44
\begin{frame}
    \centering
    \Huge System Architecture
\end{frame}

% FRAME 45
\begin{frame}{Architecture Flowchart}
\begin{center}
\begin{tikzpicture}[node distance=2.2cm, auto,
    block/.style={rectangle, draw, fill=blue!10, text width=2.5cm, text centered, rounded corners, minimum height=1cm, font=\small},
    db/.style={cylinder, draw, shape border rotate=90, aspect=0.25, fill=yellow!20, text centered, font=\small},
    arrow/.style={thick,->,>=stealth}]
    \node [block] (cam) {Camera Node\\(iOS/CSI)};
    \node [block, right of=cam, node distance=4cm] (pi) {Raspberry Pi 4B\\Edge Server};
    \node [db, right of=pi, node distance=3.5cm, text width=1.5cm, minimum height=1cm] (dbnode) {SQLite\\Database};
    \node [block, below of=pi, node distance=2.5cm, fill=green!15] (charger) {EV Charger\\Relay Control};
    
    \draw [arrow] (cam) -- node[above, font=\tiny]{Binary JPEG} node[below, font=\tiny]{WebSocket} (pi);
    \draw [arrow, <->] (pi) -- node[above, font=\tiny]{SQL Query} (dbnode);
    \draw [arrow] (pi) -- node[right, font=\tiny]{JSON Auth} (charger);
\end{tikzpicture}
\end{center}
\end{frame}

% FRAME 46
\begin{frame}{Use Case Diagram (1/2)}
\begin{center}
\begin{tikzpicture}[
    actor/.style={circle, draw, minimum size=0.6cm, font=\small},
    usecase/.style={ellipse, draw, fill=blue!8, text width=2.5cm, text centered, minimum height=0.6cm, font=\small},
    arrow/.style={->, thick}
]
\node[actor] (driver) at (0,0) {EV Driver};
\node[usecase] (park) at (4, 1) {Park at Station};
\node[usecase] (view) at (4, -1) {View Feedback UI};
\draw[arrow] (driver) -- (park);
\draw[arrow] (driver) -- (view);
\end{tikzpicture}
\end{center}
\end{frame}

% FRAME 47
\begin{frame}{Use Case Diagram (2/2)}
\begin{center}
\begin{tikzpicture}[
    actor/.style={circle, draw, minimum size=0.6cm, font=\small},
    usecase/.style={ellipse, draw, fill=blue!8, text width=2.5cm, text centered, minimum height=0.6cm, font=\small},
    arrow/.style={->, thick}
]
\node[actor] (admin) at (0,0) {System Admin};
\node[usecase] (manage) at (4, 1) {Manage Database};
\node[usecase] (update) at (4, -1) {Update Models};
\draw[arrow] (admin) -- (manage);
\draw[arrow] (admin) -- (update);
\end{tikzpicture}
\end{center}
\end{frame}

% FRAME 48
\begin{frame}{Level 0 DFD (Context Diagram) (1/2)}
\begin{itemize}
    \item System establishes boundary limits.
    \item Inputs: JPEG frames from the camera.
    \item Processing: ALPR and authorization logic.
\end{itemize}
\end{frame}

% FRAME 49
\begin{frame}{Level 0 DFD (Context Diagram) (2/2)}
\begin{center}
\begin{tikzpicture}[
    entity/.style={rectangle, draw, fill=gray!10, text width=1.5cm, text centered, minimum height=0.8cm, font=\small},
    process/.style={circle, draw, fill=blue!15, minimum size=2.5cm, text centered, font=\small},
    arrow/.style={->, thick}
]
\node[entity] (camera) at (-3.5, 0) {Camera};
\node[process] (system) at (0, 0) {ALPR System};
\node[entity] (db) at (3.5, 0) {Vehicle DB};
\draw[arrow] (camera) -- node[above, font=\tiny]{JPEG} (system);
\draw[arrow, <->] (system) -- node[above, font=\tiny]{Verify} (db);
\end{tikzpicture}
\end{center}
\end{frame}

% FRAME 50
\begin{frame}{Level 1 DFD (Internal Data Flow) (1/2)}
\begin{itemize}
    \item Breaks the system into distinct sequential operations.
    \item Operations: Decode $\to$ Detect $\to$ OCR $\to$ Validate $\to$ Query.
\end{itemize}
\end{frame}
"""
with open("presentation_80.tex", "a") as f:
    f.write(content)
