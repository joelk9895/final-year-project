import os

content = r"""
% FRAME 51
\begin{frame}{Level 1 DFD (Internal Data Flow) (2/2)}
\begin{center}
\begin{tikzpicture}[
    process/.style={circle, draw, fill=blue!10, minimum size=1.5cm, text centered, font=\tiny, text width=1.3cm},
    store/.style={rectangle, draw, fill=yellow!10, text width=1.8cm, text centered, font=\tiny},
    arrow/.style={->, thick, font=\tiny}
]
\node[process] (decode) at (0, 0) {1.0 Decode};
\node[process] (yolo) at (2.5, 0) {2.0 YOLO};
\node[process] (ocr) at (5, 0) {3.0 OCR};
\node[process] (validate) at (5, -2) {4.0 Validate};
\node[process] (lookup) at (2.5, -2) {5.0 DB Lookup};
\node[store] (db) at (2.5, -4) {D1: Vehicles};

\draw[arrow] (-1.5, 0) -- node[above]{JPEG} (decode);
\draw[arrow] (decode) -- node[above]{BGR} (yolo);
\draw[arrow] (yolo) -- node[above]{Crop} (ocr);
\draw[arrow] (ocr) -- node[right]{Text} (validate);
\draw[arrow] (validate) -- node[above]{Plate} (lookup);
\draw[arrow, <->] (lookup) -- (db);
\end{tikzpicture}
\end{center}
\end{frame}

% FRAME 52
\begin{frame}{Database ER Diagram}
\begin{center}
\begin{tikzpicture}[
    entity/.style={rectangle, draw, fill=blue!10, text width=2.5cm, text centered, minimum height=1cm, font=\small},
    attr/.style={ellipse, draw, fill=green!10, text width=1.8cm, text centered, font=\tiny}
]
\node[entity] (vehicle) at (0, 0) {\textbf{Vehicle}};
\node[attr] (plate) at (-3, 1) {plate\_number (PK)};
\node[attr] (owner) at (0, 1.8) {owner\_name};
\node[attr] (balance) at (3, 1) {balance};
\node[attr] (phone) at (-3, -1) {phone};
\node[attr] (status) at (3, -1) {status};
\draw (vehicle) -- (plate);
\draw (vehicle) -- (owner);
\draw (vehicle) -- (balance);
\draw (vehicle) -- (phone);
\draw (vehicle) -- (status);
\end{tikzpicture}
\end{center}
\end{frame}

\section{Modules}

% FRAME 53
\begin{frame}
    \centering
    \Huge Modules \& Processing
\end{frame}

% FRAME 54
\begin{frame}{Module 1: Camera Streaming (iOS)}
\begin{itemize}
    \item Uses AVFoundation \texttt{AVCaptureSession} at \texttt{.vga640x480}.
    \item Flattens raw buffer to \texttt{CIImage} and compresses to JPEG (quality 0.6).
    \item Transmits binary data to the Pi.
    \item Uses SwiftUI \& \texttt{Combine} to dynamically render green bounding boxes and OCR text on screen.
\end{itemize}
\end{frame}

% FRAME 55
\begin{frame}{Module 2: Edge Inference Server}
\begin{itemize}
    \item Decodes frames via OpenCV (\texttt{cv2.imdecode}) in a thread pool.
    \item Prepares frames for YOLO (resize to $640 \times 640$, normalize, NCHW).
    \item Executes INT8 model using \texttt{ONNX Runtime} (aarch64).
    \item Employs Non-Maximum Suppression (NMS) to eliminate duplicate boxes.
\end{itemize}
\end{frame}

% FRAME 56
\begin{frame}{Module 3 \& 4: OCR, Validate, DB}
\begin{itemize}
    \item \textbf{OCR:} EasyOCR (CRNN) processes cropped plate bounding boxes.
    \item \textbf{Validate:} Cleans whitespace and tests against strict Regex pattern: \texttt{\^{}[A-Z]\{2\}[0-9]\{2\}[A-Z]\{1,2\}[0-9]\{4\}\$}
    \item \textbf{Database:} SQLite \texttt{SELECT owner, balance FROM vehicles WHERE plate = ? AND status = 'active'}
\end{itemize}
\end{frame}

% FRAME 57
\begin{frame}{Detection Pipeline Flowchart}
\begin{center}
\begin{tikzpicture}[node distance=1.5cm, auto,
    block/.style={rectangle, draw, fill=blue!10, text width=6cm, text centered, rounded corners, minimum height=0.6cm, font=\footnotesize},
    arrow/.style={thick,->,>=stealth}]
    \node [block] (rcv) {Receive JPEG via WebSocket};
    \node [block, below of=rcv, node distance=1.2cm] (yolo) {YOLO INT8 Inference};
    \node [block, below of=yolo, node distance=1.2cm] (crop) {Crop Bounding Box $\to$ EasyOCR};
    \node [block, below of=crop, node distance=1.2cm] (val) {Regex Validation};
    \node [block, below of=val, node distance=1.2cm] (db) {SQLite Lookup};
    \node [block, below of=db, node distance=1.2cm, fill=green!20] (auth) {Return Auth JSON Payload};
    \draw [arrow] (rcv) -- (yolo);
    \draw [arrow] (yolo) -- (crop);
    \draw [arrow] (crop) -- (val);
    \draw [arrow] (val) -- (db);
    \draw [arrow] (db) -- (auth);
\end{tikzpicture}
\end{center}
\end{frame}

% FRAME 58
\begin{frame}{WebSocket Concurrency Model}
\begin{center}
\begin{tikzpicture}[node distance=1.5cm, auto,
    block/.style={rectangle, draw, fill=orange!10, text width=3.5cm, text centered, rounded corners, minimum height=1cm, font=\footnotesize},
    arrow/.style={thick,<->,>=stealth}]
    \node [block] (client1) {Client 1\\(Lane A)};
    \node [block, right of=client1, node distance=4.5cm, fill=blue!10] (server) {\texttt{asyncio} Event Loop\\WebSocket Server};
    \node [block, right of=server, node distance=4.5cm] (client2) {Client 2\\(Lane B)};
    
    \draw [arrow] (client1) -- (server);
    \draw [arrow] (client2) -- (server);
\end{tikzpicture}
\end{center}
\vspace{0.3cm}
\begin{itemize}
    \item \texttt{asyncio} prevents blocking while waiting for network I/O.
    \item CPU-heavy CV tasks are pushed to \texttt{run\_in\_executor} thread pools.
\end{itemize}
\end{frame}

\section{Theoretical Background}

% FRAME 59
\begin{frame}
    \centering
    \Huge YOLO Object Detection
\end{frame}

% FRAME 60
\begin{frame}{YOLO Paradigm}
\begin{itemize}
    \item "You Only Look Once" changed computer vision from classification to regression.
    \item Predicts bounding box coordinates and probabilities in one forward pass.
    \item Extremely fast compared to Region Proposal Networks (e.g., Faster R-CNN).
    \item Implicitly captures global context, reducing false positives in background.
\end{itemize}
\end{frame}

% FRAME 61
\begin{frame}{YOLO Grid Prediction}
\begin{itemize}
    \item Divides image into an $S \times S$ grid.
    \item Cell responsible if object center falls within it.
    \item Each cell predicts $B$ bounding boxes ($x,y,w,h$) + Confidence.
    \item $Validation Score = Pr(\text{Object}) \times \text{IoU}_{pred}^{truth}$
\end{itemize}
\end{frame}

% FRAME 62
\begin{frame}{YOLO Loss Function (1/2)}
Multi-part loss balancing localization and classification:
\begin{equation}
    \mathcal{L} = \mathcal{L}_{coord} + \mathcal{L}_{confidence} + \mathcal{L}_{classification}
\end{equation}
\end{frame}

% FRAME 63
\begin{frame}{YOLO Loss Function (2/2)}
\begin{equation}
\begin{aligned}
    \mathcal{L} = \;&\lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
    &+ \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 \dots \right] \\
    &+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 + \dots
\end{aligned}
\end{equation}
\end{frame}

% FRAME 64
\begin{frame}
    \centering
    \Huge Quantization Theory
\end{frame}

% FRAME 65
\begin{frame}{Motivation for Quantization}
\begin{itemize}
    \item FP32 models (200MB) exceed edge cache sizes and memory bandwidth limits.
    \item Mapping floating-point weights to 8-bit integers (INT8) shrinks size by $4\times$.
    \item Replaces costly FP MAC operations with integer arithmetic.
\end{itemize}
\end{frame}

% FRAME 66
\begin{frame}{Affine Quantization Math (1/2)}
Maps real values ($x$) to integers ($x_q$) via Scale ($S$) and Zero-Point ($Z$):
\begin{equation}
    x \approx S \times (x_q - Z)
\end{equation}
Forward mapping is clamped:
\begin{equation}
    x_q = \text{clamp}\left( \text{round}\left( \frac{x}{S} \right) + Z, Q_{min}, Q_{max} \right)
\end{equation}
\end{frame}

% FRAME 67
\begin{frame}{Affine Quantization Math (2/2)}
Matrix multiplication ($y = w \cdot x + b$) becomes purely integer:
\begin{equation}
    y_q = \frac{S_w S_x}{S_y} \sum_{i} (w_{q,i} - Z_w)(x_{q,i} - Z_x) + \frac{b}{S_y} + Z_y
\end{equation}
(The multiplier $\frac{S_w S_x}{S_y}$ is pre-computed as a bit-shift offset).
\end{frame}

% FRAME 68
\begin{frame}{Quantization Pipeline Steps}
\begin{enumerate}
    \item \textbf{Export:} PyTorch $\to$ ONNX FP32.
    \item \textbf{Calibration:} Pass 100 representative plates to determine activation min/max.
    \item \textbf{Static Quantization:} Produce INT8 ONNX graph.
    \item \textbf{Patching:} Fix ARM operator incompatibilities.
\end{enumerate}
\end{frame}

\section{Implementation Algorithms}

% FRAME 69
\begin{frame}{Algorithm: Edge Server Handler}
\begin{algorithm}[H]
\caption{WebSocket Connection Handler}
\begin{algorithmic}[1]
\State $image \gets \text{cv2.imdecode}(binary\_frame)$
\State $boxes \gets \text{YOLO.run}(image)$
\For{each $box$ in $boxes$}
    \If{$\text{IoU}(box, prev) > 0.45$} \text{ // NMS}
        \State \textbf{continue}
    \EndIf
    \State $crop \gets \text{image}[y_1:y_2, x_1:x_2]$
    \State $text \gets \text{EasyOCR.readtext}(crop)$
    \State \Call{ValidateAndVerify}{$text$}
\EndFor
\end{algorithmic}
\end{algorithm}
\end{frame}

% FRAME 70
\begin{frame}{Algorithm: Database Verification}
\begin{algorithm}[H]
\caption{Validate and Verify License Plate}
\begin{algorithmic}[1]
\Function{ValidateAndVerify}{$text$}
    \State $clean \gets \text{Uppercase and Remove Spaces}(text)$
    \If{$\text{RegexMatch}(clean, \text{"\^[A-Z]\{2\}[0-9]\{2\}..."})$}
        \State $db \gets \text{SQLiteConnection()}$
        \State $record \gets db.\text{execute("SELECT owner FROM vehicles WHERE plate = ?", } clean)$
        \If{$record \neq \text{Null}$}
            \State \Return \text{JSON Auth Payload}
        \EndIf
    \EndIf
    \State \Return \text{Null}
\EndFunction
\end{algorithmic}
\end{algorithm}
\end{frame}

% FRAME 71
\begin{frame}{Algorithm: ONNX Graph Patching}
\begin{algorithm}[H]
\caption{Fix ConvInteger INT8 for ARM CPUs}
\begin{algorithmic}[1]
\For{each $node$ in $ONNX\_Graph$}
    \If{$node.type == \text{"ConvInteger"}$}
        \State $weights \gets node.inputs[1]$
        \If{$weights.dataType == \text{INT8}$}
            \State $uint8\_weights \gets (weights + 128).cast(\text{UINT8})$
            \State $node.replace(weights, uint8\_weights)$
        \EndIf
    \EndIf
\EndFor
\end{algorithmic}
\end{algorithm}
\end{frame}

\section{Testing \& Results}

% FRAME 72
\begin{frame}
    \centering
    \Huge Testing \& Results
\end{frame}

% FRAME 73
\begin{frame}{Performance Benchmarks}
\begin{table}[]
\centering
\renewcommand{\arraystretch}{1.2}
\footnotesize
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{FP32 Model} & \textbf{INT8 Model} \\
\midrule
Model Size (MB) & 200 & 25 \\
Avg. Inference (ms) & 1842 & 487 \\
Speedup Factor & 1.0$\times$ & 3.78$\times$ \\
mAP@0.5 & 0.942 & 0.931 \\
\bottomrule
\end{tabular}
\end{table}
\end{frame}

% FRAME 74
\begin{frame}{Graph: Inference Latency}
\begin{center}
\begin{tikzpicture}
\begin{axis}[
    ybar, bar width=30pt,
    ylabel={Inference Time (ms)},
    symbolic x coords={FP32, INT8},
    xtick=data,
    ymin=0, ymax=2000,
    nodes near coords,
    width=7cm, height=5cm,
    fill=red!40
]
\addplot coordinates {(FP32, 1842) (INT8, 487)};
\end{axis}
\end{tikzpicture}
\end{center}
\end{frame}

% FRAME 75
\begin{frame}{Graph: Peak RAM Usage}
\begin{center}
\begin{tikzpicture}
\begin{axis}[
    ybar, bar width=30pt,
    ylabel={RAM Usage (MB)},
    symbolic x coords={FP32, INT8},
    xtick=data,
    ymin=0, ymax=1400,
    nodes near coords,
    width=7cm, height=5cm,
    fill=blue!40
]
\addplot coordinates {(FP32, 1284) (INT8, 478)};
\end{axis}
\end{tikzpicture}
\end{center}
\end{frame}

% FRAME 76
\begin{frame}{Graph: End-to-End Latency Breakdown}
\begin{center}
\begin{tikzpicture}
\begin{axis}[
    xbar stacked,
    symbolic y coords={Pipeline},
    ytick=data,
    xlabel={Time (ms)},
    xmin=0, xmax=850,
    height=3.5cm, width=10cm,
    legend style={at={(0.5,-0.5)}, anchor=north, legend columns=4, font=\tiny}
]
\addplot[fill=blue!40] coordinates {(45,Pipeline)};  \addlegendentry{Decode (5\%)}
\addplot[fill=red!40] coordinates {(487,Pipeline)}; \addlegendentry{YOLO (60\%)}
\addplot[fill=green!40] coordinates {(245,Pipeline)}; \addlegendentry{OCR (30\%)}
\addplot[fill=yellow!60] coordinates {(34,Pipeline)}; \addlegendentry{Misc (5\%)}
\end{axis}
\end{tikzpicture}
\end{center}
\end{frame}

% FRAME 77
\begin{frame}{OCR Accuracy Degradation}
\begin{table}[]
\centering
\renewcommand{\arraystretch}{1.2}
\footnotesize
\begin{tabular}{|l|c|c|}
\hline
\textbf{Condition} & \textbf{Character Acc.} & \textbf{Plate Acc.} \\
\hline
Good Lighting & 98.2\% & 94.5\% \\
Low Light & 89.7\% & 78.4\% \\
Distance $>$4m & 85.4\% & 71.8\% \\
\hline
\end{tabular}
\end{table}
\end{frame}

\section{Future Scope \& Conclusion}

% FRAME 78
\begin{frame}
    \centering
    \Huge Future Scope
\end{frame}

% FRAME 79
\begin{frame}{Future Enhancements}
\begin{itemize}
    \item \textbf{GPIO Relay Integration:} Actually trigger physical high-voltage contactors.
    \item \textbf{Cloud Sync:} SQLite to Firebase migration for cross-station roaming.
    \item \textbf{Lighter OCR:} Swap EasyOCR for PaddleOCR-Lite to shave off 100ms.
    \item \textbf{OCPP Integration:} Interface with standard charging protocols.
\end{itemize}
\end{frame}

% FRAME 80
\begin{frame}{Conclusion}
\begin{itemize}
    \item Edge computer vision successfully transforms the EV charging workflow.
    \item Quantization unlocks deep learning on \$55 hardware.
    \item Sub-second latency delivers true "Park and Plug" autonomy.
\end{itemize}
\vspace{0.5cm}
\begin{center}
    \Large \textbf{Thank You!}
\end{center}
\end{frame}

\end{document}
"""
with open("presentation_80.tex", "a") as f:
    f.write(content)
