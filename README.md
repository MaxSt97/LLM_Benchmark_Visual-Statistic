# LLM_Benchmark_Visual-Statistic

**Hintergrund**
Das vorliegende Skript dient der Automatisierung der statistischen Tests und der Erstellung der grafischen Elemente. 

**Ausführung:**

1.  Docker-Image erstellen:
    ```bash
    docker build -t bigcodebench-fehlercheck .
    ```
2.  Docker-Container starten:
    ```bash
    docker run -it bigcodebench-fehlercheck
    ```
**Ausgabe:**

*   **Konsolenausgabe:** Zeigt den die Ergebnisse der Tests.

**Hinweis:**

*   Um das Visual-Skript ausführen zu lassen, kann im Dockerfile  (`Statistical_Tests.py`) durch (`Visuals.py`) ersetzt werden.
