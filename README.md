# LLM_Benchmark_Visual-Statistic

**Hintergrund**
Das vorliegende Skript dient der Automatisierung der statistischen Tests und der Erstellung der grafischen Elemente. Sie wurden im Rahmen einer Masterarbeit genutzt. Alle weiteren relevanten Informationen sind in der schriftlichen Ausführung enthalten.

**Voraussetzungen:**

*   **Docker:** Das Skript ist für die Ausführung in einem Docker-Container konzipiert.

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

*   **Konsolenausgabe:** Zeigt die Ergebnisse der Tests.

**Hinweis:**

*   Um das Visual-Skript ausführen zu lassen, kann im Dockerfile  (`Statistical_Tests.py`) durch (`Visuals.py`) ersetzt werden.
*   Ersetze im Skript `plt.show()` durch `plt.savefig('name_der_datei.png')`, um die Plots in Dateien zu speichern. 
*   Manche Funktionsaufrufe wurden speziell angepasst um eine bestimmte Grafik zu erzeugen. Die genauen Details, wie sich die jeweilige Grafik zusammensetzt sind in der Masterarbeit unter der jeweiligen Abbildung zu finden.
*   Docker-Desktop muss auf dem Rechner installiert und aktiv sein.
