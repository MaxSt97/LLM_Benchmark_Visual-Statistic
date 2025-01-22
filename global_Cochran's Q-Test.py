import os

import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
import statsmodels.stats.multitest as multi


def load_and_prepare_data(paths):
    """
    Lädt die CSV-Dateien aus den angegebenen Verzeichnissen (für t=0, t=0.5, t=1)
    und erstellt ein gemeinsames DataFrame. Dabei wird:
      - 'run' mit dem Ordnernamen belegt
      - 'temperature' aus dem Unterordner entnommen
      - 'Modellname' aus dem Dateinamen bestimmt
      - nur Iteration 1 für 'solution' verwendet
    """
    df_list = []

    for path in paths:
        run_name = os.path.basename(path)
        for temp_folder in ["t=0", "t=0.5", "t=1"]:
            temp_path = os.path.join(path, temp_folder)
            if not os.path.exists(temp_path):
                continue  # Falls ein Ordner fehlt, Überspringen
            for filename in os.listdir(temp_path):
                if filename.endswith(".csv"):
                    filepath = os.path.join(temp_path, filename)
                    # Lies zunächst die ersten 61 Zeilen (anpassbar je nach Struktur)
                    df = pd.read_csv(filepath, nrows=61)

                    df["temperature"] = float(temp_folder.split("=")[1])
                    df["Modellname"] = filename.replace("log_", "").replace(".csv", "")
                    df["run"] = run_name
                    df_list.append(df)

    # Kombiniere die DataFrames
    df = pd.concat(df_list, ignore_index=True)

    # Kennzeichnung von Blackbox- und Open-Source-Modelle (Beispiel)
    df["blackbox"] = df["Modellname"].apply(
        lambda x: False
        if x in ["deepseek_deepseek-chat", "qwen_qwen-2.5-coder-32b-instruct"]
        else True
    )
    # Spalte (Fehler) hinzufügen, die mit dem letzten Element des Strings aus der Spalte Task befüllt wird.
    # Getrennt wird am äußersten rechten "_"
    df["error"] = df["Task"].apply(lambda x: x.rsplit("_", 1)[-1])

    # Spalte Task hinzufügen ohne error prefix
    df["Task_error_combination"] = df["Task"].apply(lambda x: x.rsplit("_", 1)[0])

    # Nur Iteration 1 als Indikator für Erfolg (0/1)
    df["solution"] = df["Iteration 1"]

    # Solution Iteration Spalten hinzufügen
    df["Solution Iteration 1"] = df["Iteration 1"]

    df["Solution Iteration 2"] = df["Iteration 2"]

    df["improvement_1_to_2"] = (df["Solution Iteration 1"] == False) & (df["Solution Iteration 2"] == True)

    # Erstelle eine Maske für True-Werte in Iteration 1, Iteration 2 und Iteration 3
    mask_iter3 = (df["Iteration 1"] == True) | (df["Iteration 2"] == True) | (df["Iteration 3"] == True)
    df["Solution Iteration 3"] = mask_iter3

    # DF Spalte Task filtern, sodass nur Daten enthalten sind, die mit Task_ beginnen enthalten sind.
    df = df[df["Task"].str.startswith("Task")]

    return df


def perform_cochrans_q_test_by_model(df, run_name):
    """
    Führt pro Modell (innerhalb eines Runs) einen Cochran's Q-Test durch.
    Voraussetzungen:
    - Spalte 'Task' existiert oder du ersetzt sie durch einen eindeutigen Identifier.
    - 'temperature' enthält t=0, t=0.5, t=1.
    - 'solution' ist 0/1 (Fail/Success).
    """
    results = []
    run_data = df[df["run"] == run_name]
    models = run_data["Modellname"].unique()

    for model in models:
        model_data = run_data[run_data["Modellname"] == model]

        # Pivot: index=Task (Aufgabe), columns=temperature, values=solution
        pivot = (
            model_data
            .pivot_table(index="Task", columns="temperature", values="solution", aggfunc="first")
            .astype(float)  # aus object -> float
            .fillna(0)  # fehlende Werte füllen
            .astype(int)  # zum Schluss in int konvertieren
        )

        # Prüfen, ob alle 3 Temperaturen vorhanden sind
        if pivot.shape[1] == 3:
            result = cochrans_q(pivot.values)
            interpretation = "significant" if result.pvalue < 0.05 else "not significant"

            results.append({
                "run": run_name,
                "model": model,
                "Q-statistic": result.statistic,
                "p-value": result.pvalue,
                "interpretation": interpretation
            })
        else:
            # Falls nicht alle drei Temperaturstufen vorhanden sind
            results.append({
                "run": run_name,
                "model": model,
                "Q-statistic": None,
                "p-value": None,
                "interpretation": "not tested (missing temperature data)"
            })

    return results


def calculate_std_per_model_and_temp(df):
    """
    Berechnet die Run-zu-Run-Standardabweichung der Erfolgsraten
    für jedes Modell und jede Temperatur.

    Schritte:
    1) Gruppieren nach (Modell, Temperatur, run),
       dort die Erfolgsrate = mean(solution) bestimmen.
    2) Diese Erfolgsraten über alle Runs hinweg nach (Modell, Temperatur)
       erneut gruppieren und die Standardabweichung berechnen.
    """
    # Schritt 1: Erfolgsrate pro (Modell, Temperatur, Run)
    grouped_runs = (
        df.groupby(["Modellname", "temperature", "run"])["solution"]
          .mean()  # => Erfolgsrate pro Run
          .reset_index(name="success_rate")
    )

    # Schritt 2: Standardabweichung dieser Erfolgsraten über die Runs (Stichproben-Std, ddof=1)
    std_df = (
        grouped_runs.groupby(["Modellname", "temperature"])["success_rate"]
                   .std(ddof=1)
                   .reset_index(name="std_success_rate")
    )

    return std_df


def perform_cochrans_q_test_by_prompt(df):
    """
    Führt für jedes Modell einen Cochran's Q-Test durch, um den Einfluss des Prompts zu evaluieren.
    Es werden nur die Daten aus "Prompt A run 1" und "Prompt B run 1" mit der Temperatureinstellung t=0 verwendet.

    Voraussetzungen:
    - Spalte 'Task' existiert oder du ersetzt sie durch einen eindeutigen Identifier.
    - 'run' enthält 'Prompt A run 1' und 'Prompt B run 1'.
    - 'solution' ist 0/1 (Fail/Success).
    """
    results = []
    # Filtere die relevanten Daten: nur t=0 und die beiden Prompt-Runs
    filtered_data = df[(df["temperature"] == 0) & (df["run"].isin(["Prompt A run 1", "Prompt B run 1"]))]
    models = filtered_data["Modellname"].unique()

    for model in models:
        model_data = filtered_data[filtered_data["Modellname"] == model]

        # Pivot: index=Task (Aufgabe), columns=run (Prompt), values=solution
        pivot = (
            model_data.pivot_table(index="Task", columns="run", values="solution", aggfunc="first")
            .astype(float)  # aus object -> float
            .fillna(0)  # fehlende Werte füllen
            .astype(int)  # zum Schluss in int konvertieren
        )

        # Prüfen, ob beide Prompts vorhanden sind
        if pivot.shape[1] == 2:
            result = cochrans_q(pivot.values)
            interpretation = "significant" if result.pvalue < 0.05 else "not significant"

            results.append({
                "model": model,
                "Q-statistic": result.statistic,
                "p-value": result.pvalue,
                "interpretation": interpretation
            })
        else:
            # Falls Daten für einen Prompt fehlen
            results.append({
                "model": model,
                "Q-statistic": None,
                "p-value": None,
                "interpretation": "not tested (missing prompt data)"
            })

    return results


def apply_multiple_test_correction(results, method='fdr_by'):
    """
    Wendet eine Korrektur für multiples Testen auf die p-Werte in den Ergebnissen an.

    Args:
      results: Eine Liste von Dictionaries, die die Ergebnisse der Tests enthalten.
      method: Die Methode für die Korrektur (z.B. 'fdr_by' für Benjamini-Yekutieli).

    Returns:
      Eine Liste von Dictionaries mit den korrigierten p-Werten und Interpretationen.
    """
    # 1) Sammle p-Werte und ihre Indizes
    p_vals_with_index = [(i, r["p-value"]) for i, r in enumerate(results) if r["p-value"] is not None]

    # 2) Sortiere nach den p-Werten (optional, falls erforderlich)
    p_vals_with_index.sort(key=lambda x: x[1])
    indices, p_vals = zip(*p_vals_with_index)

    # 3) Korrektur durchführen
    reject, corrected_p_vals, _, _ = multi.multipletests(p_vals, method=method)

    # 4) Ordne die korrigierten p-Werte korrekt zu
    for idx, corrected_p in zip(indices, corrected_p_vals):
        results[idx]["corrected_p-value"] = corrected_p
        results[idx]["corrected_interpretation"] = (
            "significant" if corrected_p < 0.05 else "not significant"
        )

    # 5) Für Ergebnisse ohne p-Wert (None) Standardwert beibehalten
    for r in results:
        if "corrected_p-value" not in r:
            r["corrected_p-value"] = None
            r["corrected_interpretation"] = r["interpretation"]

    return results


def perform_mcnemar_test_1_vs_2_only_failures(df, run_name):
    """
    Führt den McNemar-Test für den Vergleich von Iteration 1 und Iteration 2 durch,
    wobei nur Fälle betrachtet werden, in denen Iteration 1 fehlgeschlagen ist.

    Args:
        df: Der Pandas DataFrame mit den Daten.
        run_name: Der Name des Laufs, der analysiert werden soll.

    Returns:
        Eine Liste von Dictionaries, die die Ergebnisse für jedes Modell enthalten.
    """
    results = []
    run_data = df[(df["temperature"] == 0) & (df["run"] == run_name)]

    if run_data.empty:
        print(f"Keine Daten für run '{run_name}' gefunden.")
        return []

    models = run_data["Modellname"].unique()

    for model in models:
        model_data = run_data[run_data["Modellname"] == model]

        if model_data.empty:
            results.append({
                "run": run_name,
                "model": model,
                "Iteration Comparison": "Iteration 2 vs. Iteration 1 (nur Misserfolge in Iteration 1)",
                "statistic": None,
                "p-value": None,
                "interpretation": "Keine Daten für dieses Modell in diesem Lauf",
                "contingency_table": None
            })
            continue

        failures_in_iteration_1 = model_data[model_data["Solution Iteration 1"] == False].copy()

        # Behandeln von fehlenden Werten in "Solution Iteration 2"
        missing_iteration_2 = failures_in_iteration_1["Solution Iteration 2"].isna()
        if missing_iteration_2.any():
            print(
                f"Warnung: Für Modell '{model}' und Lauf '{run_name}' fehlen für {missing_iteration_2.sum()} Fälle, in denen Iteration 1 fehlschlug, die Werte für Iteration 2. Diese Fälle werden ignoriert.")
            failures_in_iteration_1 = failures_in_iteration_1.dropna(subset=["Solution Iteration 2"])

        if failures_in_iteration_1.empty:
            results.append({
                "run": run_name,
                "model": model,
                "Iteration Comparison": "Iteration 2 vs. Iteration 1 (nur Misserfolge in Iteration 1)",
                "statistic": None,
                "p-value": None,
                "interpretation": "Keine Fehler in Iteration 1 für dieses Modell oder fehlende Werte in Iteration 2 nach Fehler in Iteration 1",
                "contingency_table": None
            })
            continue

        failures_in_iteration_1["success_in_iteration_2"] = failures_in_iteration_1["Solution Iteration 2"] == True

        contingency_table = pd.crosstab(failures_in_iteration_1["Solution Iteration 1"],
                                        failures_in_iteration_1["success_in_iteration_2"])
        contingency_table.columns = ["Iteration 2 False", "Iteration 2 True"]
        contingency_table.index = ["Iteration 1 False"]

        if contingency_table.empty or (contingency_table["Iteration 2 True"] == 0).all() and (
                contingency_table["Iteration 2 False"] == 0).all():
            results.append({
                "run": run_name,
                "model": model,
                "Iteration Comparison": "Iteration 2 vs. Iteration 1 (nur Misserfolge in Iteration 1)",
                "statistic": None,
                "p-value": None,
                "interpretation": "Keine Verbesserung von Iteration 1 zu 2 gefunden, leere Tabelle oder keine Varianz in den Daten",
                "contingency_table": contingency_table.to_dict() if not contingency_table.empty else None
            })
            continue

        result = mcnemar(contingency_table, correction=True)
        interpretation = "signifikant" if result.pvalue < 0.05 else "nicht signifikant"

        results.append({
            "run": run_name,
            "model": model,
            "Iteration Comparison": "Iteration 2 vs. Iteration 1 (nur Misserfolge in Iteration 1)",
            "statistic": result.statistic,
            "p-value": result.pvalue,
            "interpretation": f"Die Verbesserung von Iteration 1 zu 2 ist {interpretation} (p={result.pvalue:.4f})",
            "contingency_table": contingency_table.to_dict()
        })

    return results
def evaluate_error_significance_cochrans_q_combinations(df):
    """
    Bewertet die Signifikanz eines Leistungsunterschieds zwischen Fehlerkategorien-Kombinationen
    für Iteration 1, t=0 je Modell mit Cochran's Q-Test und p-Wert-Korrektur.

    Args:
      df: Das DataFrame mit den Daten (bereits gefiltert für t=0 und Iteration 1).
    """
    results = []

    # Modelle und Fehlertyp-Kombinationen definieren
    models = df["Modellname"].unique()
    error_combinations = [
        ("syntaxerror", "logicerror"),
        ("syntaxerror", "runtimeerror")
        #("logicerror", "runtimeerror")
    ]

    for model in models:
        # Filtere Daten für ein bestimmtes Modell und relevante Bedingungen
        model_data = df[(df["Modellname"] == model) & (df["temperature"] == 0) & (df["run"] == "Prompt A run 1")]

        for error1, error2 in error_combinations:
            # Filtere Daten für die zwei Fehlertypen
            filtered_data = model_data[model_data["error"].isin([error1, error2])]

            # Pivot-Tabelle: Task_error_combination als Index, error als Spalten
            pivot = (
                filtered_data.pivot_table(
                    index="Task_error_combination",  # Task-ID ohne Fehlertyp
                    columns="error",                # Fehlertypen (syntaxerror, logicerror, etc.)
                    values="Iteration 1",           # Erfolgswerte (1/0)
                    aggfunc="first"
                )
                .astype(float)
                .fillna(0)  # Fehlende Werte mit 0 füllen (kein Erfolg)
                .astype(int)
            )

            # Cochran's Q-Test durchführen, wenn beide Fehlerkategorien vorhanden sind
            if pivot.shape[1] == 2:
                result = cochrans_q(pivot.values)
                interpretation = "significant" if result.pvalue < 0.05 else "not significant"

                results.append({
                    "model": model,
                    "error_combination": f"{error1} vs {error2}",
                    "Q-statistic": result.statistic,
                    "p-value": result.pvalue,
                    "interpretation": interpretation
                })
            else:
                results.append({
                    "model": model,
                    "error_combination": f"{error1} vs {error2}",
                    "Q-statistic": None,
                    "p-value": None,
                    "interpretation": "not tested (missing error data)"
                })

    # p-Werte korrigieren (z. B. Holm-Methode)
    results = apply_multiple_test_correction(results, method="holm")

    return results





if __name__ == "__main__":
    # Pfade zu den Verzeichnissen mit den CSV-Daten
    paths = [
        r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt A run 1",
        r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt A run 2",
        r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt A run 3",
        r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt B run 1"
    ]

    # Daten laden und vorbereiten
    prepared_data = load_and_prepare_data(paths)

    # Cochran's Q-Test je Modell und je Run
    all_results = []
    for run_name in prepared_data["run"].unique():
        run_results = perform_cochrans_q_test_by_model(prepared_data, run_name)
        all_results.extend(apply_multiple_test_correction(run_results, method='holm'))

    print("=== Cochran's Q-Test by Model & Run (Holm-korrigiert) ===")
    is_any_significant = False
    for res in all_results:
        print(res)
        if res["corrected_interpretation"] == "significant":
            is_any_significant = True

    if not is_any_significant:
        print("\nKeines der getesteten Modelle ist signifikant.")

    pd.set_option('display.max_columns', None)

    # Standardabweichung zwischen den Runs
    std_df = calculate_std_per_model_and_temp(prepared_data)
    print("\n=== Standardabweichungen (Run-zu-Run) je Modell + Temperatur ===")
    print(std_df)

    # Cochran's Q-Test je Modell für den Prompt-Vergleich
    prompt_results = perform_cochrans_q_test_by_prompt(prepared_data)
    prompt_results = apply_multiple_test_correction(prompt_results, method='holm')

    print("\n=== Cochran's Q-Test by Prompt (Holm-korrigiert) ===")
    for res in prompt_results:
        print(res)



    # 5. Cochran's Q-Test für Fehlerkategorien (Iteration 1, t=0)
    error_significance_results = evaluate_error_significance_cochrans_q_combinations(prepared_data)

    print("\n=== Cochran's Q-Test für Fehlerkategorien (Iteration 1, t=0) ===")
    for res in error_significance_results:
        print(res)

    # Cochran's Q-Test je Modell und Fehlertyp für den Iterationsvergleich (NEU)
    all_iteration_error_results = []
    iteration_error_results = perform_mcnemar_test_1_vs_2_only_failures(prepared_data, "Prompt A run 1")
    all_iteration_error_results.extend(
        apply_multiple_test_correction(iteration_error_results, method='holm'))  # Holm Korrektur hinzugefügt

    # 6. Cochran's Q-Test für Fehlerkategorien der Iterationen 1 und 2
    print("\n Mc Nemar Test für Iteration 1 und 2")
    for res in all_iteration_error_results:
        print(res)


