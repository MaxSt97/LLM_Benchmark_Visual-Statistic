import os
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
import statsmodels.stats.multitest as multi
import numpy as np
from scipy.stats import binom


def load_and_prepare_data(paths):
    """
    Lädt die CSV-Dateien aus den angegebenen Verzeichnissen und erstellt ein gemeinsames DataFrame.
    """
    df_list = []

    for path in paths:
        run_name = os.path.basename(path)
        for temp_folder in ["t=0", "t=0.5", "t=1"]:
            temp_path = os.path.join(path, temp_folder)
            if not os.path.exists(temp_path):
                continue
            for filename in os.listdir(temp_path):
                if filename.endswith(".csv"):
                    filepath = os.path.join(temp_path, filename)
                    df = pd.read_csv(filepath, nrows=61)

                    df["temperature"] = float(temp_folder.split("=")[1])
                    df["Modellname"] = filename.replace("log_", "").replace(".csv", "")
                    df["run"] = run_name
                    df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    # Spalte (Fehler) hinzufügen
    df["error"] = df["Task"].apply(lambda x: x.rsplit("_", 1)[-1])

    # Spalte Task hinzufügen ohne error prefix
    df["Task_error_combination"] = df["Task"].apply(lambda x: x.rsplit("_", 1)[0])

    # Nur Iteration 1 als Indikator für Erfolg
    df["solution"] = df["Iteration 1"]

    # DF Spalte Task filtern, sodass nur Daten enthalten sind, die mit Task_ beginnen
    df = df[df["Task"].str.startswith("Task")]
    df.to_csv("prepared_data.csv", index=False)
    return df

def perform_cochrans_q_test_by_model(df, run_name):
    """
    Führt pro Modell (innerhalb eines Runs) einen Cochran's Q-Test durch (für Temperaturvergleich).
    """
    results = []
    run_data = df[df["run"] == run_name]
    models = run_data["Modellname"].unique()

    for model in models:
        model_data = run_data[run_data["Modellname"] == model]

        pivot = (
            model_data
            .pivot_table(index="Task", columns="temperature", values="solution", aggfunc="first")
            .astype(float)
            .fillna(0)
            .astype(int)
        )

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
            results.append({
                "run": run_name,
                "model": model,
                "Q-statistic": None,
                "p-value": None,
                "interpretation": "not tested (missing temperature data)"
            })

    return results


def perform_mcnemar_test_by_prompt(df, run_name_I, run_name_II):
    """
    Führt einen McNemar-Test durch, wenn die Daten binär (True/False) sind.
    Stellt sicher, dass die Gruppierung nach Modellname und Aufgabe korrekt erfolgt.
    """
    results = []

    # Filter für die relevanten Runs
    filtered_data = df[(df["run"].isin([run_name_I, run_name_II]))]
    models = filtered_data["Modellname"].unique()

    for model in models:
        # Daten für das Modell filtern
        model_data = filtered_data[filtered_data["Modellname"] == model]
        pivot = model_data.pivot_table(index="Task", columns="run", values="Iteration 1", aggfunc="first")

        if pivot.shape[1] == 2:
            pivot = pivot[[run_name_I, run_name_II]]  # Sicherstellen, dass nur die relevanten Runs verwendet werden

            # Initialisiere die Kontingenztabelle
            contingency_table = np.zeros((2, 2), dtype=int)

            # Durchlaufe jede Zeile und fülle die Kontingenztabelle
            for index, row in pivot.iterrows():
                if pd.notna(row[run_name_I]) and pd.notna(row[run_name_II]):
                    a = row[run_name_I]
                    b = row[run_name_II]

                    # Werte in Kontingenztabelle einsortieren
                    if a == 1 and b == 1:
                        contingency_table[1, 1] += 1
                    elif a == 1 and b == 0:
                        contingency_table[1, 0] += 1
                    elif a == 0 and b == 1:
                        contingency_table[0, 1] += 1
                    elif a == 0 and b == 0:
                        contingency_table[0, 0] += 1

            # Kontingenztabelle prüfen
            print(f"Model: {model}\nContingency Table:\n{contingency_table}\n")

            # McNemar-Test durchführen, falls Tabelle gültig
            try:
                result = mcnemar(contingency_table, correction=True, exact=True)
                interpretation = "significant" if result.pvalue < 0.05 else "not significant"
                statistic = result.statistic
                pvalue = result.pvalue
            except ValueError as e:  # Falls die Kontingenztabelle nicht gültig ist
                statistic = None
                pvalue = None
                interpretation = f"Error during McNemar Test: {e}"

            # Ergebnisse speichern
            results.append({
                "model": model,
                "contingency_table": contingency_table.tolist(),
                "statistic": statistic,
                "p-value": pvalue,
                "interpretation": interpretation
            })

        else:
            results.append({
                "model": model,
                "contingency_table": None,
                "statistic": None,
                "p-value": None,
                "interpretation": "not tested (missing prompt data)"
            })

    return results

def calculate_improvement_rate_and_confidence_interval(df, run_name, iteration_x, iteration_y):
    """
    Berechnet die Verbesserungsrate und das 95%-Konfidenzintervall für den Vergleich zwischen zwei Iterationen.

    :param df: DataFrame mit den Daten.
    :param run_name: Name des Runs (z.B., "Prompt A run 1").
    :param iteration_x: Name der vorherigen Iteration (z.B., "Iteration 1").
    :param iteration_y: Name der nachfolgenden Iteration (z.B., "Iteration 2").
    :return: DataFrame mit Verbesserungsrate und Konfidenzintervall für jedes Modell.
    """
    results = []

    # Filtern der Daten für den spezifischen Run und t=0
    run_data = df[(df["run"] == run_name) & (df["temperature"] == 0)]

    for model in run_data["Modellname"].unique():
        model_data = run_data[run_data["Modellname"] == model]

        # Anzahl der Fehler in Iteration X
        failures_x = len(model_data[model_data[iteration_x] == False])

        # Anzahl der Erfolge in Iteration Y, die in Iteration X Fehler waren
        successes_y = len(model_data[(model_data[iteration_x] == False) & (model_data[iteration_y] == True)])

        # Berechnen der Verbesserungsrate
        improvement_rate = successes_y / failures_x if failures_x > 0 else 0

        # Berechnen des 95%-Konfidenzintervalls
        if failures_x > 0:
            ci_low, ci_high = binom.interval(0.95, failures_x, improvement_rate)
            ci_low /= failures_x
            ci_high /= failures_x
        else:
            ci_low, ci_high = None, None

        results.append({
            "model": model,
            "run": run_name,
            "iteration_comparison": f"{iteration_y} vs {iteration_x}",
            "improvement_rate": improvement_rate,
            "CI_lower": ci_low,
            "CI_upper": ci_high
        })

    return pd.DataFrame(results)

def calculate_success_rate_by_error_type(df, run_name):
    """
    Berechnet die Erfolgsrate und das 95%-Konfidenzintervall für jeden Fehlertyp in Iteration 1 (bei t=0).

    :param df: DataFrame mit den Daten.
    :param run_name: Name des Runs, der analysiert werden soll (z.B., "Prompt A run 1").
    :return: DataFrame mit Erfolgsrate und Konfidenzintervall für jedes Modell und jeden Fehlertyp.
    """
    results = []

    # Filtern der Daten für den spezifischen Run und t=0
    filtered_data = df[(df["run"] == run_name) & (df["temperature"] == 0)]

    for model in filtered_data["Modellname"].unique():
        model_data = filtered_data[filtered_data["Modellname"] == model]

        for error_type in ["syntaxerror", "logicerror", "runtimeerror"]:
            # Daten für den aktuellen Fehlertyp filtern
            error_type_data = model_data[model_data["error"] == error_type]

            # Anzahl der Erfolge und Versuche für den aktuellen Fehlertyp
            successes = len(error_type_data[error_type_data["Iteration 1"] == True])
            total_attempts = len(error_type_data)

            # Erfolgsrate berechnen
            success_rate = successes / total_attempts if total_attempts > 0 else 0

            # 95%-Konfidenzintervall berechnen
            if total_attempts > 0:
                ci_low, ci_high = binom.interval(0.95, total_attempts, success_rate)
                ci_low /= total_attempts
                ci_high /= total_attempts
            else:
                ci_low, ci_high = None, None

            results.append({
                "model": model,
                "error_type": error_type,
                "success_rate": success_rate,
                "CI_lower": ci_low,
                "CI_upper": ci_high
            })

    return pd.DataFrame(results)

def perform_mcnemar_test_by_error_type(df, run_name):
    """
    Führt für jedes Modell einen McNemar-Test durch, um den Einfluss des Fehlertyps zu evaluieren.
    Es werden jeweils zwei Fehlertypen miteinander verglichen.
    Es werden nur die Daten aus Iteration 1, t=0 und dem angegebenen Run verwendet.
    """
    results = []
    # Filtere nach t=0 und dem spezifischen Run
    filtered_data = df[(df["temperature"] == 0) & (df["run"] == run_name)]
    models = filtered_data["Modellname"].unique()
    error_combinations = [
        ("syntaxerror", "logicerror"),
        ("syntaxerror", "runtimeerror")
        # ("logicerror", "runtimeerror")  # Kann bei Bedarf hinzugefügt werden
    ]

    for model in models:
        model_data = filtered_data[filtered_data["Modellname"] == model]

        for error1, error2 in error_combinations:
            # Filtere Daten für die zwei Fehlertypen
            filtered_model_data = model_data[model_data["error"].isin([error1, error2])]

            # Bereinigte Task-ID erstellen (ohne Fehlertyp-Suffix)
            filtered_model_data["Task_ID"] = filtered_model_data["Task"].str.replace(f"_{error1}$", "", regex=True).str.replace(f"_{error2}$", "", regex=True)

            # Pivot-Tabelle: index=Task_ID, columns=error, values="Iteration 1"
            pivot = filtered_model_data.pivot_table(index="Task_ID", columns="error", values="Iteration 1", aggfunc="first")

            # Überprüfen, ob beide Fehlertypen in den Spalten vorhanden sind
            if error1 in pivot.columns and error2 in pivot.columns:
                # Entfernen von Zeilen, die in beiden Spalten NaN-Werte haben
                pivot = pivot.dropna(subset=[error1, error2], how='all')

                # Initialisiere die Kontingenztabelle
                contingency_table = np.zeros((2, 2), dtype=int)

                # Durchlaufe jede Zeile und fülle die Kontingenztabelle
                for index, row in pivot.iterrows():
                    a = int(row[error1]) if pd.notna(row[error1]) else 0
                    b = int(row[error2]) if pd.notna(row[error2]) else 0

                    # Werte in Kontingenztabelle einsortieren
                    contingency_table[a, b] += 1

                # McNemar-Test durchführen
                try:
                    result = mcnemar(contingency_table, correction=True)
                    interpretation = "significant" if result.pvalue < 0.05 else "not significant"
                    statistic = result.statistic
                    pvalue = result.pvalue
                except ValueError as e:  # Falls die Kontingenztabelle nicht gültig ist
                    statistic = None
                    pvalue = None
                    interpretation = f"Error during McNemar Test: {e}"

                results.append({
                    "model": model,
                    "error_comparison": f"{error1} vs {error2}",
                    "contingency_table": contingency_table.tolist(),
                    "statistic": statistic,
                    "p-value": pvalue,
                    "interpretation": interpretation
                })
            else:
                results.append({
                    "model": model,
                    "error_comparison": f"{error1} vs {error2}",
                    "statistic": None,
                    "p-value": None,
                    "interpretation": "not tested (missing error type data)"
                })


    return results

def apply_multiple_test_correction(results, method='fdr_by'):
    """
    Wendet eine Korrektur für multiples Testen auf die p-Werte in den Ergebnissen an.
    """
    p_vals_with_index = [(i, r["p-value"]) for i, r in enumerate(results) if r["p-value"] is not None]
    p_vals_with_index.sort(key=lambda x: x[1])
    indices, p_vals = zip(*p_vals_with_index)

    reject, corrected_p_vals, _, _ = multi.multipletests(p_vals, method=method)

    for idx, corrected_p in zip(indices, corrected_p_vals):
        results[idx]["corrected_p-value"] = corrected_p
        results[idx]["corrected_interpretation"] = (
            "significant" if corrected_p < 0.05 else "not significant"
        )

    for r in results:
        if "corrected_p-value" not in r:
            r["corrected_p-value"] = None
            r["corrected_interpretation"] = r["interpretation"]

    return results

if __name__ == "__main__":
    # Pfade definieren
    paths = [
        r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt A run 1",
        r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt A run 2",
        r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt A run 3",
        r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt B run 1"
    ]

    prepared_data = load_and_prepare_data(paths)

    # === Cochran's Q-Test ===
    all_results = []
    for run_name in prepared_data["run"].unique():
        run_results = perform_cochrans_q_test_by_model(prepared_data, run_name)
        all_results.extend(apply_multiple_test_correction(run_results, method='holm'))

    print("=== Cochran's Q-Test by Model & Run (Holm-korrigiert) ===")
    for res in all_results:
        print(res)

    # === McNemar-Test für Prompt-Vergleich ===
    prompt_results = perform_mcnemar_test_by_prompt(prepared_data, "Prompt A run 1", "Prompt B run 1")
    prompt_results = apply_multiple_test_correction(prompt_results, method='holm')

    print("\n=== McNemar-Test by Prompt (Holm-korrigiert) ===")
    for res in prompt_results:
        print(res)

    # === Verbesserungsrate (Iteration 2 vs. Iteration 1) für Prompt A run 1 ===
    improvement_results_2_1 = calculate_improvement_rate_and_confidence_interval(prepared_data, "Prompt A run 1", "Iteration 1", "Iteration 2")
    pd.set_option('display.max_columns', None)  # Nur einmal setzen
    print("\n=== Verbesserungsrate und Konfidenzintervall (Iteration 2 vs. Iteration 1) ===")
    print(improvement_results_2_1)

    # === Stabilität der Verbesserungsraten (Iteration 2 vs. Iteration 1) ===
    improvement_results = {}
    for run in ["Prompt A run 1", "Prompt A run 2", "Prompt A run 3"]:
        improvement_results[run] = calculate_improvement_rate_and_confidence_interval(prepared_data, run, "Iteration 1", "Iteration 2")

    improvement_results_all_runs = pd.concat(improvement_results.values())
    pivot_table = improvement_results_all_runs.pivot_table(index="model", columns="run", values="improvement_rate")

    std_dev = pivot_table.std(axis=1)
    mean_improvement_rate = pivot_table.mean(axis=1)
    variation_coefficient = std_dev / mean_improvement_rate

    stability_df = pd.DataFrame({
        "std_dev": std_dev,
        "variation_coefficient": variation_coefficient,
        **pivot_table.to_dict() # Fügt die Spalten der Pivot-Tabelle direkt hinzu
    }).sort_values(by="std_dev")

    print("\n=== Stabilität der Verbesserungsraten (Iteration 2 vs. Iteration 1) ===")
    print(stability_df)
    stability_df.to_csv("stability_results.csv")

    # === McNemar-Test für Fehlertypen ===
    error_results = perform_mcnemar_test_by_error_type(prepared_data, "Prompt A run 1")
    error_results = apply_multiple_test_correction(error_results, method='holm')

    print("\n=== McNemar-Test by Error Type (Holm-korrigiert) ===")
    for res in error_results:
        print(res)

    # === Erfolgsrate nach Fehlertyp ===
    success_rate_results = calculate_success_rate_by_error_type(prepared_data, "Prompt A run 1")
    print("\n=== Success Rate by Error Type ===")
    print(success_rate_results)



