import pandas as pd
import scipy.stats as stats
import itertools
import numpy as np
import os
from statsmodels.stats.multitest import multipletests


def load_and_prepare_data(path):
    # Lädt Daten aus Excel-Dateien, fügt eine Temperaturspalte hinzu und kombiniert die DataFrames.
    # Erstellt die benötigten Spalten 'Modellname', 'blackbox' und 'solution'.
    df_05 = pd.read_excel(os.path.join(path, "Zusammengeführt t=0.5.xlsx"))
    df_1 = pd.read_excel(os.path.join(path, "Zusammengeführt t=1.xlsx"))
    df_0 = pd.read_excel(os.path.join(path, "Zusammengeführt t=0.xlsx"))

    df_05["temperature"] = 0.5
    df_1["temperature"] = 1
    df_0["temperature"] = 0

    df = pd.concat([df_05, df_1, df_0])

    df["Modellname"] = df["Modell"].str.replace("log_", "").str.replace(".csv", "")
    df["blackbox"] = df["Modellname"].apply(lambda x: False if x in ["deepseek_deepseek-chat", "qwen_qwen-2.5-coder-32b-instruct"] else True)
    df["solution"] = df["Iteration 1"]  # Nur Iteration 1 berücksichtigen
    df.to_csv("Zusammengeführt.csv", index=False)
    return df


def create_contingency_table(df1, df2):
    # Erstellt eine 2x2-Kontingenztabelle für den Vergleich der Erfolgsraten zweier Modelle.

    a = df1["solution"].sum()
    c = len(df1) - a
    b = df2["solution"].sum()
    d = len(df2) - b

    return [[a, b], [c, d]]


def perform_chi2_test(contingency_table):
    # Führt einen Chi-Quadrat-Test oder Fisher's Exact Test auf Basis der Kontingenztabelle durch.
    # Gibt das Testergebnis und die erwarteten Häufigkeiten (falls zutreffend) zurück.

    expected_counts_ok = all(stats.chi2_contingency(contingency_table, correction=False).expected_freq.flatten() >= 5)
    if expected_counts_ok:
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        return (chi2_stat, p_val, dof), expected
    else:
        oddsratio, p_val = stats.fisher_exact(contingency_table)
        return (oddsratio, p_val), None


def interpret_chi2_results(test_result, model1, model2, alpha=0.05):
    # Interpretiert die Ergebnisse des Chi-Quadrat-Tests oder Fisher's Exact Tests.
    # Gibt das Ergebnis des Vergleichs aus und gibt den p-Wert zurück.

    statistic, p_val = test_result[:2]
    if len(test_result) > 2:
        dof = test_result[2]
    else:
        dof = "Fisher's Exact Test"

    if isinstance(dof, int):
        print(f"  Chi-Quadrat-Test: Statistik={statistic:.4f}, p-Wert={p_val:.4f}, Freiheitsgrade={dof}")
    else:
        print(f"  {dof}: Odds Ratio={statistic:.4f}, p-Wert={p_val:.4f}")

    if p_val <= alpha:
        print(f"  Ergebnis: Die Erfolgsrate von Modell {model1} unterscheidet sich in der ersten Iteration statistisch signifikant von der Erfolgsrate von Modell {model2} (p={p_val:.4f}).")
    else:
        print(f"  Ergebnis: Die Erfolgsrate von Modell {model1} unterscheidet sich in der ersten Iteration nicht statistisch signifikant von der Erfolgsrate von Modell {model2} (p={p_val:.4f}).")

    return p_val


def calculate_success_rates(df1, df2):
    # Berechnet und gibt die Erfolgsraten für zwei Modelle in Iteration 1 aus.

    success_rate_1 = df1["Iteration 1"].mean() * 100
    success_rate_2 = df2["Iteration 1"].mean() * 100
    print(f"  Erfolgsrate Modell 1: {success_rate_1:.2f}%")
    print(f"  Erfolgsrate Modell 2: {success_rate_2:.2f}%")


def multiple_model_comparisons(df, alpha=0.05, method='fdr_bh'):
    # Führt paarweise Vergleiche aller Modelle bei Temperatur 0 durch.
    # Korrigiert die p-Werte für multiples Testen und gibt die Ergebnisse aus.

    df_temp0 = df[df["temperature"] == 0]  # Nur Daten für temperature=0 verwenden
    models = df_temp0["Modellname"].unique()
    num_models = len(models)
    p_value_matrix = pd.DataFrame(index=models, columns=models)
    significant_diffs = pd.DataFrame(index=models, columns=models, dtype=bool)
    p_value_matrix_adj = pd.DataFrame(index=models, columns=models)  # DataFrame für adjustierte p-Werte

    # Initialisiere die Diagonale mit False
    for i in range(num_models):
        significant_diffs.iloc[i, i] = False

    # Führe paarweise Vergleiche durch
    for model1, model2 in itertools.combinations(models, 2):
        df_model1 = df_temp0[df_temp0["Modellname"] == model1]
        df_model2 = df_temp0[df_temp0["Modellname"] == model2]

        contingency_table = create_contingency_table(df_model1, df_model2)
        test_result, expected = perform_chi2_test(contingency_table)

        print(f"\n--- Vergleich: {model1} vs. {model2} ---")
        if expected is not None:
            print(f"Erwartete Häufigkeiten:\n{expected}")

        p_val = interpret_chi2_results(test_result, model1, model2, alpha)
        calculate_success_rates(df_model1, df_model2)

        p_value_matrix.loc[model1, model2] = p_val
        p_value_matrix.loc[model2, model1] = p_val  # Symmetrie

        significant_diffs.loc[model1, model2] = p_val <= alpha
        significant_diffs.loc[model2, model1] = p_val <= alpha  # Symmetrie

    # Korrektur für multiples Testen (z.B. mit statsmodels)
    p_values_flat = p_value_matrix.values[np.triu_indices(num_models, k=1)]
    reject, pvals_corrected, _, _ = multipletests(p_values_flat, alpha=alpha, method=method)

    # Update die significant_diffs DataFrame mit den korrigierten p-Werten
    k = 0
    for i in range(num_models):
        for j in range(i + 1, num_models):
            significant_diffs.iloc[i, j] = reject[k]
            significant_diffs.iloc[j, i] = reject[k]
            p_value_matrix_adj.iloc[i, j] = pvals_corrected[k]  # Speichere adjustierte p-Werte
            p_value_matrix_adj.iloc[j, i] = pvals_corrected[k]  # Speichere adjustierte p-Werte
            k += 1

    print("\np-Werte (unadjustiert):\n", p_value_matrix)
    print("\np-Werte (adjustiert):\n", p_value_matrix_adj)  # Ausgabe der adjustierten p-Werte
    print("\nSignifikante Unterschiede (nach", method, "):\n", significant_diffs)  # Ausgabe angepasst


def global_chi2_test(df, alpha=0.05):
    # Führt einen globalen Chi-Quadrat-Test durch, um zu prüfen, ob es einen Zusammenhang zwischen Modellwahl und Erfolgsrate gibt.

    contingency_table = pd.crosstab(df["Modellname"], df["solution"])
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"\n--- Globaler Chi-Quadrat-Test ---")
    print(f"Chi-Quadrat-Statistik: {chi2_stat:.4f}, p-Wert: {p_val:.4f}, Freiheitsgrade: {dof}")
    print(f"Erwartete Häufigkeiten:\n{expected}")

    if p_val <= alpha:
        print(f"Ergebnis: Es gibt einen signifikanten Zusammenhang zwischen der Modellwahl und der Erfolgsrate (p={p_val:.4f}).")
    else:
        print(f"Ergebnis: Es gibt keinen signifikanten Zusammenhang zwischen der Modellwahl und der Erfolgsrate (p={p_val:.4f}).")

    return p_val


# Hauptprogramm
if __name__ == "__main__":
    path = r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt A"
    pd.set_option('display.max_columns', None)
    # Daten laden und vorbereiten
    df = load_and_prepare_data(path)

    # Globalen Test durchführen
    global_p_val = global_chi2_test(df, alpha=0.05)

    # Paarweise Vergleiche durchführen, wenn der globale Test signifikant ist
    if global_p_val <= 0.05:
        multiple_model_comparisons(df, alpha=0.05)
    else:
        print("\nDer globale Test zeigt keine signifikanten Unterschiede zwischen den Modellen. Paarweise Vergleiche werden nicht durchgeführt.")