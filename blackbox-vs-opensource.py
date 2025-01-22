import pandas as pd
import os
import scipy.stats as stats


def load_and_prepare_data(path):
    # Lädt die Daten aus den Excel-Dateien im angegebenen Pfad.
    # Fügt eine Temperaturspalte hinzu und kombiniert die DataFrames.
    df_0 = pd.read_excel(os.path.join(path, "Zusammengeführt t=0.xlsx"))
    df_05 = pd.read_excel(os.path.join(path, "Zusammengeführt t=0.5.xlsx"))
    df_1 = pd.read_excel(os.path.join(path, "Zusammengeführt t=1.xlsx"))

    # Fügt eine Spalte "temperature" hinzu, die die Temperatur für jeden Datensatz angibt.
    df_0["temperature"] = 0
    df_05["temperature"] = 0.5
    df_1["temperature"] = 1

    # Kombiniert die DataFrames zu einem einzigen DataFrame.
    df = pd.concat([df_05, df_1, df_0])

    # Erstellt eine Spalte "Modellname", die den Namen des Modells ohne Präfix und Suffix enthält.
    df["Modellname"] = df["Modell"].str.replace("log_", "").str.replace(".csv", "")
    # Erstellt eine Spalte "blackbox", die angibt, ob es sich um ein Blackbox-Modell handelt (True) oder nicht (False).
    df["blackbox"] = df["Modellname"].apply(lambda x: False if x in ["deepseek_deepseek-chat", "qwen_qwen-2.5-coder-32b-instruct"] else True)
    # Erstellt eine Spalte "solution", die angibt, ob in mindestens einer Iteration eine Lösung gefunden wurde.
    df["solution"] = df["Iteration 1"] | df["Iteration 2"] | df["Iteration 3"]

    return df


def create_contingency_table(df):
    # Erstellt eine Kontingenztabelle für den Vergleich der Debugging-Leistung von Blackbox- und Open-Source-Modellen.

    # Teilt den DataFrame in Blackbox- und Open-Source-Modelle auf.
    df_blackbox = df[df["blackbox"]]
    df_opensource = df[~df["blackbox"]]

    # Berechnet die Anzahl der erfolgreichen Debugging-Vorgänge (a und b) und die Anzahl der erfolglosen Debugging-Vorgänge (c und d).
    a = df_blackbox["solution"].sum()
    b = df_opensource["solution"].sum()
    c = len(df_blackbox) - a
    d = len(df_opensource) - b

    return [[a, b], [c, d]]


def perform_chi2_test(contingency_table):
    # Führt einen Chi-Quadrat-Test oder Fisher's Exact Test durch, um die statistische Signifikanz des Zusammenhangs zu ermitteln.
    # Gibt das Testergebnis und die erwarteten Häufigkeiten (falls zutreffend) zurück.

    # Überprüft, ob die erwarteten Häufigkeiten für einen Chi-Quadrat-Test ausreichend groß sind.
    expected_counts_ok = all(stats.chi2_contingency(contingency_table, correction=False).expected_freq.flatten() >= 5)
    if expected_counts_ok:
        # Führt einen Chi-Quadrat-Test durch.
        chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        return (chi2_stat, p_val, dof), expected
    else:
        # Führt Fisher's Exact Test durch, wenn die erwarteten Häufigkeiten zu klein sind.
        oddsratio, p_val = stats.fisher_exact(contingency_table)
        return (oddsratio, p_val), None


def interpret_chi2_results(test_result, temperature, alpha=0.05):
    # Interpretiert die Ergebnisse des Chi-Quadrat-Tests oder Fisher's Exact Tests.
    # Gibt eine Erklärung aus, ob die Nullhypothese verworfen wird oder nicht.

    statistic, p_val = test_result[:2]
    if len(test_result) > 2:
        dof = test_result[2]
    else:
        dof = "Fisher's Exact Test"

    if isinstance(dof, int):
        print(f"  Chi-Quadrat-Test: Statistik={statistic:.4f}, p-Wert={p_val:.4f}, Freiheitsgrade={dof}")
    else:
        print(f"  {dof}: Odds Ratio={statistic:.4f}, p-Wert={p_val:.4f}")

    if temperature == "alle":
      print(f"  Ergebnis (über alle Temperaturen):")
    else:
      print(f"  Ergebnis (Temperatur {temperature}):")

    if p_val <= alpha:
        print(f"    Da der p-Wert ({p_val:.4f}) kleiner oder gleich dem Signifikanzniveau ({alpha}) ist, wird die Nullhypothese H0.1 verworfen.")
        print("    Es gibt einen statistisch signifikanten Zusammenhang zwischen LLM-Typ (Blackbox vs. Open-Source) und Debugging-Leistung.")
    else:
        print(f"    Da der p-Wert ({p_val:.4f}) größer als das Signifikanzniveau ({alpha}) ist, kann die Nullhypothese H0.1 nicht verworfen werden.")
        print("    Es gibt keinen statistisch signifikanten Zusammenhang zwischen LLM-Typ (Blackbox vs. Open-Source) und Debugging-Leistung.")


def calculate_success_rates(contingency_table):
    # Berechnet und gibt die Erfolgsraten für Blackbox- und Open-Source-Modelle aus.

    a, b, c, d = contingency_table[0][0], contingency_table[0][1], contingency_table[1][0], contingency_table[1][1]
    print(f"  Erfolgsrate Blackbox: {a / (a + c) * 100:.2f}%")
    print(f"  Erfolgsrate Open-Source: {b / (b + d) * 100:.2f}%")


def chi2_test_by_temperature(df, temperature):
    # Führt den Chi-Quadrat-Test für eine bestimmte Temperatur durch.
    # Gibt die Ergebnisse und die Erfolgsraten für diese Temperatur aus.

    print(f"\n--- Chi-Quadrat-Test für Temperatur = {temperature} ---")
    df_temp = df[df["temperature"] == temperature]
    contingency_table = create_contingency_table(df_temp)
    test_result, expected = perform_chi2_test(contingency_table)

    if expected is not None:
        print(f"  Erwartete Häufigkeiten:\n{expected}")

    interpret_chi2_results(test_result, temperature)
    calculate_success_rates(contingency_table)


def chi2_test_all_temperatures(df):
    # Führt den Chi-Quadrat-Test für alle Temperaturen zusammen durch.
    # Gibt die Ergebnisse und die Erfolgsraten für alle Temperaturen aus.

    print("\n--- Gesamttest über alle Temperaturen ---")
    contingency_table = create_contingency_table(df)
    test_result, expected = perform_chi2_test(contingency_table)

    if expected is not None:
        print(f"  Erwartete Häufigkeiten:\n{expected}")

    interpret_chi2_results(test_result, "alle")
    calculate_success_rates(contingency_table)


# Hauptprogramm
if __name__ == "__main__":
    path = r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt A"
    pd.set_option('display.max_columns', None)
    # Lädt und bereitet die Daten vor.
    df = load_and_prepare_data(path)

    # Führt den Chi-Quadrat-Test für jede Temperatur durch.
    for temp in df["temperature"].unique():
        chi2_test_by_temperature(df, temp)

    # Führt den Chi-Quadrat-Test für alle Temperaturen zusammen durch.
    chi2_test_all_temperatures(df)