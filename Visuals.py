import matplotlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns

matplotlib.use('TkAgg')


# Rohdaten werden vorbereitet.
def load_and_prepare_data(paths):
    # Initialisiere eine leere Liste, um die DataFrames zu speichern
    df_list = []
    # Iteriere durch die angegebenen Pfade
    for path in paths:
        # Extrahiere den Run-Namen aus dem Pfad
        run_name = os.path.basename(path)

        # Extrahiere die Run-Nummer aus dem Run-Namen
        if run_name == "Prompt B run 1":
            run_number = 4
        else:
            run_number = int(run_name.split(" ")[-1])  # Extrahiere die Zahl aus "Prompt A run 1"

        # Iteriere durch die Unterordner für jede Temperatur
        for temp_folder in ["t=0", "t=0.5", "t=1"]:
            temp_path = os.path.join(path, temp_folder)
            # Iteriere durch die Excel-Dateien im Unterordner
            for filename in os.listdir(temp_path):
                if filename.endswith(".csv"):
                    filepath = os.path.join(temp_path, filename)
                    df = pd.read_csv(filepath, nrows=61)
                    # Extrahiere die Temperatur aus dem Ordnernamen
                    df["temperature"] = float(temp_folder.split("=")[1])
                    if filename == "log_anthropic_claude-3.5-sonnet.csv":
                        filename = "log_anthropic_claude-3-5-sonnet-20241022.csv"
                    df["Modellname"] = filename.replace("log_", "").replace(".csv", "")
                    df["run"] = run_name  # Füge den Runname hinzu
                    df_list.append(df)

    # Kombiniere die DataFrames
    df = pd.concat(df_list, ignore_index=True)

    # Identifiziere Blackbox- und Open-Source-Modelle
    df["blackbox"] = df["Modellname"].apply(
        lambda x: False
        if x in ["deepseek_deepseek-chat", "qwen_qwen-2.5-coder-32b-instruct"]
        else True
    )

    df["error"] = df["Task"].apply(lambda x: x.rsplit("_", 1)[-1])

    # Solution-Spalte hinzufügen, die True ist, wenn eine der Iterationen erfolgreich war
    df["solution"] = df["Iteration 1"]
    # add column which is true if iteration 1 or 2 is true
    df['solution_final_it12'] = df[['Iteration 1', 'Iteration 2']].any(axis=1)
    df['solution_final_it123'] = df[['Iteration 1', 'Iteration 2', 'Iteration 3']].any(axis=1)

    # Filtere Daten auf Zellen, die mit "Task" beginnen
    df = df[df['Task'].str.startswith("Task")]

    return df


# Plottet die durchschnittliche Erfolgsrate pro Modell und Temperatur über alle Runs als Balkendiagramm.
def plot_mean_success_rate(df, run_name_I, run_name_II, run_name_III):

    # df filtern auf die drei runs
    df= df[df['run'].isin([run_name_I, run_name_II, run_name_III])]
    # Berechne die durchschnittliche Erfolgsrate pro Modell, Temperatur und Run
    mean_success_rate = df.groupby(["Modellname", "temperature", "run"])["solution"].mean().reset_index()

    # Berechne die durchschnittliche Erfolgsrate über alle Runs
    mean_success_rate = mean_success_rate.groupby(["Modellname", "temperature"])["solution"].mean().reset_index()

    # Farben für die Modelle definieren
    colors = {
        "anthropic_claude-3.5-haiku-20241022": "#f5b102",
        "anthropic_claude-3-5-sonnet-20241022": "#e86818",
        "deepseek_deepseek-chat": "#e2225c",
        "google_gemini-flash-1.5": "#e84cf3",
        "google_gemini-pro-1.5": "#22bad6",
        "openai_gpt-4o-2024-11-20": "#04a79d",
        "openai_gpt-4o-mini-2024-07-18": "#3e8600",
        "qwen_qwen-2.5-coder-32b-instruct": "#a3d96a",
    }

    # Neues Figure-Objekt erstellen
    fig = plt.figure(figsize=(16, 6))

    # Subplot erstellen
    ax = fig.add_subplot(1, 1, 1)

    # Balkendiagramm für jedes Modell erstellen
    bar_width = 0.2  # Breite der Balken
    gap_within_group = 0.1  # Abstand zwischen Balken innerhalb einer Gruppe
    gap_between_groups = 0.4  # Abstand zwischen den Modellgruppen

    model_names = mean_success_rate["Modellname"].unique()
    num_models = len(model_names)
    temperatures = mean_success_rate["temperature"].unique()
    num_temps = len(temperatures)

    for i, modell in enumerate(model_names):
        model_data = mean_success_rate[mean_success_rate["Modellname"] == modell]

        for j, temp in enumerate(temperatures):
            temp_data = model_data[model_data["temperature"] == temp]

            x_pos = i * (num_temps * bar_width + (num_temps - 1) * gap_within_group + gap_between_groups) + j * (
                        bar_width + gap_within_group)

            hatch = ''
            if temp == 0.5:
                hatch = '/'
            elif temp == 1:
                hatch = '..'

            bar = ax.bar(
                x_pos,
                temp_data["solution"].values[0] * 100 if not temp_data.empty else 0,
                width=bar_width,
                color=colors.get(modell, "blue"),
                hatch=hatch,
                edgecolor='white',
                label=modell if j == 0 else ""
            )

            # Wert über dem Balken hinzufügen
            if not temp_data.empty:
                ax.text(bar[0].get_x() + bar[0].get_width() / 2.,
                        bar[0].get_height() + 1,
                        f'{temp_data["solution"].values[0] * 100:.1f}',
                        ha='center', va='bottom')

    # X-Achsen-Beschriftung und Titel setzen
    ax.set_xlabel("Modell", fontsize=12)
    ax.set_ylabel("Erfolgsrate (%)", fontsize=12)
    ax.set_title("", fontsize=14)

    # Grid und Spines anpassen
    ax.grid(alpha=0.5, linestyle='--', linewidth=0.5, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # X-Achsen-Ticks und -Beschriftung anpassen
    group_positions = [i * (num_temps * (bar_width + gap_within_group) + gap_between_groups) + (
                num_temps * (bar_width + gap_within_group)) / 2 - bar_width / 2 for i in range(num_models)]
    ax.set_xticks(group_positions)
    ax.set_xticklabels(model_names, rotation=45, ha="right")

    # Y-Achsen-Limit setzen
    ax.set_ylim(0, 100)

    # Legende hinzufügen
    legend_elements = [
        mpatches.Patch(facecolor='gray', edgecolor='white', label='t=0'),
        mpatches.Patch(facecolor='gray', edgecolor='white', hatch='/', label='t=0.5'),
        mpatches.Patch(facecolor='gray', edgecolor='white', hatch='..', label='t=1')
    ]
    ax.legend(handles=legend_elements, title="", bbox_to_anchor=(1, 1), loc='upper left')

    # Layout anpassen
    fig.tight_layout()

    # Diagramm anzeigen
    plt.show()


# Plottet die durchschnittliche Erfolgsrate pro Modell und Iteration
def plot_mean_success_rate_per_iteration(df, temperature, run_name):

    # Filtere Daten für t=0
    df_t0 = df[df["temperature"] == temperature].copy()
    df_t0 = df_t0[df_t0["run"] == run_name]
    # Extrahiere Iterationsnummern
    iterations = [col for col in df_t0.columns if "Iteration" in col]

    # Konvertiere die Iterationsspalten in numerische Werte
    for iteration in iterations:
        df_t0[iteration] = pd.to_numeric(df_t0[iteration], errors='coerce')

    # Berechne die kumulierte Erfolgsrate für jede Iteration
    results = []
    for modell in df_t0["Modellname"].unique():
        df_modell = df_t0[df_t0["Modellname"] == modell]
        for i, iteration in enumerate(iterations):
            cumulative_successes = 0
            for j in range(i + 1):
                cumulative_successes += df_modell[iterations[j]].sum()

            cumulative_success_rate = cumulative_successes / (60)  # 60 Fragen
            results.append({
                "Modellname": modell,
                "Iteration": iteration,
                "Erfolgsrate": cumulative_success_rate,
                "Iterationsnummer": i + 1
            })

    mean_success_rate = pd.DataFrame(results)

    # Farben für die Modelle definieren
    colors = {
        "anthropic_claude-3-5-sonnet-20241022": "#e86818",
        "anthropic_claude-3.5-haiku-20241022": "#f5b102",
        "deepseek_deepseek-chat": "#e2225c",
        "google_gemini-flash-1.5": "#e84cf3",
        "google_gemini-pro-1.5": "#22bad6",
        "openai_gpt-4o-2024-11-20": "#04a79d",
        "openai_gpt-4o-mini-2024-07-18": "#3e8600",
        "qwen_qwen-2.5-coder-32b-instruct": "#a3d96a",
    }



    # Neues Figure-Objekt erstellen
    fig = plt.figure(figsize=(16, 6))

    # Subplot erstellen
    ax = fig.add_subplot(1, 1, 1)

    # Balkendiagramm für jedes Modell erstellen
    bar_width = 0.2
    gap_within_group = 0.1  # Abstand zwischen Balken innerhalb einer Gruppe
    gap_between_groups = 0.4 # Abstand zwischen den Modellgruppen
    group_width = bar_width * len(iterations) + gap_within_group * (len(iterations) -1) # Breite einer Modellgruppe
    model_names = mean_success_rate["Modellname"].unique()
    num_models = len(model_names)
    x_positions = np.arange(num_models) * (group_width + gap_between_groups) # Startposition jeder Modellgruppe

    for i, modell in enumerate(model_names):
        model_data = mean_success_rate[mean_success_rate["Modellname"] == modell]

        for j, row in model_data.iterrows():
            hatch = ''
            if row['Iterationsnummer'] == 2:
                hatch = '/'
            elif row['Iterationsnummer'] == 3:
                hatch = '..'
            bar = ax.bar(
                x_positions[i] + (row["Iterationsnummer"] - 1) * (bar_width + gap_within_group),
                row["Erfolgsrate"] * 100,  # Erfolgsrate in Prozent
                width=bar_width,
                color=colors.get(modell, "blue"),
                hatch=hatch,
                label=modell if row['Iterationsnummer'] == 1 else "",
                edgecolor='white'
            )

            # Prozentwert über dem Balken hinzufügen
            ax.text(bar[0].get_x() + bar[0].get_width() / 2.,
                    bar[0].get_height() + 1,  # 1% über dem Balken
                    f'{row["Erfolgsrate"] * 100:.1f}',  # Geänderte Formatierung
                    ha='center', va='bottom')


    # X-Achsen-Beschriftung und Titel setzen
    ax.set_ylabel("Erfolgsrate (%)", fontsize=12)
    ax.set_title(f"", fontsize=14)

    # Grid und Spines anpassen
    ax.grid(alpha=0.5, linestyle='--', linewidth=0.5, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # X-Achsen-Ticks und -Beschriftung anpassen
    ax.set_xticks(x_positions + group_width / 2 - bar_width / 2)
    ax.set_xticklabels(model_names, rotation=45, ha="right")

    # Y-Achsen-Limit setzen
    ax.set_ylim(0, 100)

    # Legende hinzufügen (nur für Iterationen)
    legend_elements = [
        mpatches.Patch(facecolor='gray', label='Iteration 1', edgecolor='white'),
        mpatches.Patch(facecolor='gray', hatch='/', label='Iteration 2', edgecolor='white'),
        mpatches.Patch(facecolor='gray', hatch='..', label='Iteration 3', edgecolor='white')
    ]
    ax.legend(handles=legend_elements, title="", bbox_to_anchor=(1, 1), loc='upper left')

    # Layout anpassen
    fig.tight_layout()

    # Diagramm anzeigen
    plt.show()


# Erstellt ein Balkendiagramm mit der durchschnittlichen Erfolgsrate pro Modell (aggregiert über alle vorhandenen Temperaturen, Runs, Iterationen und Prompts).
def plot_overall_success_rate(df):

    df = df[df['run'].isin(["Prompt A run 1", "Prompt B run 1"])]
    # 1) Aggregation: berechne Erfolgsrate pro Modell über alle Runs, Temperaturen, Iterationen und Prompts
    overall = df.groupby("Modellname")['solution_final_it123'].mean().reset_index()
    overall["solution"] = overall['solution_final_it123'] * 100  # in %

    # 2) Farben definieren
    colors = {
        "anthropic_claude-3.5-haiku-20241022": "#f5b102",
        "anthropic_claude-3-5-sonnet-20241022": "#e86818",
        "deepseek_deepseek-chat": "#e2225c",
        "google_gemini-flash-1.5": "#e84cf3",
        "google_gemini-pro-1.5": "#22bad6",
        "openai_gpt-4o-2024-11-20": "#04a79d",
        "openai_gpt-4o-mini-2024-07-18": "#3e8600",
        "qwen_qwen-2.5-coder-32b-instruct": "#a3d96a",
    }

    # 3) Figure und Axes erstellen
    fig, ax = plt.subplots(figsize=(10, 6))

    # Balkendiagramm
    bars = ax.bar(
        overall["Modellname"],
        overall["solution"],
        color=[colors.get(name, "blue") for name in overall["Modellname"]]
    )

    # 4) Achsenbeschriftung, Titel, Layout
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("Erfolgsrate (%)", fontsize=12)
    ax.set_title("", fontsize=14)
    ax.set_ylim(0, 100)

    # X-Achse lesbar machen (wenn die Modellnamen lang sind)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Gitter und Spines anpassen
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Balken mit Prozentzahlen beschriften
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 Punkte vertikaler Abstand
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    plt.show()


"""def plot_success_rate_by_error_and_temperature(df):

    # Berechne die Anzahl der korrekten Antworten (True) pro Modell, Temperatur, Fehlertyp und Run
    success_counts = df.groupby(["Modellname", "temperature", "error", "run"])["solution"].sum().reset_index()

    # Berechne die durchschnittliche Anzahl korrekter Antworten über alle Runs
    avg_success_counts = success_counts.groupby(["Modellname", "temperature", "error"])["solution"].mean().reset_index()

    # Normalisiere die durchschnittliche Anzahl korrekter Antworten, indem du durch die Anzahl der Fragen pro Fehlertyp teilst (hier 20)
    avg_success_counts["solution"] = avg_success_counts["solution"] / 20

    # Pivot-Tabelle erstellen, um die Daten für das gestapelte Balkendiagramm vorzubereiten
    pivot_df = avg_success_counts.pivot_table(
        values="solution",
        index=["Modellname", "temperature"],
        columns="error",
        fill_value=0
    ).reset_index()

    # Farben für die Modelle definieren
    colors = {
        "anthropic_claude-3.5-haiku-20241022": "#f5b102",
        "anthropic_claude-3-5-sonnet-20241022": "#e86818",
        "deepseek_deepseek-chat": "#e2225c",
        "google_gemini-flash-1.5": "#e84cf3",
        "google_gemini-pro-1.5": "#22bad6",
        "openai_gpt-4o-2024-11-20": "#04a79d",
        "openai_gpt-4o-mini-2024-07-18": "#3e8600",
        "qwen_qwen-2.5-coder-32b-instruct": "#a3d96a",
    }

    # Fehlertypen in einer definierten Reihenfolge sortieren
    error_order = ["error_free", "logical_error", "factual_error", "unsolvable_error", "omission_error",
                   "extraction_error"]

    # Temperaturen sortieren
    temperature_order = [0, 0.5, 1]

    # Erstelle ein gestapeltes Balkendiagramm für jedes Modell
    for modell in pivot_df["Modellname"].unique():
        modell_df = pivot_df[pivot_df["Modellname"] == modell]

        # Sortiere die Daten nach der definierten Temperaturreihenfolge
        modell_df["temperature"] = pd.Categorical(modell_df["temperature"], categories=temperature_order, ordered=True)
        modell_df = modell_df.sort_values("temperature")"""


# Erstellt ein Balkendiagramm, das die Erfolgsraten der Modelle für Prompt A und Prompt B darstellt.
def plot_success_rate_by_prompt(df, run_name_I, run_name_II):

    # Filtere die Daten: nur t=0 und Iteration 1
    filtered_data = df[(df["temperature"] == 0) & df["run"].isin([run_name_I, run_name_II])]

    # Berechne die durchschnittliche Erfolgsrate pro Modell und Prompt
    mean_success_rate = (
        filtered_data.groupby(["Modellname", "run"])["solution"]
        .mean()
        .reset_index(name="success_rate")
    )

    # Pivotiere die Daten, um die Prompts als Spalten zu erhalten
    pivot_df = mean_success_rate.pivot(index="Modellname", columns="run", values="success_rate").reset_index()
    pivot_df.columns = ["Modellname", "Prompt A run 1", "Prompt B run 1"]

    # Farben für die Modelle definieren
    colors = {
        "anthropic_claude-3.5-haiku-20241022": "#f5b102",
        "anthropic_claude-3-5-sonnet-20241022": "#e86818",
        "deepseek_deepseek-chat": "#e2225c",
        "google_gemini-flash-1.5": "#e84cf3",
        "google_gemini-pro-1.5": "#22bad6",
        "openai_gpt-4o-2024-11-20": "#04a79d",
        "openai_gpt-4o-mini-2024-07-18": "#3e8600",
        "qwen_qwen-2.5-coder-32b-instruct": "#a3d96a",
    }

    # Breite der Balken
    bar_width = 0.2
    # Abstand innerhalb der Gruppe
    gap_within_group = 0.1 # fast kein Abstand innerhalb der Gruppe
    # Abstand zwischen den Gruppen
    gap_between_groups = 0.4

    # X-Positionen für die Balken
    r1 = np.arange(len(pivot_df)) * (bar_width * 2 + gap_between_groups) # Platz für 2 Balken und den Abstand zwischen Gruppen
    r2 = [x + bar_width + gap_within_group for x in r1]

    # Balkendiagramm erstellen
    fig, ax = plt.subplots(figsize=(12, 6))

    # Balken für Prompt A
    bars_a = ax.bar(r1, pivot_df["Prompt A run 1"] * 100,
                    color=[colors.get(name, "blue") for name in pivot_df["Modellname"]], width=bar_width,
                    edgecolor='white', label='Prompt A run 1')

    # Balken für Prompt B mit denselben Farben wie Prompt A, aber schraffiert
    bars_b = ax.bar(r2, pivot_df["Prompt B run 1"] * 100,
                    color=[colors.get(name, "blue") for name in pivot_df["Modellname"]], width=bar_width,
                    edgecolor='white', hatch='//', label='Prompt B run 1')

    # Achsenbeschriftungen, Titel, Layout
    ax.set_xlabel("Modell", fontsize=12)
    ax.set_ylabel("Erfolgsrate (%)", fontsize=12)
    ax.set_title("", fontsize=14)
    ax.set_xticks([r + bar_width/2 + gap_within_group/2 for r in r1]) # X-Ticks mittig in der Gruppe
    ax.set_xticklabels(pivot_df["Modellname"], rotation=45, ha="right")
    ax.set_ylim(0, 100)

    # Legende anpassen:  Prompt B schraffiert
    legend_elements = [
        mpatches.Patch(facecolor='gray', edgecolor='white', label='Prompt A run 1'),
        mpatches.Patch(facecolor='gray', edgecolor='white', hatch='//', label='Prompt B run 1')
    ]
    ax.legend(handles=legend_elements)

    # Gitter und Spines anpassen
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Balken mit Prozentzahlen beschriften
    for bar in bars_a + bars_b:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 Punkte vertikaler Abstand
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Layout anpassen
    fig.tight_layout()

    # Diagramm anzeigen
    plt.show()

# Balkendiagramm (Erfolgsrate) mit separaten Balken für jede Modell-Fehler-Kombination
def plot_success_rate_by_error_and_model(df, run_name, iteration):

    # Daten filtern
    filtered_data = df[(df["temperature"] == 0) & (df[iteration] == True) & (df['run'] == run_name)]

    # Erfolge zählen und Erfolgsrate berechnen
    success_counts = filtered_data.groupby(["Modellname", "error"])[iteration].sum().reset_index()
    success_counts["success_rate"] = (success_counts[iteration] / 20) * 100

    # Farben für die Fehler (nur die drei vorgegebenen)
    error_colors = {
        "logicerror": "#008080",  # Teal für Logikfehler
        "syntaxerror": "#800080",  # Purple für Syntaxfehler
        "runtimeerror": "#FFA500"   # Orange für Laufzeitfehler
    }

    # Eindeutige Fehlertypen und Modellnamen ermitteln
    # Berücksichtigt werden nur die Fehler, die in error_colors definiert sind
    error_types = sorted([error for error in success_counts['error'].unique() if error in error_colors])
    models = sorted(success_counts['Modellname'].unique())

    # An Abstände von plot_success_rate_by_prompt angepasste Parameter
    bar_width = 0.2
    gap_within_group = 0.1
    gap_between_groups = 0.4

    # X-Positionen und Labels berechnen
    x_positions = []
    x_tick_labels = []
    x_tick_positions = []

    for i, model in enumerate(models):
        start_pos = i * (len(error_types) * bar_width + gap_between_groups)
        model_error_positions = [start_pos + j * (bar_width + gap_within_group/len(error_types)) for j in range(len(error_types))]
        x_positions.extend(model_error_positions)
        x_tick_positions.append(np.mean(model_error_positions))
        x_tick_labels.append(model)

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(15, 6))  # Angepasste Figure Größe

    # Balken erstellen (mit Farben nach Fehlertyp)
    # Nur Balken für die drei definierten Fehlertypen erstellen
    for i, row in success_counts.iterrows():
        if row['error'] in error_colors:
            model_index = models.index(row['Modellname'])
            error_index = error_types.index(row['error'])
            position = x_positions[model_index * len(error_types) + error_index]
            ax.bar(position, row['success_rate'], width=bar_width, color=error_colors.get(row['error']), label=row['error'])

    # Beschriftung über den Balken
    for i, row in success_counts.iterrows():
        if row['error'] in error_colors:
            height = row['success_rate']
            model_index = models.index(row['Modellname'])
            error_index = error_types.index(row['error'])
            position = x_positions[model_index * len(error_types) + error_index]
            ax.annotate(f'{height:.1f}%',
                        xy=(position, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Achsenbeschriftung und Titel
    ax.set_xlabel("Modell", fontsize=12)
    ax.set_ylabel("Erfolgsrate (%)", fontsize=12)
    # titel anpassen positionierung weiter nach oben
    ax.set_title("", fontsize=14, y=1.05)

    # X-Ticks und Labels anpassen
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 100)

    # Legende anpassen: bbox_to_anchor=(1, 1) und loc='upper left'
    handles = [mpatches.Patch(facecolor=error_colors[error_type], label=error_type) for error_type in error_types]
    ax.legend(handles=handles, title="", bbox_to_anchor=(1, 1), loc='upper left')

    # Grid und Spines anpassen
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    # Diagramm anzeigen
    plt.show()


# Heatmap für die Erfolgsrate pro Modell und Aufgabe.
def plot_task_model_iterations(df, run_name):

    df = df[df['run'] == run_name]
    # Validierung des DataFrames
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df muss ein Pandas DataFrame sein.")

    required_columns = ['Modellname', 'Task', 'Iteration 1', 'Iteration 2', 'Iteration 3']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame muss die Spalten {required_columns} enthalten.")

    iterations = ['Iteration 1', 'Iteration 2', 'Iteration 3']

    # Funktion zur Bestimmung der ersten erfolgreichen Iteration
    def get_first_success(row):
        for i, iter_col in enumerate(iterations, start=1):
            if row.get(iter_col, False):
                return i
        return 0  # 0 bedeutet keine Lösung

    # Neue Spalte hinzufügen
    df['first_success_iteration'] = df.apply(get_first_success, axis=1)

    # Pivot-Tabelle erstellen: Zeilen = Modelle, Spalten = Aufgaben, Werte = Iterationsnummer
    pivot_df = df.pivot_table(index='Modellname', columns='Task', values='first_success_iteration', fill_value=0)

    # Sicherstellen, dass die Aufgaben in der richtigen Reihenfolge sind (numerisch oder alphabetisch)
    try:
        sorted_tasks = sorted(pivot_df.columns, key=lambda x: int(x.replace('Task', '')))
    except ValueError:
        sorted_tasks = sorted(pivot_df.columns)
    pivot_df = pivot_df[sorted_tasks]

    # Farbkarte definieren
    cmap = mcolors.ListedColormap(['red', 'green', 'yellow', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Figgröße dynamisch basierend auf der Anzahl der Aufgaben und Modelle
    num_tasks = len(sorted_tasks)
    num_models = len(pivot_df.index)
    fig_width = max(8, num_tasks * 0.7)  # Skalierung angepasst
    fig_height = max(5, num_models * 0.5)  # Skalierung angepasst

    plt.figure(figsize=(fig_width, fig_height))

    # Heatmap erstellen
    ax = sns.heatmap(
        pivot_df,
        cmap=cmap,
        norm=norm,
        linewidths=0.5,
        linecolor='gray',
        cbar=False,  # Wir erstellen eine benutzerdefinierte Legende
        square=False,
        xticklabels=True,
        yticklabels=True,
        annot=False # Annotationen entfernt
    )

    # Achsenbeschriftungen und Titel
    ax.set_xlabel('Aufgaben', fontsize=14)
    ax.set_ylabel('Modelle', fontsize=14)
    ax.set_title('Lösungsiteration je Modell und Aufgabe', fontsize=16)

    # X-Ticks anpassen
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    # Benutzerdefinierte Legende erstellen
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Iteration 1'),
        Patch(facecolor='yellow', edgecolor='black', label='Iteration 2'),
        Patch(facecolor='blue', edgecolor='black', label='Iteration 3'),
        Patch(facecolor='red', edgecolor='black', label='Keine Lösung')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left', title='Iterationsstatus', fontsize=10)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.4)  # Anpassen der Ränder

    plt.show()


# Balkendiagramm für Gesamtpunktzahl der Modelle in Abhängigkeit von der Lösungsiteration
def plot_iteration_scores(df, run_name, temperature, sort_by_score=True):

    # Daten filtern
    filtered_df = df[(df['run'] == run_name) & (df['temperature'] == temperature)]
    iterations = ['Iteration 1', 'Iteration 2', 'Iteration 3']
    model_scores = {}

    for model in filtered_df['Modellname'].unique():
        model_df = filtered_df[filtered_df['Modellname'] == model]
        score = 0
        for index, row in model_df.iterrows():
            for i, iteration in enumerate(iterations):
                if row[iteration]:
                    score += (3 - i)
                    break
        model_scores[model] = score

    # Sortierung hinzufügen
    if sort_by_score:
        model_scores = dict(sorted(model_scores.items(), key=lambda item: item[1], reverse=True))

    model_names = list(model_scores.keys())
    scores = list(model_scores.values())

    # Farben für die Modelle definieren (wiederverwendet)
    colors = {
        "anthropic_claude-3.5-haiku-20241022": "#f5b102",
        "anthropic_claude-3-5-sonnet-20241022": "#e86818",
        "deepseek_deepseek-chat": "#e2225c",
        "google_gemini-flash-1.5": "#e84cf3",
        "google_gemini-pro-1.5": "#22bad6",
        "openai_gpt-4o-2024-11-20": "#04a79d",
        "openai_gpt-4o-mini-2024-07-18": "#3e8600",
        "qwen_qwen-2.5-coder-32b-instruct": "#a3d96a",
    }
    bar_colors = [colors.get(model, "blue") for model in model_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(model_names, scores, color=bar_colors)

    ax.set_xlabel("Modell", fontsize=12)
    ax.set_ylabel("Punktzahl", fontsize=12)
    ax.set_title(f"Punktzahl pro Modell (basierend auf Iterationen) für {run_name} und t={temperature}", fontsize=14)
    ax.set_xticklabels(model_names, rotation=45, ha="right")

    # Beschriftung über den Balken
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    plt.show()

# Hauptprogramm
if __name__ == "__main__":
    paths = [
        r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt A run 1",
        r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt A run 2",
        r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt A run 3",
        r"C:\Users\MaximilianStoepler\OneDrive - Deutsche Bahn\Studium\Masterarbeit\Ergebnisse\Prompt B run 1"
    ]

    prepared_data = load_and_prepare_data(paths)

    # Liniendiagramm der durchschnittlichen Erfolgsraten je Temeratur plotten über alle Runs (Prompt A) in paths
    plot_mean_success_rate(prepared_data, "Prompt A run 1", "Prompt A run 1", "Prompt A run 1")

    # Liniendiagramm der durchschnittlichen Erfolgsraten pro Iteration für t=0, 0.5 und 1 plotten
    for temperature in [0, 0.5, 1]:
        plot_mean_success_rate_per_iteration(prepared_data, temperature, "Prompt A run 1")

    # Balkendiagramm mit "Gesamtüberblick" über alle Bedingungen
    plot_overall_success_rate(prepared_data)

    # Balkendiagramm der Erfolgsraten für Prompt A und Prompt B
    plot_success_rate_by_prompt(prepared_data, "Prompt A run 1", "Prompt B run 1")

    # Balkendiagramm der Erfolgsraten je Error je Modell
    plot_success_rate_by_error_and_model(prepared_data, "Prompt A run 1", "solution_final_it12") # Variable die je Iteration berechnet. Ändern auf solution_final_it12 für Iteration 1 & 2 und auf solution_final_it123 für alle Iterationen

    #Heatmap der Lösungsiterationen je Modell und Aufgabe
    plot_task_model_iterations(prepared_data,"Prompt A run 1")

    # Balkendiagramm der Punktzahlen je Modell
    plot_iteration_scores(prepared_data, "Prompt A run 1", 0.0)  # anderer Temperaturwert

