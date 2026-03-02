import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_XLSX = "dataset_maladie_40vars.xlsx"          # <-- chemin vers ton fichier
PRINT_ALL_ROWS = False            # True = imprime tout (peut être très long)
MAX_UNIQUE_FOR_CATEGORY = 30      # limite pour graphes par catégorie


def show_full(df: pd.DataFrame, sheet_name: str):
    print(f"\n===== ONGLET: {sheet_name} =====")
    if PRINT_ALL_ROWS:
        with pd.option_context(
                "display.max_rows", None,
                "display.max_columns", None,
                "display.width", 200,
                "display.max_colwidth", None
        ):
            print(df)
    else:
        print(f"Shape: {df.shape}")
        print("Aperçu (5 premières lignes):")
        print(df.head(5))


def plot_numeric_columns(df: pd.DataFrame, sheet_name: str):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        print(f"Aucune colonne numérique dans l'onglet {sheet_name}.")
        return

    # Courbes (valeur vs index) + histogrammes
    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            continue

        plt.figure(figsize=(10, 4))
        plt.plot(s.index.to_numpy(), s.to_numpy(), linewidth=1.5)
        plt.title(f"{sheet_name} - {col} (courbe)")
        plt.xlabel("Index (ligne)")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.hist(s.to_numpy(), bins=30)
        plt.title(f"{sheet_name} - {col} (histogramme)")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.tight_layout()
        plt.show()

    # Corrélation si au moins 2 colonnes numériques
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)

        plt.figure(figsize=(1 + 0.6 * len(num_cols), 1 + 0.6 * len(num_cols)))
        plt.imshow(corr.to_numpy(), interpolation="nearest")
        plt.title(f"{sheet_name} - Corrélation (numérique)")
        plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
        plt.yticks(range(len(num_cols)), num_cols)
        plt.colorbar()
        plt.tight_layout()
        plt.show()


def plot_category_vs_numeric(df: pd.DataFrame, sheet_name: str):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return

    # Choisir une colonne catégorielle "raisonnable"
    cat_candidates = []
    for c in df.columns:
        if c in num_cols:
            continue
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= MAX_UNIQUE_FOR_CATEGORY:
                cat_candidates.append((c, nunique))

    if not cat_candidates:
        return

    # on prend la plus petite cardinalité (plus lisible)
    cat_col = sorted(cat_candidates, key=lambda x: x[1])[0][0]

    for num_col in num_cols:
        tmp = df[[cat_col, num_col]].dropna()
        if tmp.empty:
            continue

        grouped = tmp.groupby(cat_col)[num_col].mean().sort_values(ascending=False)

        plt.figure(figsize=(10, 4))
        plt.bar(grouped.index.astype(str).to_numpy(), grouped.to_numpy())
        plt.title(f"{sheet_name} - Moyenne de {num_col} par {cat_col}")
        plt.xlabel(cat_col)
        plt.ylabel(f"Moyenne({num_col})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


def main():
    xls = pd.ExcelFile(INPUT_XLSX)
    print(f"Onglets détectés: {xls.sheet_names}")

    for sheet in xls.sheet_names:
        df = pd.read_excel(INPUT_XLSX, sheet_name=sheet)

        show_full(df, sheet)
        plot_numeric_columns(df, sheet)
        plot_category_vs_numeric(df, sheet)


if __name__ == "__main__":
    main()