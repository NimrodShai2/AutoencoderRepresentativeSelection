import pandas as pd


def read_and_process_UHGG(file_path):
    """
    Reads the UHGG file and processes it into a DataFrame, including only
    the genome quality columns.

    Args:
        file_path (str): Path to the UHGG metadata file(.tsv).

    Returns:
        pd.DataFrame: Processed DataFrame with 'genome' and 'species' columns.
    """
    df = pd.read_csv(file_path, sep='\t')

    # Select relevant columns and preprocess the data
    features = ['Length', 'N_contigs', 'N50', 'GC_content',
                'Contamination', 'rRNA_5S', 'rRNA_16S', 'rRNA_23S', 'tRNAs', 'Completeness']

    # Ensure all features  are numeric
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

    # Clean up the DataFrame from any rows with NaN or non-logical values
    df = df.dropna(subset=features)
    df.loc[:, 'Completeness'] = df['Completeness'].clip(0, 100)
    df.loc[:, 'Contamination'] = df['Contamination'].clip(0, 100)
    df.loc[:, 'GC_content'] = df['GC_content'].clip(0, 100)

    df = df[features]

    return df


def read_full_UHGG(file_path):
    """
    Reads the full UHGG file and processes it into a DataFrame, including all columns.

    Args:
        file_path (str): Path to the UHGG metadata file(.tsv).

    Returns:
        pd.DataFrame: Processed DataFrame with all columns.
    """
    df = pd.read_csv(file_path, sep='\t')
    return df
