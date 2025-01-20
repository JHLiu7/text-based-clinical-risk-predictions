import pandas as pd
import numpy as np
import pickle as pk
import os, re 
from datetime import datetime
from tqdm import tqdm
import argparse


def clean(n): 
    return re.sub(r'\[\*\*(.*?)\*\*\]', '___', n)

def main():
    cohort_path = '/path/to/test-cohort-3386.csv'
    MIMIC_DIR = '/path/to/mimic-iii'
    output_path = '/path/to/data-mort-test.csv'

    note_df = pd.read_csv(os.path.join(MIMIC_DIR, 'NOTEEVENTS.csv.gz'), usecols=['ROW_ID', 'TEXT'])

    cohort_df = pd.read_csv(cohort_path)

    tcol = 'text_by_day1'

    def retrieve_notes(ids):
        notes = note_df[note_df['ROW_ID'].isin(ids)]['TEXT'].tolist()
        return ' '.join([clean(note) for note in notes])

    cohort_df[tcol] = cohort_df['NOTE_ROW_ID_BY_DAY1'].apply(
        lambda x: retrieve_notes(list(map(int, x.strip('[]').split(','))))
    )

    cohort_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
