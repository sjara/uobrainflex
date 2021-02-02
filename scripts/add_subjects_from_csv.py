"""
Read CSV file with information about subjects and add each one to DataJoint.

Run this script by specifying the CSV file as an argument:
python add_subjects_from_csv.py subjects.csv

NOTE: this script assumes you have already added the required strains.

CSV FORMAT (example):
subject_id,sex,date_of_birth,strain_name,species
testcsv001,U,2020-02-01,C57BL/6J,Mus musculus
testcsv002,U,2020-02-01,C57BL/6J,Mus musculus
"""

import sys
import csv
import uobrainflex.utils.djconnect
from uobrainflex.pipeline import subject as subjectSchema


if len(sys.argv) < 2:
    raise ValueError('You need to specify a CSV file with subject information.')

filename = sys.argv[1]

subjectTable = subjectSchema.Subject()

with open(filename) as csvfile:
    reader = csv.DictReader(csvfile)
    for oneSubject in reader:
        subjectTable.insert1(oneSubject, skip_duplicates=True)
        print(oneSubject)
        
