"""
DataJoint schema for subject information.
"""

import datajoint as dj

schema = dj.schema('subject')

@schema
class Species(dj.Lookup):
    definition = """
    species: varchar(32)
    """
    contents = [['Mus musculus']]

@schema
class Strain(dj.Lookup):
    definition = """
    strain_name:                 varchar(64)
    ---
    strain_description:          varchar(255)
    strain_url:                  varchar(255)
    strain_ts=CURRENT_TIMESTAMP: timestamp
    """

@schema
class Subject(dj.Manual):
    definition = """
    subject_id:                   varchar(16)
    ---
    sex:                          enum("M", "F", "U")
    date_of_birth:                date
    -> Strain
    -> Species
    subject_ts=CURRENT_TIMESTAMP: timestamp
    """


