"""
DataJoint schema for subject information.
"""

import datajoint as dj

schema = dj.schema('subject')

@schema
class Species(dj.Lookup):
    definition = """
    species: varchar(24)
    """
    contents = [['Mus musculus']]

@schema
class Strain(dj.Manual):
    definition = """
    strain_name:                 varchar(255)
    ---
    strain_description=null:     varchar(255)
    strain_url=null:             varchar(255)
    strain_ts=CURRENT_TIMESTAMP: timestamp
    """

@schema
class Subject(dj.Manual):
    definition = """
    subject_id:                   varchar(255)
    ---
    sex:                          enum("M", "F", "U")
    date_of_birth:                date
    -> Strain
    -> Species
    subject_ts=CURRENT_TIMESTAMP: timestamp
    """
