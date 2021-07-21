"""
Add columns to a DataJoint table.

Note that according to:
https://docs.datajoint.org/python/definition/03-Table-Definition.html
"To change the table definition, one must first drop the existing table. 
This means that all the data will be lost, and the new definition will
be applied to create the new empty table."

Therefore, the steps to add columns are as follows:
1. Load the DJ table using the older definition.
2. Fetch the data (as a pandas dataframe).
3. Save the dataframe. We use DataFrame.to_pickle().
4. Change the table definition (e.g., in pipeline/acquisition.py).
5. Drop the table.
6. Restart python (if running in interactive mode).
7. Load the DJ table (it will use the new definition, but it will be empty)
8. Load the dataframe you saved.
9. Insert the dataframe data into the DJ table.

NOTES:
- Your DJ user needs to have permission to drop tables (the generic user may not).
- As of July 2021, using TABLE.alter(), which would be a better way,
  doesn't seem to work with definitions that include foreign keys.
- When inserting data that contains dates, I got the error:
   AttributeError: 'numpy.datetime64' object has no attribute 'translate'
  I followed a suggestion from the link below and converted these to strings.
   https://stackoverflow.com/questions/43108164/python-to-mysql-timestamp-object-has-no-attribute-translate
"""

import os
import sys
import pandas as pd
import uobrainflex.utils.djconnect
from uobrainflex.pipeline import acquisition as acquisitionSchema

backupFile = '/var/tmp/djtable.pkl'

behaviorSession = acquisitionSchema.BehaviorSession()
testTable = acquisitionSchema.TestTable()

step = int(input('\nWhich step are you running?\n[1] Back up table.\n[2] Restore table.\n'))

if step==1:
    # -- Create backup --
    dframe = behaviorSession.fetch(format='frame')
    if not os.path.isfile(backupFile):
        dframe.to_pickle(backupFile)
    else:
        raise IOError(f'Backup file {backupFile} exists. It will not be overwritten.')
    print(f'Table saved to {backupFile}')
    print('To drop the table run: behaviorSession.drop()')
    print('Then restart python')
    # behaviorSession.drop()
elif step==2:
    # -- Restore backup --
    dframe2 = pd.read_pickle(backupFile)
    # -- Convert datatime64 to str so SQL insert does not complain --
    for key in dframe2.keys():
        if dframe2[key].dtype=='datetime64[ns]':
            dframe2[key] = dframe2[key].astype(str)
    # -- Insert backed up data into new table --
    behaviorSession.insert(dframe2, skip_duplicates=True)
    #testTable.insert(dframe2, skip_duplicates=True)
    print(dframe2)
    print('\nThe table above was restored to DataJoint.')
else:
    print(f'Step {step} not defined.')


