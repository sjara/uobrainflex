"""
Provides an interface to add or delete items from a DataJoint database.
"""

from uobrainflex.utils import djtools
from uobrainflex.pipeline import subject as subjectSchema
from uobrainflex.pipeline import acquisition as acquisitionSchema
from uobrainflex.pipeline import experimenter as experimenterSchema


specialFormat = {'date_of_birth':'YYYY-MM-DD',
                 'sex':'M/F/U'}

subject = subjectSchema.Subject()
strain = subjectSchema.Strain()
species = subjectSchema.Species()
behaviorSession = acquisitionSchema.BehaviorSession()
lab = experimenterSchema.Lab()
experimenter = experimenterSchema.Experimenter()

tablesDict = {'subject':subject,
              'strain':strain,
              'species':species,
              'behavior_session':behaviorSession,
              'lab':lab,
              'experimenter':experimenter}
tableNames = list(tablesDict)

# -- Get primary keys --
primaryKeys = []
for tableName, tableDB in tablesDict.items():
    primaryKeys.append(tableDB.primary_key[0])

# --- Select table ---
print('')
for indk, tableName in enumerate(tableNames):
    print('[{}] {}'.format(indk, tableName))
tableInd = int(input('\nWhat would you like to edit (enter number from the list above): '))
selectedTable = tablesDict[tableNames[tableInd]]


keepLooping = True
while keepLooping:
    dataFromTable = selectedTable.fetch(format='frame')
    print('')
    print(dataFromTable)
    
    # --- Add or delete ---
    print('\n[0] Add item\n[1] Delete item\n[2] Quit')
    operationInd = int(input('\nWhat would you like to do (enter number): '))

    if operationInd==0:    # Add item
        primaryKey = dataFromTable.index.name
        columns = list(dataFromTable.reset_index())
        entryDict = {}
        print('\n\nEnter attributes below')
        for colname in columns:
            if colname[-3:]=='_ts':
                # Timestamps are set automatically
                attribValue = None
            elif (colname!=primaryKey) and (colname in primaryKeys):
                subTableInd = primaryKeys.index(colname)
                subTable = tablesDict[tableNames[subTableInd]]
                dataFromSubTable = subTable.fetch(format='frame')
                print(dataFromSubTable.reset_index())
                attribInd = int(input('{} (enter row number): '.format(colname)))
                attribValue = dataFromSubTable.index[attribInd]
            else:
                if colname in specialFormat:
                    attribValue = input('{} ({}): '.format(colname,specialFormat[colname]))
                else:
                    attribValue = input('{}: '.format(colname))
            entryDict[colname] = attribValue
        selectedTable.insert1(entryDict, skip_duplicates=True)
        print('\nSUCCESS! The new entry has been added to the table:\n')
        print(selectedTable)
    elif operationInd==1:  # Delete item
        print(dataFromTable.reset_index())
        itemInd = int(input('\nRow to delete: '))
        primaryKey = dataFromTable.index.name
        rowKey = dataFromTable.iloc[itemInd].name
        query = selectedTable & '{}="{}"'.format(primaryKey, rowKey)
        query.delete()
        print('\nSUCCESS! The entry has been deleted from the table:\n')
        print(selectedTable)
    else:
        keepLooping = False
