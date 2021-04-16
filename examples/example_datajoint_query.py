"""
Retrieve some data from the DataJoint database.
"""

import uobrainflex.utils.djconnect
from uobrainflex.pipeline import subject as subjectSchema
from uobrainflex.pipeline import acquisition as acquisitionSchema

subject = subjectSchema.Subject()
print('All subjects:')
print(subject)

sessionsOneSubject = acquisitionSchema.BehaviorSession & 'subject_id = "BW020"'
print('Sessions subject BW020')
print(sessionsOneSubject)

sessionsOneSubjectDataFrame = sessionsOneSubject.fetch(format='frame')
print('Sessions subject BW020 as a pandas DataFrame')
print(sessionsOneSubjectDataFrame)

