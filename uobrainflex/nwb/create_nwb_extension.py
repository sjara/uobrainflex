"""
Create NWB extension for uobrainflex project.

You need to run script to create the .yaml files that define the extension.
"""

from pynwb.spec import NWBGroupSpec, NWBNamespaceBuilder
from pynwb.file import LabMetaData

# Create a builder for the namespace
ns_builder = NWBNamespaceBuilder(doc='Extension for uobrainflex metadata',
                                 name='uobrainflex_metadata',
                                 version='1.0',
                                 author='Santiago Jaramillo')

# Create extension
LabMetaData_ext = NWBGroupSpec(
    doc='Extension for uobrainflex metadata',
    neurodata_type_def='LabMetaData_ext',
    neurodata_type_inc='LabMetaData',
)

LabMetaData_ext.add_attribute(
        name='rig',
        doc='Rig where data was collected.',
        dtype='text',
        shape=None,
)

LabMetaData_ext.add_attribute(
        name='session_type',
        doc='Type of behavior session.',
        dtype='text',
        shape=None,
)

LabMetaData_ext.add_attribute(
        name='training_stage',
        doc='Stage of behavioral training.',
        dtype='text',
        shape=None,
)

LabMetaData_ext.add_attribute(
        name='auditory_stim_association',
        doc='Descriptor of auditory stimulus-lick port association.',
        dtype='text',
        shape=None,
)

LabMetaData_ext.add_attribute(
        name='visual_stim_association',
        doc='Descriptor of auditory stimulus-lick port association.',
        dtype='text',
        shape=None,
)

LabMetaData_ext.add_attribute(
        name='behavior_version',
        doc='Version of schema used to translate LabView data to NWB.',
        dtype='text',
        shape=None,
)

# Add the extension
ext_source = 'uobrainflex.specs.yaml'
ns_builder.include_type('LabMetaData', namespace='core')
ns_builder.add_spec(ext_source, LabMetaData_ext)

# Save the namespace and extensions
ns_path = 'uobrainflex.namespace.yaml'
ns_builder.export(ns_path)

