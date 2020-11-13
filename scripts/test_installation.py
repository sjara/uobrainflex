"""
Test installation of uobrainflex package.
"""

# TODO:
# Check that each directory in the config file exists

from uobrainflex import config

print('The contents of your config file is:')

for section,lines in dict(config).items():
    print('[{}]'.format(section))
    for key,val in lines.items():
        print('  {} = {}'.format(key,val))

print('\nThe package seems ready for you to use.\n')
