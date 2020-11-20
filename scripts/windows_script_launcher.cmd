:: This is a template useful for launching our python scripts on Windows
::
:: You can make a copy of this file, modify it, and use it to launch a
:: script by just double-clicking on the icon.

call C:\Users\%USERNAME%\Anaconda\Scripts\activate.bat
call conda activate uobrainflex
call python C:\Users\%USERNAME%\src\uobrainflex\scripts\test_installation.py
