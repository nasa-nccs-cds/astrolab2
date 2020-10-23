astrolab2
===============================

Jupyterlab workbench supporting visual exploration and classification of astronomical xray and light curve data.

Installation
------------

To install use pip:

    $ pip install astrolab
    $ jupyter nbextension enable --py --sys-prefix astrolab

To install for jupyterlab

    $ jupyter labextension install astrolab

For a development installation (requires npm),

    $ git clone https://github.com/nasa-nccs-cds/astrolab2.git
    $ cd astrolab2
    $ pip install -e .
    $ jupyter nbextension install --py --symlink --sys-prefix astrolab
    $ jupyter nbextension enable --py --sys-prefix astrolab
    $ jupyter labextension install js

When actively developing your extension, build Jupyter Lab with the command:

    $ jupyter lab --watch

This takes a minute or so to get started, but then automatically rebuilds JupyterLab when your javascript changes.

Note on first `jupyter lab --watch`, you may need to touch a file to get Jupyter Lab to open.

