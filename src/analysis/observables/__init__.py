"""Observables computed from the MD trajectories: structure (RDF, density) and
transport (self-diffusivity, shear viscosity, ionic conductivity).

The estimators live at the top of this package (``diffusivity.py``,
``viscosity_gk.py``, ``conductivity.py``, ``rdf.py``, ``density.py``) and are
driven by the scripts in ``structural/`` and ``transport/``. Every driver is run
from the repo root as a module, e.g.::

    python -m src.analysis.observables.structural.compute_painn_replica_rdfs
    python -m src.analysis.observables.transport.diffusivity_replicas --jobs 8

``paths.py`` is the single source of truth for where inputs and outputs live
(``python -m src.analysis.observables.paths`` prints the manifest). See
``README.md`` here for the package layout and ``transport/METHODS.md`` for the
transport methods.
"""
