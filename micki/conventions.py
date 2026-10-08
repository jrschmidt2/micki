"""Global choice of modeling conventions: micki's own (the default) or
CatMap's.

    import micki
    micki.set_conventions('catmap')        # for the rest of the program
    with micki.conventions('catmap'):      # for objects built in the block
        ...

The conventions are read when species and reactions are constructed, and
each object keeps those it was built with (also in copies). Under 'catmap'
the defaults of arguments that are not given change to CatMap's:

- Gas: reference state 1 bar (pref=1) instead of 1 M; all species use
  ASE's standard atomic weights (ase.data.atomic_masses) instead of the
  most abundant isotope masses of micki.masses.
- Reaction: clip='coverage' (the transition state is raised to the higher
  of the initial and final states at the current coverages; steps without
  a transition state are barrierless) and alpha=0.5 (CatMap's
  'intermediate_state'). Transition states follow only the lateral
  interactions of the reactants and products, not their dE shifts, as in
  CatMap, where changing a species' energy does not move a transition
  state; for any alpha, and so also in thermodynamic rate control.
- Model: setting a lattice (configurational entropy, which CatMap does not
  have) warns; reactions and species built under different conventions
  cannot be combined in a Model.

Explicit arguments always win, e.g. Reaction(..., clip=None) (no
clipping), alpha=None (alpha computed self-consistently), or
Gas(..., rhoref=1.).
"""

import contextlib
import contextvars

CONVENTIONS = ('micki', 'catmap')

_conventions = contextvars.ContextVar('micki_conventions', default='micki')


class _Default:
    """Marks an argument that was not given (so None can be a value)."""

    def __repr__(self):
        return 'default'


DEFAULT = _Default()


def _check(name):
    if name not in CONVENTIONS:
        raise ValueError('conventions must be one of {}, not {!r}'.format(
            CONVENTIONS, name))


def set_conventions(name):
    """Use the conventions `name` ('micki' or 'catmap') from now on."""
    _check(name)
    _conventions.set(name)


def get_conventions():
    return _conventions.get()


@contextlib.contextmanager
def conventions(name):
    """Use the conventions `name` for objects built inside the block."""
    _check(name)
    token = _conventions.set(name)
    try:
        yield
    finally:
        _conventions.reset(token)
