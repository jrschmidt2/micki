"""Lateral interaction models of other codes, written as species.lateral"""

import sympy as sym


def _response(theta, response, slope, cutoff, smoothing):
    """CatMap's interaction response functions F(theta_tot)
    (catmap.functions.smooth_piecewise_linear)."""
    if response == 'linear':
        return sym.Float(slope)
    if response == 'piecewise_linear':
        smoothing = 0.
    elif response != 'smooth_piecewise_linear':
        raise ValueError("response must be 'linear', 'piecewise_linear' or "
                         "'smooth_piecewise_linear', not {!r}".format(response))
    x0 = cutoff - smoothing
    x1 = cutoff + smoothing
    # theta > x0 in the branches that divide by it; Max keeps the branches
    # that are not taken finite at theta = 0 (numpy evaluates all of them)
    denom = sym.Max(theta, x0) if x0 > 0 else theta
    pieces = [(0, theta <= x0)]
    if x1 > x0:
        a = slope / (2 * (x1 - x0))
        pieces.append((a * (theta - x0)**2 / denom, theta <= x1))
    pieces.append((slope * (theta - cutoff) / denom, True))
    return sym.Piecewise(*pieces)


def first_order(adsorbates, eps, response='linear', slope=1., cutoff=0.25,
                smoothing=0.05):
    """Set CatMap's first-order lateral interactions as species.lateral.

    For each species i in eps (adsorbates or transition states),

        i.lateral = sum_j F(theta_tot(j)) * eps[i][j] * theta_j,

    where theta_tot(j) is the total coverage of the site of j (the first
    entry of j.sites) by all non-TS species in adsorbates, and F is CatMap's
    interaction response function: 'linear' (F = slope), 'piecewise_linear'
    (0 up to cutoff, slope * (theta - cutoff) / theta above) or
    'smooth_piecewise_linear' (the same, with a quadratic piece between
    cutoff - smoothing and cutoff + smoothing). The defaults are CatMap's.

    eps maps species to {species: parameter in eV}, i.e. the rows of
    CatMap's (differential) interaction matrix, including any cross terms
    and transition-state rows; use the matrix CatMap reports (output
    variable 'interaction_matrix') to reproduce a CatMap model exactly.
    Transition states given here should normally be used with
    Reaction(..., explicit_ts=True). Existing lateral expressions of the
    species in eps are replaced.
    """
    coverage = {}
    labels = set()
    for species in adsorbates:
        if species.ts or not species.sites:
            continue
        site = species.sites[0]
        coverage[site] = coverage.get(site, 0) + species.symbol
        labels.add(species.label)
    for i, row in eps.items():
        lateral = 0
        for j, value in row.items():
            if j.ts or not j.sites:
                raise ValueError('{} is not an adsorbate; interactions are '
                                 'with adsorbate coverages'.format(j.label))
            if j.label not in labels:
                raise ValueError('{} is not in adsorbates'.format(j.label))
            F = _response(coverage[j.sites[0]], response, slope, cutoff,
                          smoothing)
            lateral += F * value * j.symbol
        i.lateral = lateral
