import numpy as np
from .util import map_with_progress
from .util import check_arrays

def unfold_all(
        site_phases,
        translation_sfracs,
        translation_deperms,
        translation_phases,
        gpoint_sfracs,
        qpoint_sfrac,
        eigenvectors,
        gamma_only,
        progress_prefix,
):
    progress = None
    if progress_prefix is not None:
        def progress(done, count):
            print(f'{progress_prefix}Unfolding {done:>5} of {count} eigenvectors')

    return np.array(list(map_with_progress(
        eigenvectors, progress,
        lambda eigenvector: unfold_one(
            site_phases=site_phases,
            translation_sfracs=translation_sfracs,
            translation_deperms=translation_deperms,
            translation_phases=translation_phases,
            gpoint_sfracs=gpoint_sfracs,
            qpoint_sfrac=qpoint_sfrac,
            eigenvector=eigenvector.reshape((-1, 3)),
            gamma_only=gamma_only,
        )
    )))

# NOTE: For any change to this function, the corresponding function
#       in the Rust unfold_lib must be changed accordingly!
def unfold_one(
        site_phases,
        translation_sfracs,
        translation_deperms,
        translation_phases,
        gpoint_sfracs,
        qpoint_sfrac,
        eigenvector,
        gamma_only,
):
    """
    :param site_phases: Shape ``(quotient,)``, complex.
    The phase factors that must be multiplied into each site when producing
    displacements from the eigenvector. Should be of the form ``e ^ (i q.x)``.

    :param translation_sfracs: Shape ``(quotient, 3)``, real.
    The quotient space translations (PC lattice modulo super cell),
    in units of the supercell basis vectors.

    :param translation_deperms: Shape ``(quotient, sc_sites)``, integral.
    Permutations such that ``(carts + translation_carts[i])[deperms[i]]``
    is ordered like the original carts. (i.e. when applied to the coordinate
    data, it translates by ``-1 * translation_carts[i]``)

    For any vector of per-site metadata ``values``, ``values[deperms[i]]`` is
    effectively translated by ``translation_carts[i]``.

    :param translation_phases: Shape ``(quotient, sc_sites)``, complex.
    The phase factors that must be factored into each atom's components after
    a translation to account for the fact that permuting the sites does not
    produce the same images of sites as actually translating them would.

    :param gpoint_sfracs: Shape ``(quotient, 3)``, real.
    Translations in the reciprocal quotient space, in units of the reciprocal
    lattice basis vectors.

    (SC reciprocal lattice modulo primitive BZ)

    :param qpoint_sfrac: Shape ``(3,)``, real.
    The Q point in the SC reciprocal cell at which the eigenvector was computed.

    :param eigenvector: Shape ``(sc_sites, 3)``, complex.
    A normal mode of the supercell. (arbitrary norm)

    :param gamma_only: boolean.
    A switch that causes the output to only contain the gamma probability.
    (all other elements will be zero)

    :return: Shape ``(quotient,)``, real.
    Probabilities of `eigenvector` projected onto each point
    ``qpoint + gpoints[i]``.
    """

    site_phases = np.array(site_phases)
    translation_sfracs = np.array(translation_sfracs)
    translation_deperms = np.array(translation_deperms)
    gpoint_sfracs = np.array(gpoint_sfracs)
    qpoint_sfrac = np.array(qpoint_sfrac)
    eigenvector = np.array(eigenvector)
    sizes = check_arrays(
        site_phases = (site_phases, ['sc_sites'], np.complexfloating),
        translation_sfracs = (translation_sfracs, ['quotient', 3], np.floating),
        translation_deperms = (translation_deperms, ['quotient', 'sc_sites'], np.integer),
        gpoint_sfracs = (gpoint_sfracs, ['quotient', 3], np.floating),
        qpoint_sfrac = (qpoint_sfrac, [3], np.floating),
        eigenvector = (eigenvector, ['sc_sites', 3], [np.floating, np.complexfloating]),
    )
    # print(repr(eigenvector))

    # Function with the magnitudes of the eigenvector, but the phases
    # of the displacement vector.
    bloch_function = eigenvector * site_phases[:, None]

    # Expectation value of every translation operation.
    inner_prods = np.array([
        np.vdot(bloch_function, t_phases[:, None] * bloch_function[t_deperm])
        for (t_deperm, t_phases) in zip(translation_deperms, translation_phases)
    ])

    # Expectation value of each projector P(Q -> Q + G)
    def compute_for_g(g):
        # PBZ (Q + G) dot r for every r
        k_dot_rs = (qpoint_sfrac + g) @ translation_sfracs.T

        # Phases from Allen Eq 3.  Due to our differing phase conventions,
        # we have exp(+i...) rather than exp(-i...).
        phases = np.exp(2j * np.pi * k_dot_rs)

        prob = sum(inner_prods * phases) / sizes['quotient']

        # analytically, these are all real, positive numbers
        assert abs(prob.imag) < 1e-7
        assert -1e-7 < prob.real
        return max(prob.real, 0.0)

    if gamma_only:
        assert (gpoint_sfracs[0] == [0, 0, 0]).all()
        gpoint_probs = np.zeros(shape=(len(gpoint_sfracs),))
        gpoint_probs[0] = compute_for_g(gpoint_sfracs[0])
    else:
        gpoint_probs = np.array([compute_for_g(g) for g in gpoint_sfracs])

    # Recall that the eigenvector is not normalized because it could be the zero
    # vector (due to projection onto a layer).
    if not gamma_only:
        np.testing.assert_allclose(gpoint_probs.sum(), np.linalg.norm(eigenvector)**2, atol=1e-7)
    return gpoint_probs
