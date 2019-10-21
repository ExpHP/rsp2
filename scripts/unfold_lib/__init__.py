import numpy as np
from pymatgen import Structure

from . import unfold_py, unfold_rust
from .util import cartesian_product

def prepare_implementation(implementation):
    if implementation in ['rust', 'auto']:
        try:
            unfold_rust.build()
        except unfold_rust.BuildError:
            assert unfold_rust.unfold_all is None
            if implementation == 'rust':
                raise

def unfold_all(
        superstructure: Structure,
        supercell: 'Supercell',
        eigenvectors,
        qpoint_sfrac,
        translation_deperms,
        gamma_only: bool,
        implementation,
        progress_prefix = None,
):
    """
    :param superstructure: ``pymatgen.Structure`` object with `sites` sites.
    :param supercell: ``Supercell`` object.
    :param eigenvectors: Shape ``(num_evecs, 3 * sites)``, complex or real.

    Each row is an eigenvector.  Their norms may be less than 1, if the
    structure has been projected onto a single layer, but should not exceed 1.
    (They will NOT be automatically normalized by this function, as projection
    onto a layer may create eigenvectors of zero norm)

    :param translation_deperms:  Shape ``(quotient, sites)``.
    Permutations such that ``(carts + translation_carts[i])[deperms[i]]`` is
    equivalent to ``carts`` under superlattice translational symmetry, where
    ``carts`` is the supercell carts.

    :param qpoint_sfrac: Shape ``(3,)``, real.
    The K point in the SC reciprocal cell at which the eigenvector was computed,
    in fractional coords.

    :param implementation: ``"rust"`` or ``"python"``

    :param gamma_only: When `True`, only the values at gamma will be computed,
    and the rest of the output will be zero. For safety, this function will
    validate that the first `qpoint_sfrac` is `[0, 0, 0]`.

    :param progress_prefix: String used to prefix progress reports.
    ``None`` disables progress reporting.

    :return: Shape ``(num_evecs, quotient)``
    For each vector in ``eigenvectors``, its projected probabilities
    onto ``k + g`` for each g in ``supercell.gpoint_sfracs()``.
    """
    super_lattice = superstructure.lattice.matrix
    prim_lattice = np.linalg.inv(supercell.matrix) @ super_lattice

    # The quotient group of primitive lattice translations modulo the supercell
    translation_carts = supercell.translation_pfracs() @ prim_lattice
    translation_sfracs = translation_carts @ np.linalg.inv(super_lattice)

    # debug_quotient_points(translation_carts[:, :2], super_lattice[:2,:2])

    # The quotient group of supercell reciprocal lattice points modulo the
    # primitive reciprocal cell
    gpoint_sfracs = supercell.gpoint_sfracs()

    # debug_quotient_points((gpoint_sfracs @ np.linalg.inv(super_lattice).T)[:, :2], np.linalg.inv(prim_lattice).T[:2,:2])

    super_lattice_recip = superstructure.lattice.inv_matrix.T
    qpoint_cart = qpoint_sfrac @ super_lattice_recip
    super_carts = superstructure.cart_coords

    # The method is defined on a Bloch function with wavevector Q.  The
    # eigenvector does not satisfy this property; it is periodic under the
    # lattice. Rather, the *displacements* are a Bloch function.
    #
    # Because the sqrt(mass) factors could mess with our normalization, we don't
    # construct the true displacement vectors, but rather some sort of
    # Frankenstein creation that has magnitudes of the eigenvector and the
    # phases of the displacement vector.
    #
    # These are those phase factors.
    # (remember we follow the sign convention of Phonopy, not of Allen)
    site_phases = np.exp(2j * np.pi * np.dot(super_carts, qpoint_cart))

    # We also require correction phase factors to accompany the permutation
    # representation of our translation operators, to handle when sites are
    # mapped to different images of the supercell.
    #
    # See the comment above get_translation_phases for more details.
    translation_phases = np.vstack([
        get_translation_phases(
            qpoint_cart=qpoint_cart,
            super_carts=super_carts,
            translation_cart=translation_cart,
            translation_deperm=translation_deperm,
        )
        for (translation_cart, translation_deperm)
        in zip(translation_carts, translation_deperms)
    ])

    if unfold_rust.unfold_all is not None and implementation != 'python':
        func = unfold_rust.unfold_all
    else:
        func = unfold_py.unfold_all

    return func(
        site_phases=site_phases,
        translation_sfracs=translation_sfracs,
        translation_deperms=translation_deperms,
        translation_phases=translation_phases,
        gpoint_sfracs=gpoint_sfracs,
        qpoint_sfrac=qpoint_sfrac,
        eigenvectors=eigenvectors,
        gamma_only=gamma_only,
        progress_prefix=progress_prefix,
    )

# Legend:
# - a diagram of integers (labeled "Indices") depicts coordinates, by displaying
#   the number `i` at the position of the `i`th atom.
# - a diagram with any other label depicts a list of metadata (such as elements
#   or eigenvector components) by arranging them starting from index 0 in the
#   lower left, and etc. as if they were to label the original coords.
# - Parentheses surround the position of the original zeroth atom.
# - Suppose the qpoint Q of the eigenvector is orthogonal to the x axis.
#   We let α = exp(i Q.r), where r is the vector from atom 0 to atom 3.
#   Letters a-i are 3-vectors containing the components of the eigenvector
#   at each site.
#
#                 6  7  8    Bloch Function:     gα²  hα²  iα²
#    Indices:    3  4  5     (phonopy phase     dα   eα   fα
#              (0) 1  2       convention)     (a)   b    c
#
# Consider the translation that moves the 0th atom to the location originally at
# index 3. The *correct* bloch function should be:
#
#
#       Expected            dα    eα    fα
#       Bloch Function:    a     b     c
#                       (gα−¹) hα−¹  iα−¹
#
# However, naively applying the deperm to the bloch function (to "translate" it
# by this vector) instead yields:
#
#       Permuted            dα    eα    fα
#       Bloch Function:    a     b     c
#                       (gα²)  hα²   iα²
#
# As you can see, g, h, and i do not have the correct phases because those
# atoms mapped to different images. To find the superlattice translation that
# describes this discrepancy, we must look at the coords.  First, translate the
# coords by literally applying the translation.  Then, apply the inverse coperm
# to make the indices match their original sites.
#
#   (applying translation...)      (...then applying inverse coperm)
#                  6  7  8                     0  1  2
#                 3  4  5                     6  7  8
#    Indices:    0  1  2         Indices:    3  4  5
#              (x) x  x                    (x) x  x
#
# If you subtract these from the original coordinates, you get a list of
# super-lattice point translations that map the permuted atoms to their correct
# images. Atoms 3..9 need no correction, while atoms 0..3 require a phase
# correction by some super-lattice vector R.
# The correction to be applied is exp(i Q.R), which in this case is α−³.
#
#                       0  0  0                         1    1    1
#    Image vectors:    0  0  0    Phase corrections:   1    1    1
#                    (R) R  R                         α−³  α−³  α−³
#
# Those phase corrections are the output of this function.
def get_translation_phases(
        qpoint_cart,
        super_carts,
        translation_cart,
        translation_deperm,
):
    inverse_coperm = translation_deperm # inverse of inverse

    # translate, permute, and subtract to get superlattice points
    image_carts = super_carts - (super_carts + translation_cart)[inverse_coperm]

    # dot each atom's R with Q to produce its phase correction
    return np.exp(2j * np.pi * image_carts @ qpoint_cart)

#---------------------------------------------------------------

class Supercell:
    def __init__(self, matrix):
        """
        :param matrix: Shape ``(3, 3)``, integer.
        Integer matrix satisfying
        ``matrix @ prim_lattice_matrix == super_lattice_matrix``
        where the lattice matrices are understood to store a lattice primitive
        translation in each row.
        """
        if isinstance(matrix, Supercell):
            self.matrix = matrix.matrix
            self.repeats = matrix.repeats
            self.t_repeats = matrix.t_repeats
        else:
            matrix = np.array(matrix, dtype=int)
            assert matrix.shape == (3, 3)
            self.matrix = matrix
            self.repeats = find_repeats(matrix)
            self.t_repeats = find_repeats(matrix.T)

    def translation_pfracs(self):
        """
        :return: Shape ``(quotient, 3)``, integral.

        Fractional coordinates of quotient-space translations,
        in units of the primitive cell lattice basis vectors.
        """
        return cartesian_product(*(np.arange(n) for n in self.repeats)).astype(float)

    def gpoint_sfracs(self):
        """
        :return: Shape ``(quotient, 3)``, integral.

        Fractional coordinates of quotient-space gpoints,
        in units of the supercell reciprocal lattice basis vectors.
        """
        return cartesian_product(*(np.arange(n) for n in self.t_repeats)).astype(float)

    def recip(self):
        """
        Get a supercell that describes the reciprocal primitive lattice in terms
        of the reciprocal supercell lattice.
        """
        return Supercell(self.matrix.T)

def find_repeats(supercell_matrix):
    """
    Get the number of distinct translations along each lattice primitive
    translation. (it's the diagonal of the row-based HNF of the matrix)

    :param supercell_matrix: Shape ``(3, 3)``, integer.
    Integer matrix satisfying
    ``matrix @ prim_lattice_matrix == super_lattice_matrix``
    where the lattice matrices are understood to store a lattice primitive
    translation in each row.

    :return:
    """
    from abelian import hermite_normal_form
    from sympy import Matrix

    expected_volume = abs(round(np.linalg.det(supercell_matrix)))

    supercell_matrix = Matrix(supercell_matrix) # to sympy

    # abelian.hermite_normal_form is column-based, so give it the transpose
    hnf = hermite_normal_form(supercell_matrix.T)[1].T
    hnf = np.array(hnf).astype(int) # to numpy

    assert round(np.linalg.det(hnf)) == expected_volume
    return np.diag(hnf)
