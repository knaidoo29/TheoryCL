from scipy import special


def get_jl(x, l):
    """ Returns the spherical bessel function at x for mode l.

    Parameters
    ----------
    x : float
        x coordinate.
    l : int
        Spherical bessel mode.
    """
    return special.spherical_jn(l, x)
