import numpy as np
from scipy import special, optimize


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


def get_qln(l, n):
    """ Returns the position of the nth root of the spherical bessel function with
    mode l.

    Parameters
    ----------
    l : int
        Spherical bessel mode.
    n : int
        nth root.
    """
    x_min = special.jn_zeros(l, n)[n-1]
    x_max = special.jn_zeros(l+1, n)[n-1]
    root = optimize.brentq(get_jl, x_min, x_max, args=(l))
    return root


def get_kln(l, n, Rmax):
    """ Gives the K-mode for each spherical bessel function with modes l and n.

    Parameters
    ----------
    l : int
        Spherical bessel mode.
    n : int
        nth root.
    Rmax : float
        The maximum radius probed by the spherical bessel function.
    """
    return get_qln(l, n)/Rmax


def get_lmax(Rmax, kmax):
    """ Returns a rough estimate of the maximum l-mode needed to probe all k-modes.

    Parameters
    ----------
    Rmax : float
        The maximum radius probed by the spherical bessel function.
    kmax : float
        The maximum K-mode that you want to probe.
    """
    return Rmax*kmax


def get_nmax(Rmax, kmax):
    """ Returns a rough estimate of the maximum n-mode needed to probe all k-modes.

    Parameters
    ----------
    Rmax : float
        The maximum radius probed by the spherical bessel function.
    kmax : float
        The maximum K-mode that you want to probe.
    """
    return Rmax*kmax/np.pi
