import numpy as np
from scipy.interpolate import interp1d


def numerical_differentiate(x, f, equal_spacing=False, interpgrid=1000, kind='cubic'):
    """For unequally spaced data we interpolate onto an equal spaced 1d grid which
    we ten use the symmetric two-point derivative and the non-symmetric three point
    derivative estimator.

    Parameters
    ----------
    x : array
        X-axis.
    f : array
        Function values at x.
    equal_spacing : bool, optional
        Automatically assumes data is not equally spaced and will interpolate from it.
    interp1dgrid : int, optional
        Grid spacing for the interpolation grid, if equal spacing is False.
    kind : str, optional
        Interpolation kind.

    Returns
    -------
    df : array
        Numerical differentiation values for f evaluated at points x.

    Notes
    -----
    For non-boundary values:

    df   f(x + dx) - f(x - dx)
    -- = ---------------------
    dx            2dx

    For boundary values:

    df   - f(x + 2dx) + 4f(x + dx) - 3f(x)
    -- = ---------------------------------
    dx                  2dx
    """
    if equal_spacing == False:
        interpf = interp1d(x, f, kind=kind)
        x_equal = np.linspace(x.min(), x.max(), interpgrid)
        f_equal = interpf(x_equal)
    else:
        x_equal = np.copy(x)
        f_equal = np.copy(f)
    dx = x_equal[1] - x_equal[0]
    df_equal = np.zeros(len(x_equal))
    # boundary differentials
    df_equal[0] = (-f_equal[2] + 4*f_equal[1] - 3.*f_equal[0])/(2.*dx)
    df_equal[-1] = (f_equal[-3] - 4*f_equal[-2] + 3.*f_equal[-1])/(2.*dx)
    # non-boundary differentials
    df_equal[1:-1] = (f_equal[2:] - f_equal[:-2])/(2.*dx)
    if equal_spacing == False:
        interpdf = interp1d(x_equal, df_equal, kind=kind)
        df = interpdf(x)
    else:
        df = np.copy(df_equal)
    return df
