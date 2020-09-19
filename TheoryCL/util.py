from scipy.integrate import trapz, cumtrapz


def integrate(x, y, total=False, equal_spacing=False):
    """Finds the integral of y wrt to x using the trapezium rule.

    Parameters:
    -----------
    x : array
        Axis over the which the integration is performed.
    y : array
        The function f(x) for which the integration is performed.
    total : bool, optional
        If True will only output the integral over the full range of x.
    equal_spacing : bool, optional
        If true then the spacings between x is assumed to be equal, and the
        integral is performed as a sum. if false then the spacings can change and
        is performed using the trapezium rule via a fortran sub-routine.

    Internal:
    ---------
    _dx : array
        The difference between each ascending x value of an array. Only valid for
        equal_spacing=True.

    Returns:
    --------
    y_integrand : array/float
        Integral of the function y.
    """
    if equal_spacing is False:
        if total is True:
            y_integrand = trapz(y, x=x)
        else:
            y_integrand = cumtrapz(y, x=x, initial=0)
    else:
        _dx = x[1] - x[0]
        if total is True:
            y_integrand = trapz(y, dx=_dx)
        else:
            y_integrand = cumtrapz(y, dx=_dx, initial=0)
    return y_integrand
