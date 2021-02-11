import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

from .. import maths


def a2z(a):
    """Converts a to z.

    Parameters
    ----------
    a : float
        The scale factor.
    """
    return 1./a - 1.


def z2a(z):
    """Converts z to a.

    Parameters
    ----------
    z : float
        Redshift.
    """
    return 1./(1.+z)


def _get_ez(z, omega_m, omega_l, omega_r):
    """Returns the total amount of matter + dark energy at z.

    Parameters
    ----------
    z : float
        Redshift.
    omega_m : float
        Present matter density.
    omega_l : float
        Present dark energy density.
    omega_r : float
        Present radiation density.
    """
    return np.sqrt(omega_r*(1.+z)**4 + omega_m*(1.+z)**3 + omega_l)


def get_Hz(z, omega_m, omega_l, omega_r, h0):
    """Returns the hubble expansion at z.

    Parameters
    ----------
    z : float
        Redshift.
    omega_m : float
        Present matter density.
    omega_l : float
        Present dark energy density.
    omega_r : float
        Present radiation density.
    h0 : float
        Hubble constant.
    """
    return (100.*h0)*_get_ez(z, omega_m, omega_l, omega_r)


def _get_Dz_integrand(z, omega_m, omega_l, omega_r, h0):
    """Integral for the linear growth function.

    Parameters
    ----------
    z : float
        Redshift.
    omega_m : float
        Present matter density.
    omega_l : float
        Present dark energy density.
    omega_r : float
        Present radiation density.
    h0 : float
        Hubble constant.
    """
    return (1.+z)/(get_Hz(z, omega_m, omega_l, omega_r, h0)**3.)


def _get_Dz_val(z, omega_m, omega_l, omega_r, h0):
    """Normalised linear growth function for one z value.

    Parameters
    ----------
    z : float
        Redshift.
    omega_m : float
        Present matter density.
    omega_l : float
        Present dark energy density.
    omega_r : float
        Present radiation density.
    h0 : float
        Hubble constant.
    """
    Dz_0, err = integrate.quad(_get_Dz_integrand, 0., np.infty, args=(omega_m, omega_l, omega_r, h0))
    Dz_0 *= get_Hz(0., omega_m, omega_l, omega_r, h0)
    Dz, err = integrate.quad(_get_Dz_integrand, z, np.infty, args=(omega_m, omega_l, omega_r, h0))
    Dz *= get_Hz(z, omega_m, omega_l, omega_r, h0)
    return Dz/Dz_0


def get_Dz(z, omega_m, omega_l, omega_r, h0):
    """Normalised linear growth function for z float or array.

    Parameters
    ----------
    z : float/array
        Redshifts.
    omega_m : float
        Present matter density.
    omega_l : float
        Present dark energy density.
    omega_r : float
        Present radiation density.
    h0 : float
        Hubble constant.
    """
    if np.isscalar(z) == False:
        Dz = np.array([_get_Dz_val(z_val, omega_m, omega_l, omega_r, h0) for z_val in z])
    else:
        Dz = _get_Dz_val(z, omega_m, omega_l, omega_r, h0)
    return Dz


def get_D2z(Dz, prefix=-3./7.):
    """Gets the second-order growth function D2 at redshift z using the approximation
    D2 = -(3/7)D^2.

    Parameters
    ----------
    z : float
        Redshift.
    interp : bool
        If true value is interpolated from pre-tabulated values of Dz, if not this
        is calculated exactly.
    """
    return prefix*(Dz**2.)


def _get_r_integrand(z, omega_m, omega_l, omega_r):
    """Comoving distance integral.

    Parameters
    ----------
    z : float
        Redshift.
    omega_m : float
        Present matter density.
    omega_l : float
        Present dark energy density.
    omega_r : float
        Present radiation density.
    """
    return 1./_get_ez(z, omega_m , omega_l, omega_r)


def _get_r_val(z, omega_m, omega_l, omega_r):
    """Returns the comoving distance at for one z value.

    Parameters
    ----------
    z : float
        Redshift.
    omega_m : float
        Present matter density.
    omega_l : float
        Present dark energy density.
    omega_r : float
        Present radiation density.
    """
    r, err = integrate.quad(_get_r_integrand, 0., z, args=(omega_m, omega_l, omega_r))
    r *= 3000.
    return r


def get_r(z, omega_m, omega_l, omega_r):
    """Returns the comoving distance at z.

    Parameters
    ----------
    z : float
        Redshift.
    omega_m : float
        Present matter density.
    omega_l : float
        Present dark energy density.
    omega_r : float
        Present radiation density.
    """
    if np.isscalar(z) == False:
        r = np.array([_get_r_val(z_val, omega_m, omega_l, omega_r) for z_val in z])
    else:
        r = _get_r_val(z, omega_m, omega_l, omega_r)
    return r


def get_omega_m_z(z, omega_m, omega_l, omega_r):
    """Returns the matter density at z.

    Parameters
    ----------
    z : float
        Redshift.
    omega_m : float
        Present matter density.
    omega_l : float
        Present dark energy density.
    omega_r : float
        Present radiation density.
    """
    return (omega_m*(1.+z)**3.)/_get_ez(z, omega_m, omega_l, omega_r)**2.


def get_fz(z, omega_m, omega_l, omega_r, alpha=0.55):
    """Returns the approximation of the growth function dlnD/dlna.

    Parameters
    ----------
    z : float
        Redshift.
    omega_m : float
        Present matter density.
    omega_l : float
        Present dark energy density.
    omega_r : float
        Present radiation density.
    alpha : float
        default = 0.6.
    """
    return get_omega_m_z(z, omega_m, omega_l, omega_r)**alpha


def get_f2z(z, omega_m, omega_l, omega_r, alpha=6./11.):
    """Returns the approximation of the second-order growth function dlnD/dlna.

    Parameters
    ----------
    z : float
        Redshift.
    omega_m : float
        Present matter density.
    omega_l : float
        Present dark energy density.
    omega_r : float
        Present radiation density.
    alpha : float
        default = 6/11.
    """
    return 2.*get_omega_m_z(z, omega_m, omega_l, omega_r)**alpha


def get_fz_numerical(z, Dz, **kwargs):
    """Calculates f(z) numerically from linear growth function.

    Parameters
    ----------
    z : array
        Tabulated redshift.
    Dz : array
        Tabulated linear growth function.

    Returns
    -------
    fz : array
        Returns numerical fz.
    """
    a = z2a(z)
    loga = np.log(a)
    logD = np.log(Dz)
    fz = maths.numerical_differentiate(loga, logD, **kwargs)
    return fz


def get_sigma_8(kh, pk):
    """Calulates sigma_8 from an input power spectrum.

    Parameters
    ----------
    kh, pk : array_like
        The linear power spectrum and corresponding k value.

    Returns
    -------
    sigma_8 : float
        The calculated value of sigma_8.
    """
    w = 3.*(np.sin(kh*8.)-(kh*8.*np.cos(kh*8.)))/(kh*8.)**3.
    integrand = kh*kh*pk*w*w
    sigma_8_2 = (1./(2.*np.pi**2.))*integrate.simps(integrand, kh)
    sigma_8 = sigma_8_2**0.5
    return sigma_8


def get_z_array(zmin, zmax, zbin_num, zbin_mode='linear'):
    """Returns a redshift array either with linear or log binning.

    Parameters
    ----------
    zmin : float
        Redshift minimum.
    zmax : float
        Redshift maximum.
    zbin_num : int
        Size of array.
    zbin_mode : str
        Linear or log binning.
    """
    if zbin_mode == 'linear':
        return np.linspace(zmin, zmax, zbin_num)
    elif zbin_mode == 'log':
        return np.logspace(np.log10(zmin+1.), np.log10(zmax+1.), zbin_num) - 1.
    else:
        print(zbinning_mode, " is not supported. Use either 'linear' or 'log'.")
