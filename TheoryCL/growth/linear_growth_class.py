import numpy as np
import camb as cb
from scipy import integrate
from scipy.interpolate import interp1d
from . import linear_growth_functions as lgf


class CosmoLinearGrowth:

    """ A class for computing, storing and interpolating cosmological linear growth functions:

    - rz: comoving distance.
    - Hz: Hubble parameter.
    - Dz: Linear growth function, normalised to 1 at z=0.
    - fz: dlnD/dlna.
    - pk: linear power spectrum

    Usage
    -----

    import pyGenISW

    # Initialises the class, which will call the function __init__ in the class.
    CLG = pyGenISW.CosmoLinearGrowth()

    # Set cosmology
    CLG.cosmo(omega_m=0.25, omega_l=0.75, h0=0.7, omega_b=0.044, ns=0.95,
              As=2.45e-9, sigma8=0.8)

    # Create table of values.
    CLG.calc_table(zmin=0., zmax=10., zbin_num=10000, zbin_mode='linear')

    # Get values by interpolation
    z = np.linspace(0.1, 0.5, 100)
    rz = CLG.get_rz(z)
    Hz = CLG.get_Hz(z)
    Dz = CLG.get_Dz(z)
    Fz = CLG.get_fz(z)

    # Calculate from integrals and formulas
    rz = CLG.get_rz(z, interp=False)
    Hz = CLG.get_Hz(z, interp=False)
    Dz = CLG.get_Dz(z, interp=False)
    Fz = CLG.get_fz(z, interp=False)

    # Pre-calculate linear power spectrum
    CLG.calc_pk()

    # Interpolate pk using camb interpolator
    k = np.linspace(0.001, 0.1, 100)
    pk = CLG.get_pk(k)

    # Renormalises to give the desired sigma8.
    pk = CLG.get_pk(k, renormalise=True)

    CLG.clean()
    """


    def __init__(self):
        """Initialises class variables."""
        # Cosmological constants
        self.omega_m = None
        self.omega_l = None
        self.h0 = None
        self.omega_b = None
        self.ns = None
        self.As = None
        self.sigma8 = None
        self.omega_r = None
        # z binnings for table
        self.zmin = None
        self.zmax = None
        self.zbin_num = None
        self.zbin_mode = None
        # table
        self.f1_alpha = None
        self.f2_alpha = None
        self.D2_const = None
        self.z_table = None
        self.rz_table = None
        self.Hz_table = None
        self.Dz_table = None
        self.fz_table = None
        self.D2z_table = None
        self.f2z_table = None
        # power spectrum terms
        self.z_pk = None
        self.kmin = None
        self.kmax = None
        self.kbin_num = None
        self.camb_kh = None
        self.camb_pk = None
        self.camb_sigma8 = None
        self.camb_pk_interpolator = None
        # Numerical variables
        self.num_sigma8 = None
        self.use_numerical = False
        self.use_numerical_pk = False
        self.interp_pk_log = None
        self.kh_table = None
        self.pk_table = None


    def cosmo(self, omega_m=0.25, omega_l=0.75, h0=0.7, omega_b=0.044, ns=0.95,
              As=2.45e-9, sigma8=0.8, omega_r=0.):
        """Sets cosmological parameters.

        Parameters
        ----------
        omega_m : float
            Matter density.
        omega_l : float
            Cosmological constant.
        h0 : float
            Hubble constant.
        omega_b : float
            Baryon density.
        ns : float
            Primordial scalar spectral index.
        As : float
            Amplitude of scalar fluctations.
        sigma8 : float
            Variance of density perturbations in spheres with radius 8 Mpc/h.
        omega_r : float
            Current radiation density.
        """
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.h0 = h0
        self.omega_b = omega_b
        self.ns = ns
        self.As = As
        self.sigma8 = sigma8
        self.omega_r = omega_r


    def calc_table(self, zmin=0., zmax=10., zbin_num=1000, zbin_mode='linear', alpha=5./9., alpha2=6./11., kind='cubic', use_f_numerical=False, D2_const=-3./7.):
        """Constructs table of cosmological linear functions to be interpolated for speed.

        Parameters
        ----------
        zmin : float
            Minimum redshift for tabulated values of the linear growth functions.
        zmax : float
            Maximum redshift for tabulated values of the linear growth functions.
        zbin_num : int
            Number of redshift values to compute the growth functions.
        zbin_mode : str
            Redshift binning, either linear or log of 1+z.
        alpha : float
            The power in the approximation to f(z) = Omega_m(z)**alpha
        beta : float
            The power in the approximation to f2(z) = Omega_m(z)**beta
        kind : str
            The kind of interpolation used by the created interpolation functions as function of z and r.
        use_f_numerical : bool
            If True will calculate f numerically.
        D2_const : float
            Constant infront of the D2 approximation, D2 = D2_const*(Dz)**2.
        """
        # store some variables for table generation
        self.zmin = zmin # minimum redshift for table
        self.zmax = zmax # maximum redshift for table
        self.zbin_num = zbin_num # size of array
        self.zbin_mode = zbin_mode # linear or log
        self.f1_alpha = alpha # for fz approximation
        self.f2_alpha = alpha2 # for f2z approximation
        self.D2_const = D2_const # for D2z approximation
        # construct z array
        self.z_table = lgf.get_z_array(self.zmin, self.zmax, self.zbin_num, self.zbin_mode)
        # constructs table of linear growth functions rz, Hz, Dz and fz
        self.rz_table = lgf.get_r(self.z_table, self.omega_m, self.omega_l, self.omega_r)
        self.Hz_table = lgf.get_Hz(self.z_table, self.omega_m, self.omega_l, self.omega_r, self.h0)
        self.Dz_table = lgf.get_Dz(self.z_table, self.omega_m, self.omega_l, self.omega_r, self.h0)
        self.D2z_table = lgf.get_D2z(self.Dz_table, prefix=self.D2_const)
        if use_f_numerical == False:
            self.fz_table = lgf.get_fz(self.z_table, self.omega_m, self.omega_l, self.omega_r, self.f1_alpha)
            self.f2z_table = lgf.get_f2z(self.z_table, self.omega_m, self.omega_l, self.omega_r, self.f1_alpha)
        else:
            self.fz_table = lgf.get_fz_numerical(self.z_table, self.Dz_table)
            self.f2z_table = lgf.get_fz_numerical(self.z_table, -self.D2z_table)
        # constructs callable interpolators for rz, Hz, Dz and fz
        self.rz_interpolator = interp1d(self.z_table, self.rz_table, kind=kind)
        self.Hz_interpolator = interp1d(self.z_table, self.Hz_table, kind=kind)
        self.Dz_interpolator = interp1d(self.z_table, self.Dz_table, kind=kind)
        self.fz_interpolator = interp1d(self.z_table, self.fz_table, kind=kind)
        self.D2z_interpolator = interp1d(self.z_table, self.D2z_table, kind=kind)
        self.f2z_interpolator = interp1d(self.z_table, self.f2z_table, kind=kind)
        # constructs callable interpolators for rz, Hz, Dz and fz as a function of r
        self.zr_interpolator = interp1d(self.rz_table, self.z_table, kind=kind)
        self.Hr_interpolator = interp1d(self.rz_table, self.Hz_table, kind=kind)
        self.Dr_interpolator = interp1d(self.rz_table, self.Dz_table, kind=kind)
        self.fr_interpolator = interp1d(self.rz_table, self.fz_table, kind=kind)
        self.D2r_interpolator = interp1d(self.rz_table, self.D2z_table, kind=kind)
        self.f2r_interpolator = interp1d(self.rz_table, self.f2z_table, kind=kind)


    def num_table(self, z_table, Dz_table, kind='cubic', zbin_factor=1000, zbin_mode='linear', D2_const=-3./7.):
        """Constructs table of cosmological linear functions from tabulated values.

        Parameters
        ----------
        z_table : array
            Tabulated redshift.
        Dz_tan;e : array
            Tabulated linear growth function.zbin_num : int
        kind : str
            The kind of interpolation used by the created interpolation functions as function of z and r.
        zbin_factor : int
            Integer factor to interpolate tabulated Dz for numerical calculation of fz.
        zbin_mode : str
            Redshift binning, either linear or log of 1+z.
        D2_const : float
            Constant infront of the D2 approximation, D2 = D2_const*(Dz)**2.
        """
        self.use_numerical = False
        self.zmin = z_table.min()
        self.zmax = z_table.max()
        self.zbin_num = len(z_table)
        self.zbin_mode = zbin_mode
        # construct z array
        self.z_table = z_table
        # construct linear functions, use LCDM for r and H
        self.rz_table = lgf.get_r(self.z_table, self.omega_m, self.omega_l, self.omega_r)
        self.Hz_table = lgf.get_Hz(self.z_table, self.omega_m, self.omega_l, self.omega_r, self.h0)
        # Numerical Dz
        self.D2_const = D2_const
        self.Dz_table = Dz_table
        self.D2z_table = lgf.get_D2z(self.Dz_table)
        # constructs callable interpolators for rz, Hz, Dz and fz
        self.rz_interpolator = interp1d(self.z_table, self.rz_table, kind=kind)
        self.Hz_interpolator = interp1d(self.z_table, self.Hz_table, kind=kind)
        self.Dz_interpolator = interp1d(self.z_table, self.Dz_table, kind=kind)
        self.D2z_interpolator = interp1d(self.z_table, self.D2z_table, kind=kind)
        # constructs callable interpolators for rz, Hz, Dz and fz as a function of r
        self.zr_interpolator = interp1d(self.rz_table, self.z_table, kind=kind)
        self.Hr_interpolator = interp1d(self.rz_table, self.Hz_table, kind=kind)
        self.Dr_interpolator = interp1d(self.rz_table, self.Dz_table, kind=kind)
        self.D2r_interpolator = interp1d(self.rz_table, self.D2z_table, kind=kind)
        # Calculating fz table
        z = lgf.get_z_array(self.zmin, self.zmax, int(self.zbin_num*zbin_factor), self.zbin_mode)
        fz = lgf.get_fz_numerical(z, self.Dz_interpolator(z))
        fz_interp = interp1d(z, fz, kind=kind)
        self.fz_table = fz_interp(self.z_table)
        self.fz_interpolator = interp1d(self.z_table, self.fz_table, kind=kind)
        self.fr_interpolator = interp1d(self.rz_table, self.fz_table, kind=kind)
        f2z = lgf.get_fz_numerical(z, -self.D2z_interpolator(z))
        f2z_interp = interp1d(z, f2z, kind=kind)
        self.f2z_table = f2z_interp(self.z_table)
        self.f2z_interpolator = interp1d(self.z_table, self.f2z_table, kind=kind)
        self.f2r_interpolator = interp1d(self.rz_table, self.f2z_table, kind=kind)


    def get_rz(self, z, interp=True):
        """Gets the comoving distance at redshift z.

        Parameters
        ----------
        z : float
            Redshift.
        interp : bool
            If true value is interpolated from pre-tabulated values, if not this
            is calculated exactly.
        """
        if interp == True:
            # Interpolate rz
            return self.rz_interpolator(z)
        else:
            # Calculate rz
            return lgf.get_r(z, self.omega_m, self.omega_l, self.omega_r)


    def get_zr(self, r):
        """Interpolates z from a given value of r.

        Parameters
        ----------
        r : float
            Comoving distance.
        """
        return self.zr_interpolator(r)


    def get_Hz(self, z, interp=True):
        """Gets the Hubble parameter at redshift z.

        Parameters
        ----------
        z : float
            Redshift.
        interp : bool
            If true value is interpolated from pre-tabulated values, if not this
            is calculated exactly.
        """
        if interp == True:
            # Interpolate Hz
            return self.Hz_interpolator(z)
        else:
            # Calculate Hz
            return lgf.get_Hz(z, self.omega_m, self.omega_l, self.omega_r, self.h0)


    def get_Hr(self, r):
        """Interpolates H from a given value of r.

        Parameters
        ----------
        r : float
            Comoving distance.
        """
        return self.Hr_interpolator(r)


    def get_Dz(self, z, interp=True):
        """Gets the linear growth function D at redshift z.

        Parameters
        ----------
        z : float
            Redshift.
        interp : bool
            If true value is interpolated from pre-tabulated values, if not this
            is calculated exactly.
        """
        if interp == True or self.use_numerical == True:
            # Interpolate Dz
            return self.Dz_interpolator(z)
        else:
            # Calculate Dz
            return lgf.get_Dz(z, self.omega_m, self.omega_l, self.omega_r, self.h0)


    def get_Dr(self, r):
        """Interpolates D from a given value of r.

        Parameters
        ----------
        r : float
            Comoving distance.
        """
        return self.Dr_interpolator(r)


    def get_D2z(self, z, interp=True):
        """Gets the linear growth function D2 at redshift z.

        Parameters
        ----------
        z : float
            Redshift.
        interp : bool
            If true value is interpolated from pre-tabulated values, if not this
            is calculated exactly.
        """
        if interp == True or self.use_numerical == True:
            # Interpolate D2z
            return self.D2z_interpolator(z)
        else:
            # Calculate D2z
            return lgf.get_D2z(lgf.get_Dz(z, self.omega_m, self.omega_l, self.omega_r, self.h0))


    def get_D2r(self, r):
        """Interpolates D2 from a given value of r.

        Parameters
        ----------
        r : float
            Comoving distance.
        """
        return self.D2r_interpolator(r)


    def _check_f1_alpha(self, alpha):
        """Checks alpha is assigned a value.

        Parameters
        ----------
        alpha : float
            The power in the approximation to f(z) = Omega_m(z)**alpha
        """
        if alpha is None:
            if self.f1_alpha is None:
                self.f1_alpha = 5./9.
        else:
            self.f1_alpha = alpha


    def get_fz(self, z, alpha=None, interp=True):
        """Gets the derivative of the linear growth function f at redshift z.

        Parameters
        ----------
        z : float
            Redshift.
        alpha : float
            The power in the approximation to f(z) = Omega_m(z)**alpha
        interp : bool
            If true value is interpolated from pre-tabulated values, if not this
            is calculated exactly.
        """
        self._check_f1_alpha(alpha)
        if interp == True or self.use_numerical == True:
            # Interpolate fz
            return self.fz_interpolator(z)
        else:
            # Calculate fz
            return lgf.get_fz(z, self.omega_m, self.omega_l, self.omega_r, self.f1_alpha)


    def get_fr(self, r):
        """Interpolates f from a given value of r.

        Parameters
        ----------
        r : float
            Comoving distance.
        """
        return self.fr_interpolator(r)


    def _check_f2_alpha(self, alpha):
        """Checks alpha is assigned a value.

        Parameters
        ----------
        alpha : float
            The power in the approximation to f2(z) = Omega_m(z)**alpha
        """
        if alpha is None:
            if self.f2_alpha is None:
                self.f2_alpha = 6./11.
        else:
            self.f2_alpha = alpha


    def get_f2z(self, z, alpha=None, interp=True):
        """Gets the derivative of the linear growth function f at redshift z.

        Parameters
        ----------
        z : float
            Redshift.
        alpha : float
            The power in the approximation to f2(z) = Omega_m(z)**alpha
        interp : bool
            If true value is interpolated from pre-tabulated values, if not this
            is calculated exactly.
        """
        self._check_f2_alpha(alpha)
        if interp == True or self.use_numerical == True:
            # Interpolate fz
            return self.f2z_interpolator(z)
        else:
            # Calculate fz
            return lgf.get_f2z(z, self.omega_m, self.omega_l, self.omega_r, self.f2_alpha)


    def get_fr(self, r):
        """Interpolates f from a given value of r.

        Parameters
        ----------
        r : float
            Comoving distance.
        """
        return self.f2r_interpolator(r)


    def calc_pk(self, kmin=1e-4, kmax=1e1, kbin_num=1000, z=0., nonlinear=True):
        """Calculates the linear power spectrum from CAMB and creates callable interpolator.

        Parameters
        ----------
        kmin : float
            Minimum k for computed P(k).
        kmax : float
            Maximum k for computed P(k).
        kbin_num : int
            Number of k values for P(k) to be computed at.
        z : float
            Redshift of the computed P(k).
        """
        self.z_pk = z
        self.kmin = kmin
        self.kmax = kmax
        self.kbin_num = kbin_num
        # define parameters for CAMB to compute the power spectrum
        camb_params = cb.CAMBparams()
        camb_params.set_cosmology(H0=100.*self.h0, ombh2=self.omega_b*self.h0**2.,
                                  omch2=(self.omega_m-self.omega_b-self.omega_r)*self.h0**2., mnu=0., omk=0.)
        camb_params.InitPower.set_params(As=self.As, ns=self.ns, r=0)
        camb_params.set_for_lmax(2500, lens_potential_accuracy=0)
        camb_params.set_matter_power(redshifts=[self.z_pk], kmax=10.*self.kmax)
        if nonlinear == False:
            camb_params.NonLinear = cb.model.NonLinear_none
        else:
            camb_params.NonLinear = cb.model.NonLinear_both
        # calculate power spectrum
        camb_results = cb.get_results(camb_params)
        self.camb_kh, _z, pk = camb_results.get_matter_power_spectrum(minkh=self.kmin, maxkh=self.kmax, npoints=self.kbin_num)
        self.camb_pk = pk.flatten()
        self.camb_sigma8 = camb_results.get_sigma8_0()
        # define CAMB Pk interpolator
        self.camb_pk_interpolator = camb_results.get_matter_power_interpolator(nonlinear=False)


    def num_pk(self, kh, pk, interp_log=True, kind='cubic'):
        """Add numerical pk and creates callable interpolator.

        Parameters
        ----------
        kh : float
            Tabulated k.
        pk : float
            Tabulated P(k).
        interp_log : bool
            Interpolate in log space.
        """
        self.kh_table = kh
        self.pk_table = pk
        self.kmin = kh.min()
        self.kmax = kh.max()
        self.kbin_num = len(kh)
        self.use_numerical_pk = True
        self.interp_pk_log = interp_log
        self.num_sigma8 = lgf.get_sigma_8(kh, pk)
        if self.interp_pk_log == False:
            self.numerical_pk_interpolator = interp1d(self.kh_table, self.pk_table, kind=kind)
        else:
            self.numerical_pk_interpolator = interp1d(np.log10(self.kh_table), np.log10(self.pk_table), kind=kind)


    def get_pk(self, k, renormalise=False):
        """Interpolates the linear power spectra computed from CAMB.

        Parameters
        ----------
        k : float
            Frequency of power spectrum modes.
        renormalises : bool
            Renormalises sigma8 to the desired value.
        """
        if renormalise is True:
            if self.use_numerical_pk == True:
                if self.interp_pk_log == True:
                    return 10.**self.numerical_pk_interpolator(np.log10(k)) * ((self.sigma8/self.num_sigma8)**2.)
                else:
                    return self.numerical_pk_interpolator(k) * ((self.sigma8/self.num_sigma8)**2.)
            else:
                return self.camb_pk_interpolator.P(self.z_pk, k) * ((self.sigma8/self.camb_sigma8)**2.)
        else:
            if self.use_numerical_pk == True:
                if self.interp_pk_log == True:
                    return 10.**self.numerical_pk_interpolator(np.log10(k))
                else:
                    return self.numerical_pk_interpolator(k)
            else:
                return self.camb_pk_interpolator.P(self.z_pk, k)


    def clean(self):
        """Cleans and reassigns class functions."""
        self.__init__()
