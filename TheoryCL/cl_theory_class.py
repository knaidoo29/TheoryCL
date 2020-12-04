import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import healpy as hp
from . import progress
from . import spherical_bessel as sb
from . import linear_growth_class as lgc


class SourceCL(lgc.CosmoLinearGrowth):

    """Function for calculating the auto- and cross-angular power spectra for
    sources of the galaxy distribution, ISW and lensing. This class inherits
    the functionality of the CosmoLinearGrowth class.

    Usage
    -----

    import pyGenISW

    # Initialises SourceCL by passing the CosmoLinearGrowth class into it.
    SCL = pyGenISW.SourceCL(pyGenISW.CosmoLinearGrowth())

    # Set cosmology
    SCL.cosmo(omega_m=0.25, omega_l=0.75, h0=0.7, omega_b=0.044, ns=0.95,
              As=2.45e-9, sigma8=0.8)

    # Create table of values for later interpolation.
    SCL.calc_table(zmin=0., zmax=10., zbin_num=10000, zbin_mode='log')

    # Calculate linear power spectrum from CAMB and generates a callable interpolator.
    SCL.calc_pk()

    """

    def __init__(self, CosmoLinearGrowth):
        """Initialises the class.

        Parameters
        ----------
        CosmoLinearGrowth : class
            Parent class for calculating Cosmological linear growth functions.
        """
        super(SourceCL, self).__init__()
        self.switch2limber = None
        self.C = 3e8
        self.MPC = 3.0856e22
        self.sources_dict = {}
        self.source_count = 0
        # to aid in cycling through sources
        self.which_source = 1
        # integration values
        self.Lmax = None
        self.k_int = None
        self.r_int = None


    def clean_sources(self):
        """Cleans and resets the sources variables."""
        self.sources_dict = {}
        self.source_count = 0
        # to aid in cycling through sources
        self.which_source = 1


    def _W_ISW(self, r, k):
        zmin = self.sources_dict['source '+str(self.which_source)]['zmin']
        zmax = self.sources_dict['source '+str(self.which_source)]['zmax']
        rmin = self.get_rz(zmin)
        rmax = self.get_rz(zmax)
        w_isw = (3.*(100.*self.h0)**2. * self.omega_m)/((self.C**3.)*(k**2.))
        w_isw *= 1.-self.get_fr(r)
        w_isw *= self.get_Hr(r)
        w_isw *= np.heaviside(r-rmin, 1.)-np.heaviside(r-rmax, 1.)
        # units
        w_isw *= 1e9/(self.MPC*(self.h0**2.))
        return w_isw


    def _q_tophat(self, r):
        zmin = self.sources_dict['source '+str(self.which_source)]['zmin']
        zmax = self.sources_dict['source '+str(self.which_source)]['zmax']
        rmin = self.get_rz(zmin)
        rmax = self.get_rz(zmax)
        return 1.-(3./2.)*r*(rmax**2. - rmin**2.)/(rmax**3. - rmin**3.)


    def _W_lens(self, r, k):
        zmax = self.sources_dict['source '+str(self.which_source)]['zmax']
        q_function = self.sources_dict['source '+str(self.which_source)]['q function']
        rmax = self.get_rz(zmax)
        w_lens= (3.*(100.*self.h0)**2. * self.omega_m)/(self.C**2.)
        w_lens *= (1.+self.get_zr(r))/(self.get_Dr(r)*r*k**2.)
        w_lens *= q_function(r)
        # units
        w_lens *= 1e6/(self.MPC*self.h0)
        return w_lens


    def _W_gal(self, r):
        b_function = self.sources_dict['source '+str(self.which_source)]['b function']
        Theta_function = self.sources_dict['source '+str(self.which_source)]['Theta function']
        w_gal = b_function(r)*Theta_function(r)
        w_gal *= self.h0/self.MPC
        return w_gal


    def _add2sources_dict(self, source_dict):
        self.source_count += 1
        self.sources_dict['source '+str(self.source_count)] = source_dict
        sources_X = []
        sources_Y = []
        for i in range(1, self.source_count+1):
            for j in range(i, self.source_count+1):
                sources_X.append(i)
                sources_Y.append(j)
        self.sources_X = sources_X
        self.sources_Y = sources_Y


    def set_source_ISW(self, zmin, zmax):
        source_dict = {
            'source type': 'ISW',
            'zmin': zmin,
            'zmax': zmax
        }
        self._add2sources_dict(source_dict)


    def set_source_lens_tophat(self, zmin, zmax):
        source_dict = {
            'source type': 'lens',
            'zmin': zmin,
            'zmax': zmax,
            'q function': self._q_tophat
        }
        self._add2sources_dict(source_dict)

    def get_sample_Theta_z(self, z, nz):
        r = self.get_rz(z)
        Theta_z= (r**2.)*nz/integrate.simps((r**2.)*nz, r)
        return Theta_z

    def get_sample_q_z(self, z, nz):
        r = self.get_rz(z)
        Theta_z = self.get_sample_Theta_z(z, nz)
        condition = np.where(r != 0)[0]
        q_z = np.array([integrate.simps(((r[condition]-r_val)/r[condition])*Theta_z[condition], r[condition]) for r_val in self.r_int])
        return q_z

    def _q_interp(self, r):
        z = self.get_zr(r)
        interp_q_z = self.sources_dict['source '+str(self.which_source)]['q interp']
        return interp_q_z(z)

    def set_source_lens_sample(self, z, nz):
        q_z = self.get_sample_q_z(z, nz)
        interp_q_z = interp1d(self.get_zr(self.r_int), q_z, kind='nearest', fill_value=0., bounds_error=False)
        source_dict = {
            'source type': 'lens',
            'zmin': z.min(),
            'zmax': z.max(),
            'q function': self._q_interp,
            'q interp': interp_q_z
        }
        self._add2sources_dict(source_dict)

    def _b_const(self, r):
        b = self.sources_dict['source '+str(self.which_source)]['b']
        if np.isscalar(r) is True:
            return b
        else:
            return b*np.ones(len(r))

    def _Theta_tophat(self, r):
        zmin = self.sources_dict['source '+str(self.which_source)]['zmin']
        zmax = self.sources_dict['source '+str(self.which_source)]['zmax']
        rmin = self.get_rz(zmin)
        rmax = self.get_rz(zmax)
        if np.isscalar(r) is True:
            if r >= rmin and r <= rmax:
                return (3.*r**2.)/(rmax**3.-rmin**3.)
            else:
                return 0.
        else:
            Theta = np.zeros(len(r))
            condition = np.where((r >= rmin) & (r <= rmax))[0]
            Theta[condition] = 3.*(r[condition]**2.)/(rmax**3. - rmin**3.)
            return Theta

    def _Theta_sample(self, r):
        z = self.get_zr(r)
        interp_Theta_z = self.sources_dict['source '+str(self.which_source)]['Theta interp']
        return interp_Theta_z(z)

    def set_source_gal_tophat(self, zmin, zmax, b):
        source_dict = {
            'source type': 'gal',
            'zmin': zmin,
            'zmax': zmax,
            'b': b,
            'b function': self._b_const,
            'Theta function': self._Theta_tophat
        }
        self._add2sources_dict(source_dict)

    def set_source_gal_sample(self, z, nz, b=1., b_function=None):
        if b_function is None:
            b_function = self._b_const
        Theta_z = self.get_sample_Theta_z(z, nz)
        interp_Theta_z = interp1d(z, Theta_z, kind='nearest', fill_value=0., bounds_error=False)
        source_dict = {
            'source type': 'gal',
            'zmin': z.min(),
            'zmax': z.max(),
            'b': b,
            'b function': b_function,
            'Theta function': self._Theta_sample,
            'Theta interp': interp_Theta_z
        }
        self._add2sources_dict(source_dict)


    def setup(self, Lmax, zmin=0., zmax=5., rbin_num=1000, rbin_mode='linear',
              kmin=None, kmax=None, kbin_num=1000, kbin_mode='log',
              switch2limber=30, Tcmb=2.7255, renormalise=True):
        self.Lmax = Lmax
        self.switch2limber = switch2limber
        self.Tcmb = 2.7255
        # k binning definitions
        if kmin is None:
            kmin = self.kmin + 0.1*self.kmin#0.001*(self.kmax-self.kmin)
        elif kmin < self.kmin:
            print("kmin is", kmin, "<", self.kmin, "used by CAMB to calculate P(k)")
            print("reseting kmin to ", self.kmin)
            kmin = self.kmin
        if kmax is None:
            kmax = self.kmax - 0.1*self.kmin#0.001*(self.kmax-self.kmin)
        elif kmax > self.kmax:
            print("kmax is", kmax, ">", self.kmax, "used by CAMB to calculate P(k)")
            print("reseting kmax to ", self.kmax)
            kmax = self.kmax
        if kbin_mode == 'linear':
            self.k_int = np.linspace(kmin, kmax, kbin_num)
        elif kbin_mode == 'log':
            self.k_int = np.logspace(np.log10(kmin), np.log10(kmax), kbin_num)
        # r binning definitions
        if zmin == 0.:
            zmin = 0.001
        if rbin_mode == 'linear':
            self.r_int = np.linspace(self.get_rz(zmin), self.get_rz(zmax), rbin_num)
        elif rbin_mode == 'log':
            self.r_int = np.logspace(np.log10(self.get_rz(zmin)), np.log10(self.get_rz(zmax)), rbin_num)
        self.renormalise = renormalise


    def _get_W(self, r, k, l, source_X):
        self.which_source = source_X
        source_type = self.sources_dict['source '+str(source_X)]['source type']
        if source_type == 'ISW':
            return self._W_ISW(r, k)
        elif source_type == 'gal':
            return self._W_gal(r)
        elif source_type == 'lens':
            return self._W_lens(r, k)


    def _get_CL_XY_limber(self, l):
        # get interpolated values
        rz = self.r_int
        Dz = self.get_Dr(self.r_int)
        Hz = self.get_Hr(self.r_int)
        # get kl
        condition = np.where(rz != 0.)[0]
        kl = np.zeros(len(rz))
        kl[condition] = float(l + 1./2.)/rz[condition]
        condition = np.where((kl > self.k_int[0]) & (kl < self.k_int[-1]))[0]
        # cycle through sources
        Ws = []
        for i in range(1, self.source_count+1):
            source_X = i
            W_X = np.zeros(len(kl))
            W_X[condition] = self._get_W(rz[condition], kl[condition], l, source_X)
            Ws.append(W_X)
        # get interpolated power spectrum values
        pk = np.zeros(len(kl))
        pk[condition] = self.get_pk(kl[condition], renormalise=self.renormalise)
        integral_CL_XY_limber_base = (Dz**2.)*pk
        if self.r_int[0] == 0.:
            integral_CL_XY_limber_base[0] = 0.
            integral_CL_XY_limber_base[1:] /= (rz[1:]**2.)
        else:
            integral_CL_XY_limber_base /= (rz**2.)
        # get all Cls
        CLs_XY_limber = np.array([integrate.simps(integral_CL_XY_limber_base*Ws[self.sources_X[i]-1]*Ws[self.sources_Y[i]-1], self.r_int) for i in range(0, len(self.sources_X))])
        CLs_XY_limber *= (self.MPC/self.h0)**2.
        return CLs_XY_limber


    def _get_CL_XY(self, l):
        # get interpolated values
        rz = self.r_int
        Hz = self.get_Hr(self.r_int)
        Dz = self.get_Dr(self.r_int)
        # cycle through sources
        Ils = []
        #x = np.linspace(0., self.r_int.max()*self.k_int.max()+1., 1000)
        #jl = sb.get_jl(x, l)
        #jl_interpolator = interp1d(x, jl, kind='cubic', fill_value=0., bounds_error=True)
        for i in range(1, self.source_count+1):
            source_X = i
            Il_X = [integrate.simps(Dz*self._get_W(self.r_int, k_val, l, source_X)*sb.get_jl(k_val*rz, l), self.r_int) for k_val in self.k_int]
            #Il_X = [integrate.simps(Dz*self._get_W(self.r_int, k_val, l, source_X)*jl_interpolator(k_val*self.r_int), self.r_int) for k_val in self.k_int]
            Ils.append(Il_X)
        # get interpolated power spectrum values
        pk = self.get_pk(self.k_int, renormalise=self.renormalise)
        CLs_XY = np.array([(2./np.pi)*integrate.simps((self.k_int**2.)*pk*Ils[self.sources_X[i]-1]*Ils[self.sources_Y[i]-1], self.k_int) for i in range(0, len(self.sources_X))])
        CLs_XY *= (self.MPC/self.h0)**2.
        return CLs_XY


    def get_CL(self):
        self.L_full = np.arange(self.switch2limber+1)[2:]
        self.L = np.arange(self.Lmax+1)[2:]
        self.CLs_full = []
        self.CLs_approx = []
        for i in range(0, len(self.L)):
            _CLs_limber = self._get_CL_XY_limber(self.L[i])
            self.CLs_approx.append(_CLs_limber)
            if self.L[i] <= self.switch2limber:
                _CLs = self._get_CL_XY(self.L[i])
                self.CLs_full.append(_CLs)
                progress.progress_bar(i, self.switch2limber -2 + 1, explanation='Calculating Cls (Full)    L = '+str(self.L[i])+' / '+str(self.switch2limber))
            else:
                progress.progress_bar(i - self.switch2limber, self.Lmax - self.switch2limber - 1, explanation='Calculating Cls (Approx.) L = '+str(self.L[i])+' / '+str(self.Lmax))#, num_refresh=len(self.L)-self.switch2limber-2)
        self.CLs_full = np.array(self.CLs_full)
        self.CLs_approx = np.array(self.CLs_approx)
        condition = np.where(self.L_full >= self.switch2limber-5)[0]
        self.CLs = np.copy(self.CLs_approx)
        for i in range(0, len(self.sources_X)):
            self.CLs[:len(self.CLs_full[:, i]), i] = self.CLs_full[:, i]
            self.CLs[len(self.CLs_full[:, i]):, i] = np.mean(self.CLs_full[condition, i]/self.CLs_approx[condition, i]) * self.CLs_approx[len(self.CLs_full[:, i]):, i]


    def get_DL(self):
        factor_full = self.L_full*(self.L_full + 1.)/(2.*np.pi)
        factor = self.L*(self.L + 1.)/(2.*np.pi)
        self.DLs_full = np.copy(self.CLs_full)
        self.DLs_approx = np.copy(self.CLs_approx)
        self.DLs = np.copy(self.CLs)
        for i in range(0, len(self.sources_X)):
            self.DLs_full[:, i] *= factor_full
            self.DLs_approx[:, i] *= factor
            self.DLs[:, i] *= factor

    def prep4heal(self):
        CLs4heal = []
        for i in range(0, len(self.sources_X)):
            CLs4heal.append(np.concatenate([np.zeros(2), self.CLs[:, i]]))
        self.CLs4heal = np.array(CLs4heal)

    def simulate(self, lmax, nside):
        self.sim_nside = nside
        self.sim_lmax = lmax
        self.sim_alms = hp.synalm(self.CLs4heal, lmax=self.sim_lmax, new=False)
        self.sim_maps =  [hp.alm2map(self.sim_alms[i], self.sim_nside, lmax=self.sim_lmax) for i in range(0, len(self.sim_alms))]

    def clean(self):
        self.__init__()
