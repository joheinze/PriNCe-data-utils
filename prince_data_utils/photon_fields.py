from os.path import join
import numpy as np
from scipy.interpolate import interp1d, interp2d
import scipy.constants as sconst
from astropy import units as u
from prince_data_utils import resource_path


class EBLPhotonField(object):
    pass


class Francescini2008(EBLPhotonField):
    """1) Franceschini et al. at z=0 (for plotting)

    Ref.:
        A. Franceschini et al., Astron. Astrphys. 487, 837 (2008) [arXiv:0805.1841]
    """

    def __init__(self):
        # Load
        # Franceshini z= 0
        # Energy values
        self.energy = 10**np.vstack([
            [-11.835, -11.604, -11.45, -11.303, -11.136, -11.052, -10.947,
             -10.86, -10.78, -10.751, -10.684, -10.508, -10.383, -10.303, -10.206,
             -10.082, -9.9846, -9.8598, -9.8085, -9.7314, -9.6688, -9.5597,
             -9.4713, -9.3792, -9.1359, -9.01936, -8.90553, -8.8085, -8.6459,
             -8.5075, -8.2065],
            [-11.756, -11.525, -11.37, -11.224, -11.057, -10.972, -10.868,
             -10.78, -10.701, -10.671, -10.604, -10.428, -10.303, -10.224,
             -10.127, -10.002, -9.9055, -9.7804, -9.7293, -9.6523, -9.5897,
             -9.4795, -9.3921, -9.3, -9.0567, -8.94805, -8.8262, -8.7293, -8.5666,
             -8.4283, -8.1273],
            [-11.689, -11.458, -11.303, -11.157, -10.99, -10.906, -10.801,
             -10.714, -10.634, -10.604, -10.537, -10.361, -10.236, -10.157,
             -10.06, -9.9355, -9.8386, -9.7135, -9.6623, -9.5854, -9.5227,
             -9.4136, -9.3251, -9.2331, -8.99, -8.8732, -8.7595, -8.6623, -8.4996,
             -8.3614, -8.0604],
            [-11.631, -11.4, -11.245, -11.099, -10.932, -10.847, -10.743,
             -10.656, -10.576, -10.546, -10.48, -10.303, -10.178, -10.099,
             -10.002, -9.8775, -9.7804, -9.6556, -9.6045, -9.5274, -9.4647,
             -9.3556, -9.2672, -9.1751, -8.93181, -8.8153, -8.7014, -8.6045,
             -8.4417, -8.3034, -8.0024],
            [-11.58, -11.349, -11.194, -11.048, -10.881, -10.796, -10.692,
             -10.604, -10.525, -10.495, -10.428, -10.252, -10.127, -10.048,
             -9.9512, -9.8262, -9.7293, -9.6045, -9.5533, -9.4763, -9.4136,
             -9.3044, -9.216, -9.124, -8.8807, -8.7642, -8.6501, -8.5533, -8.3905,
             -8.2523, -7.951],
            [-11.534, -11.303, -11.148, -11.002, -10.835, -10.751, -10.646,
             -10.559, -10.48, -10.45, -10.383, -10.206, -10.082, -10.002, -9.9055,
             -9.7804, -9.6836, -9.5586, -9.5075, -9.4305, -9.3678, -9.2587,
             -9.1702, -9.07825, -8.8348, -8.7183, -8.6045, -8.5075, -8.3448,
             -8.2065, -7.906],
            [-11.492, -11.262, -11.107, -10.961, -10.793, -10.709, -10.604,
             -10.517, -10.438, -10.408, -10.341, -10.165, -10.04, -9.961, -9.8642,
             -9.7392, -9.6423, -9.5173, -9.4661, -9.3891, -9.3265, -9.2173,
             -9.1289, -9.04015, -8.7934, -8.666, -8.567, -8.4661, -8.3034,
             -8.1651, -7.864],
            [-11.455, -11.224, -11.069, -10.923, -10.756, -10.671, -10.567,
             -10.48, -10.4, -10.37, -10.303, -10.127, -10.002, -9.9234, -9.8262,
             -9.7014, -9.6045, -9.4795, -9.4283, -9.3513, -9.2887, -9.1795,
             -9.08155, -9.0024, -8.7557, -8.6281, -8.5292, -8.4283, -8.2656,
             -8.1273, -7.826],
            [-11.42, -11.189, -11.035, -10.888, -10.721, -10.637, -10.532,
             -10.445, -10.366, -10.336, -10.269, -10.093, -9.9678, -9.8884,
             -9.7916, -9.6666, -9.5698, -9.4448, -9.3936, -9.3166, -9.2539,
             -9.1448, -9.05637, -8.96428, -8.728, -8.6123, -8.4905, -8.3936,
             -8.2308, -8.0925, -7.792],
            [-11.388, -11.157, -11.002, -10.856, -10.689, -10.604, -10.5,
             -10.413, -10.333, -10.303, -10.236, -10.06, -9.9355, -9.8564,
             -9.7595, -9.6343, -9.5375, -9.4125, -9.3614, -9.2844, -9.2217,
             -9.1115, -9.02462, -8.93124, -8.6958, -8.5612, -8.4622, -8.3614,
             -8.1987, -8.0604, -7.759],
            [-11.358, -11.127, -10.972, -10.826, -10.659, -10.574, -10.47,
             -10.383, -10.303, -10.273, -10.206, -10.03, -9.9055, -9.8262,
             -9.7293, -9.6045, -9.5075, -9.3826, -9.3314, -9.2545, -9.1918,
             -9.08265, -8.98464, -8.90217, -8.6658, -8.55, -8.4283, -8.3314,
             -8.1687, -8.0304, -7.729]]
        ).astype('float64')

        # Photon density in GeV^-1 cm^-3
        self.ph_density = 10**np.vstack([
            [10.8983, 11.2507, 11.3374, 11.2727, 11.0669, 10.8765, 10.5735,
             10.2867, 10.0076, 9.8984, 9.7006, 9.09, 8.749, 8.51, 8.323, 8.037,
             7.7156, 7.4308, 7.3555, 7.3084, 7.2318, 7.1517, 7.1093, 7.0432,
             6.7459, 6.52736, 6.26753, 5.9855, 5.5049, 4.9675, 4.0405],
            [11.0137, 11.3608, 11.4485, 11.3963, 11.1873, 10.992, 10.6791,
             10.3763, 10.0903, 9.9776, 9.7748, 9.163, 8.834, 8.631, 8.439, 8.144,
             7.8425, 7.5544, 7.4783, 7.4283, 7.3557, 7.2755, 7.2231, 7.157,
             6.8067, 6.59305, 6.2912, 6.0113, 5.4616, 4.9883, 4.0893],
            [11.1579, 11.5029, 11.5872, 11.5271, 11.3097, 11.1016, 10.7693,
             10.4569, 10.1623, 10.0499, 9.8473, 9.246, 8.93, 8.727, 8.543, 8.2455,
             7.9236, 7.6365, 7.5593, 7.5104, 7.4397, 7.3576, 7.3031, 7.2211,
             6.728, 6.5782, 6.2915, 5.9903, 5.4066, 5.0094, 4.0894],
            [11.2638, 11.6084, 11.6862, 11.6137, 11.381, 11.1534, 10.8029,
             10.4822, 10.1875, 10.0757, 9.8752, 9.292, 8.98, 8.782, 8.592, 8.3105,
             7.9464, 7.6646, 7.5915, 7.5474, 7.4807, 7.4026, 7.3432, 7.2381,
             6.81081, 6.5533, 6.2604, 5.9295, 5.3917, 5.0674, 4.0634],
            [11.3555, 11.6896, 11.7486, 11.6526, 11.3921, 11.1388, 10.7737,
             10.4472, 10.1553, 10.0447, 9.845, 9.2874, 8.981, 8.79, 8.6032,
             8.2852, 7.8573, 7.6005, 7.5433, 7.5283, 7.4706, 7.4114, 7.336, 7.21,
             6.7657, 6.5012, 6.1991, 5.8293, 5.3795, 5.1043, 4.03],
            [11.4278, 11.7481, 11.7837, 11.6656, 11.3741, 11.1018, 10.7226,
             10.3985, 10.1089, 9.9991, 9.8, 9.264, 8.966, 8.761, 8.6105, 8.2044,
             7.7486, 7.5206, 7.4815, 7.4915, 7.4388, 7.3927, 7.2982, 7.15125,
             6.6968, 6.4353, 6.1015, 5.7085, 5.3818, 5.1275, 4.027],
            [11.4825, 11.7844, 11.7944, 11.6578, 11.3333, 11.0492, 10.6601,
             10.34, 10.0519, 9.9434, 9.7458, 9.2246, 8.939, 8.726, 8.6112, 8.0662,
             7.6693, 7.4593, 7.4301, 7.4541, 7.4195, 7.3493, 7.2329, 7.08815,
             6.6334, 6.345, 6.017, 5.6391, 5.4874, 5.1421, 4.087],
            [11.5206, 11.7931, 11.772, 11.6161, 11.2573, 10.9504, 10.5635,
             10.2499, 9.9606, 9.8573, 9.6666, 9.1556, 8.87, 8.7174, 8.5552,
             7.8934, 7.5915, 7.3955, 7.3733, 7.4073, 7.3817, 7.2885, 7.14755,
             7.01739, 6.5807, 6.2791, 5.9082, 5.6213, 5.4256, 5.1423, 4.165],
            [11.5356, 11.7784, 11.734, 11.5562, 11.1706, 10.8438, 10.4642,
             10.1558, 9.8661, 9.7688, 9.5891, 9.088, 8.7788, 8.7154, 8.4526,
             7.7596, 7.5638, 7.3668, 7.3456, 7.2686, 7.3429, 7.2238, 7.08837,
             6.94928, 6.54, 6.2673, 5.8195, 5.5936, 5.4378, 5.1265, 4.219],
            [11.5397, 11.7547, 11.6894, 11.4908, 11.0773, 10.7388, 10.3714,
             10.0612, 9.7739, 9.6819, 9.5099, 9.033, 8.7115, 8.6784, 8.3135,
             7.6533, 7.5435, 7.3455, 7.3224, 7.3544, 7.2997, 7.1555, 7.02362,
             6.87824, 6.4798, 6.1272, 5.7792, 5.5684, 5.4387, 5.0864, 4.229],
            [11.5283, 11.7172, 11.6349, 11.4144, 10.9674, 10.632, 10.2754,
             9.9551, 9.6801, 9.5896, 9.4224, 8.962, 8.6785, 8.5872, 8.1663,
             7.5835, 7.5085, 7.3046, 7.2784, 7.3065, 7.2408, 7.07765, 6.85464,
             6.79917, 6.4118, 6.084, 5.7183, 5.5404, 5.4247, 5.0124, 4.228]]
        ).astype('float64')

    def _construct_splines(self):

        z_e_splines = []
        z_n_splines = []
        for i in range(11):
            z_e_splines.append(
                interp1d(self.energy[i, :], np.arange(2, 33, dtype='float64'),
                         fill_value="extrapolate", bounds_error=False, kind='linear'))
            z_n_splines.append(
                interp1d(np.arange(2, 33, dtype='float64'), self.ph_density[i, :],
                         fill_value="extrapolate", bounds_error=False, kind='linear'))

        common_evec = np.logspace(
            np.log10(min(self.energy[:, 0])), np.log10(max(self.energy[:, -1])), 250)

        # 3D interpolation of grid idx function g(E, z)
        # Z values of the tables (spacing from paper)
        z_dist = np.arange(0, 2.1, 0.2)
        z_map = {}
        for zi, z in enumerate(z_dist):
            z_map[z] = zi

        ee, zzi = np.meshgrid(common_evec, z_dist)

        def ngamma(e, z):
            zi = z_map[z]
            if e < self.energy[zi][0] or e > self.energy[zi][-1]:
                return 0
            return z_n_splines[z_map[z]](z_e_splines[z_map[z]](e))

        vec_ng = np.vectorize(ngamma)

        ng_values = vec_ng(ee, zzi)

        self.spl2D = [
            ("Franceschini", interp2d(common_evec,
                                      z_dist, ng_values, fill_value=0., kind='linear'))
        ]


class Inoue2013(EBLPhotonField):
    """1) Inoue et al.

    Ref.:
        A. Inoue et al., The Astrophysical Journal, Volume 768, Number 2
    """

    def __init__(self):
        # Z values
        self.z_dist = np.array([
            0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3,
            0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1,
            3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5,
            4.6, 4.7, 4.8, 4.9, 5., 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9,
            6., 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7., 7.1, 7.2, 7.3,
            7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8., 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7,
            8.8, 8.9, 9., 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.
        ])
        # Energy in GeV
        self.energy = 10**np.array([
            -12., -11.9508, -11.8996, -11.8508, -11.8013, -11.7496, -11.699,
            -11.6498, -11.6003, -11.5498, -11.5003, -11.4498, -11.4001, -11.3497,
            -11.3002, -11.2503, -11.2, -11.15, -11.1002, -11.0501, -11.,
            -10.9508, -10.8996, -10.8508, -10.8013, -10.7496, -10.699, -10.6498,
            -10.6003, -10.5498, -10.5003, -10.4498, -10.4001, -10.3497, -10.3002,
            -10.2503, -10.2, -10.15, -10.1002, -10.0501, -10., -9.95078,
            -9.89963, -9.85078, -9.80134, -9.74958, -9.69897, -9.64975, -9.60033,
            -9.54975, -9.50031, -9.44977, -9.40012, -9.34969, -9.30016, -9.25026,
            -9.19997, -9.14997, -9.10018, -9.05012, -9., -8.95078, -8.89963,
            -8.85078, -8.80134, -8.74958, -8.69897, -8.64975, -8.60033, -8.54975,
            -8.50031, -8.44977, -8.40012, -8.34969, -8.30016, -8.25026, -8.19997,
            -8.14997, -8.10018, -8.05012, -8., -7.95078, -7.89963, -7.85078,
            -7.80134, -7.74958, -7.69897, -7.64975, -7.60033, -7.54975, -7.50031,
            -7.44977, -7.40012, -7.34969, -7.30016, -7.25026, -7.19997, -7.14997,
            -7.10018, -7.05012
        ])
        # Photon density in GeV-1 cm-3
        # First coordinate z, second energy
        self.ph_density_base = 10**np.loadtxt(
            join(resource_path, 'photon_spectra', 'EBL_inoue_baseline.dat'))
        # model with upper Pop-III limit
        self.ph_density_upper = 10**np.loadtxt(
            join(resource_path, 'photon_spectra', 'EBL_inoue_up_pop3.dat'))
        # model with lower Pop-III limit
        self.ph_density_lower = 10**np.loadtxt(
            join(resource_path, 'photon_spectra', 'EBL_inoue_low_pop3.dat'))

    def _construct_splines(self):
        # 3D interpolation of grid idx function g(E, z)
        z_map = {}
        for zi, z in zip(np.arange(110), self.z_dist):
            z_map[z] = zi

        ee, zzi = np.meshgrid(self.energy, self.z_dist)

        ngamma_base = np.vectorize(lambda e, z: np.interp(
            e, self.energy, self.ph_density_base[z_map[z], :]))
        ngamma_lower = np.vectorize(lambda e, z: np.interp(
            e, self.energy, self.ph_density_lower[z_map[z], :]))
        ngamma_upper = np.vectorize(lambda e, z: np.interp(
            e, self.energy, self.ph_density_upper[z_map[z], :]))

        self.spl2D = [
            ("Inoue_base", interp2d(
                self.energy, self.z_dist, ngamma_base(ee, zzi), fill_value=0., kind='linear')),
            ("Inoue_lower", interp2d(
                self.energy, self.z_dist, ngamma_lower(ee, zzi), fill_value=0., kind='linear')),
            ("Inoue_upper", interp2d(
                self.energy, self.z_dist, ngamma_upper(ee, zzi), fill_value=0., kind='linear'))
        ]


class Gilmore2011(EBLPhotonField):

    def __init__(self):
        """ Gilmore et al.

        EBL Model published in arXiv:1104.0671

        raw data files downloaded from http://physics.ucsc.edu/~joel/EBLdata-Gilmore2012/
        """

        # Raw data files in
        # 1st column: wavelength in <ang>
        # 2nd.. column: proper flux in <erg sec^{-1} cm^{-2} ang^{-1} sr^{-1}>
        # redshifts corresponing to flux are listed in header

        self.z_dist = np.array([0.0,
                                0.015,
                                0.025,
                                0.044,
                                0.05,
                                0.2,
                                0.4,
                                0.5,
                                0.6,
                                0.8,
                                1.0,
                                1.25,
                                1.5,
                                2.0,
                                2.5,
                                3.0,
                                4.0,
                                5.0,
                                6.0,
                                7.0])
        self.fixed = np.loadtxt(
            join(resource_path, 'photon_spectra', 'eblflux_fixed.dat'), skiprows=1)
        self.fiducial = np.loadtxt(
            join(resource_path, 'photon_spectra', 'eblflux_fiducial.dat'), skiprows=1)

        z_map_gilmore = {}
        for zi, z in enumerate(self.z_dist):
            z_map_gilmore[z] = zi

        # convert the flux to a photon flux $F = \frac{E}{dA dt d\lambda d\Omega}$
        # to an energy density $N = \frac{E}{dV dE}$ using the formula:
        #: $N = \frac{\lambda^2}{h c} \cdot F \cdot \frac{4 \pi}{c}$

        # 1st column: energy in <GeV>
        # 2nd.. column: photon density <GeV^{-3} cm^{-3}>
        c = 299792458  # m / s
        h = 4.135667662 * 1e-15 * 1e-9  # GeV s
        erg2GeV = 624.151

        fixed_wavelength = self.fixed[:, 0]
        fixed_flux = self.fixed[:, 1:]
        self.fixed_energy = (c * h) / (self.fixed[:, 0] * 1e-10)
        self.fixed_density = (fixed_wavelength[:, np.newaxis]**2 / (h * c)
                              * fixed_flux * (4 * sconst.pi / c) * (erg2GeV / 1e10 / 1e2))
        # also divide by energy for spectral density
        self.fixed_density /= self.fixed_energy[:, np.newaxis]

        fiducial_wavelength = self.fiducial[:, 0]
        fiducial_flux = self.fiducial[:, 1:]
        self.fiducial_energy = (c * h) / (self.fiducial[:, 0] * 1e-10)
        self.fiducial_density = (fiducial_wavelength[:, np.newaxis]**2 / (h * c)
                                 * fiducial_flux * (4 * sconst.pi / c) * (erg2GeV / 1e10 / 1e2))
        # also divide by energy for spectral density
        self.fiducial_density /= self.fiducial_energy[:, np.newaxis]

    def _construct_splines(self):

        self.spl2D = [
            ("Gilmore_fixed", interp2d(self.fixed_energy, self.z_dist,
                                       self.fixed_density.T, fill_value=0., kind='linear')),
            ("Gilmore_fiducial", interp2d(self.fiducial_energy, self.z_dist,
                                          self.fiducial_density.T, fill_value=0., kind='linear')),
        ]


class Dominguez2010(EBLPhotonField):
    """From 
    Extragalactic background light inferred from AEGIS galaxy-SED-type fractions",
    A. Dominguez et al., 2011, MNRAS, 410, 2556, arXiv:1007.1459

    raw data files downloaded from  http://side.iaa.es/EBL/"""

    def __init__(self):
        # Raw data files in
        # 1st column: wavelength in <micron>
        # 2nd.. column: wavelength * proper flux in <nW m^{-2} sr^{-1}>
        # redshifts corresponing to flux are listed in header

        self.z_dist = np.array([0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.9])
        self.base = np.loadtxt('./ebl_dominguez11.out')
        self.upper = np.loadtxt('./ebl_upper_uncertainties_dominguez11.out')
        self.lower = np.loadtxt('./ebl_lower_uncertainties_dominguez11.out')

        z_map = {}
        for zi, z in enumerate(self.z_dist):
            z_map[z] = zi

        # convert to
        # 1st column: energy in <GeV>
        # 2nd.. column: photon density <GeV cm^{-3}>
        c = 299792458  # m / s
        h = 4.135667662 * 1e-15 * 1e-9  # GeV s
        nWs2GeV = 6.241506363094e18 * 1e-9 * 1e-9

        wavelength = self.base[:, 0]
        flux = self.base[:, 1:]
        self.energy = (c * h) / (self.base[:, 0] * 1e-6)

        self.density = (wavelength[:, np.newaxis] / (h * c)
                        * flux * (4 * sconst.pi / c) * (nWs2GeV * 1e-12))
        # now convert from comoving to proper density
        self.density *= (1 + self.z_dist[np.newaxis, :])**3
        # also divide by energy for spectral density
        self.density /= self.energy[:, np.newaxis]

        upper_wavelength = self.upper[:, 0]
        upper_flux = self.upper[:, 1:]
        self.upper_energy = (c * h) / (self.upper[:, 0] * 1e-6)

        self.upper_density = (upper_wavelength[:, np.newaxis] / (h * c)
                              * upper_flux * (4 * sconst.pi / c) * (nWs2GeV * 1e-12))
        # now convert from comoving to proper density
        self.upper_density *= (1 + self.z_dist[np.newaxis, :])**3
        # also divide by energy for spectral density
        self.upper_density /= self.upper_energy[:, np.newaxis]

        lower_wavelength = self.lower[:, 0]
        lower_flux = self.lower[:, 1:]
        self.lower_energy = (c * h) / (self.lower[:, 0] * 1e-6)

        self.lower_density = (lower_wavelength[:, np.newaxis] / (h * c)
                              * lower_flux * (4 * sconst.pi / c) * (nWs2GeV * 1e-12))
        # now convert from comoving to proper density
        self.lower_density *= (1 + self.z_dist[np.newaxis, :])**3
        # also divide by energy for spectral density
        self.lower_density /= self.lower_energy[:, np.newaxis]

    def _construct_splines(self):

        self.spl2D = [
            ("Dominguez_base", interp2d(self.energy, self.z_dist,
                                        self.density.T, fill_value=0., kind='linear')),
            ("Dominguez_upper", interp2d(self.upper_energy, self.z_dist,
                                         self.upper_density.T, fill_value=0., kind='linear')),
            ("Dominguez_lower", interp2d(self.lower_energy, self.z_dist,
                                         self.lower_density.T, fill_value=0., kind='linear')),
        ]
