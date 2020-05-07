"""Construction of 2D data tables from ASCII files obtained from models/simulations
or other sources.

Currently supported are the files generated for the papers:

    | Nuclear Physics Meets the Sources of the Ultra-High Energy Cosmic Rays
    | Denise Boncioli, Anatoli Fedynitch, Walter Winter
    | Sci.Rep. 7 (2017) 1, 4882
    | e-Print: 1607.07989 [astro-ph.HE]
    | DOI: 10.1038/s41598-017-05120-7

    | A new view on Auger data and cosmogenic neutrinos in light of different nuclear disintegration and air-shower models
    | Jonas Heinze, Anatoli Fedynitch, Denise Boncioli, Walter Winter
    | Astrophys.J. 873 (2019) 1, 88
    | e-Print: 1901.03338 [astro-ph.HE]
    | DOI: 10.3847/1538-4357/ab05ce

"""

from os import listdir
from os.path import join

import numpy as np
from six import with_metaclass

from prince_data_utils import resource_path
from prince.util import info

class CrossSectionsFromAscii(object):
    """Each class derived from this one is expected to load the
    data from some form of source independently and provide at the
    end definitions for the parameters:

        self.energy_grid, self.mothers_daughters, self.fragment_yields,
        self.inel_mothers, self.inelastic_cross_sctions.

    Args:
      f_root (str): The root name of the tabulated files, e.g. CRP2_TALYS_.

    """

    def __init__(self, f_root= 'CRP2_TALYS_'):
        self.energy_grid = None
        self.inel_mothers = None
        self.mothers_daughters = None
        self.inelastic_cross_sctions = None
        self.fragment_yields = None

        self._load(f_root)

        assert self.energy_grid is not None
        assert self.inel_mothers is not None
        assert self.mothers_daughters is not None
        assert self.inelastic_cross_sctions is not None
        assert self.fragment_yields is not None

        self._check_consistency()

    def _load(self, f_root):
        """Load cross section tables from files into memory.
        Needs to be defined in derived classes."""

        if not f_root.endswith('_'):
            f_root += '_'

        f_root = join(resource_path, 'photo-nuclear',f_root)
        info(0, 'Loading files', f_root + '*')
        self.energy_grid = np.loadtxt(f_root + 'egrid.dat.bz2')*1e-3 # to GeV
        self._inel_cs_tables = np.loadtxt(f_root + 'nonel.dat.bz2')
        self._inel_fragment_yields = np.loadtxt(f_root + 'incl_i_j.dat.bz2')

        assert self.energy_grid.shape[0] == \
            self._inel_cs_tables.shape[1] - 1 == \
            self._inel_fragment_yields.shape[1] - 2, \
            'Tables e-grids inconsistent {0} != {1} != {2}'.format(
                self.energy_grid.shape[0], self._inel_cs_tables.shape[1] - 1,
                self._inel_fragment_yields.shape[1] - 2)
        
        # Chunk the tables into their contents
        self.inel_mothers = self._inel_cs_tables[:,0].astype('int')
        self.inelastic_cross_sctions = self._inel_cs_tables[:,1:]*1e-27 #mbarn -> cm2

        self.mothers_daughters = self._inel_fragment_yields[:,0:2].astype('int')
        self.fragment_yields = self._inel_fragment_yields[:,2:]*1e-27 #mbarn -> cm2
        
    def _check_consistency(self):
        """Some cross checks for dimenstions and consistency between
        inelastic cross sections and yields are performed."""

        assert self.inel_mothers.shape[0] == self.inelastic_cross_sctions.shape[0]
        assert self.energy_grid.shape[0] == self.inelastic_cross_sctions.shape[1]

        assert self.mothers_daughters.shape[0] == self.fragment_yields.shape[0]
        assert self.energy_grid.shape[0] == self.fragment_yields.shape[1]

class PhotoMesonCSFromPickle(CrossSectionsFromAscii):

    def _load(self, f_root_):
        """Load from pickled dictionaries"""

        f_root = join(resource_path, 'photo-meson',f_root_)
        info(0, 'Loading files', f_root + '*')
        raw_csec = np.load(f_root + 'crosssec.npy',allow_pickle=True)
        energy, csec_proton, csec_neutron = raw_csec
        csec = np.load(f_root + 'redistribution_logbins.npy',allow_pickle=True)
        energy_redist, xbins, redist_proton, redist_neutron = csec

        # sophia crossections are in mubarn; convert here to cm^2
        csec_proton, csec_neutron = csec_proton * 1e-30, csec_neutron * 1e-30

        daughters_proton = np.array(list(redist_proton.keys()))
        fragments_proton = np.array(list(redist_proton.values()))
        daughters_neutron = np.array(list(redist_neutron.keys()))
        fragments_neutron = np.array(list(redist_neutron.values()))

        assert np.all(energy == energy_redist)
        assert xbins.shape[0]-1 == fragments_proton.shape[-1] == fragments_neutron.shape[-1]

        self.energy_grid = energy
        self.xbins = xbins
        self.inel_mothers = np.array([101, 100])
        self.inelastic_cross_sctions = np.stack([csec_proton, csec_neutron])
        channels_proton = np.stack(
            [np.full(daughters_proton.shape, 101), daughters_proton],axis=1)
        channels_neutron = np.stack(
            [np.full(daughters_neutron.shape, 101), daughters_neutron],axis=1)
        self.mothers_daughters = np.concatenate(
            [channels_proton, channels_neutron])
        # Fragments in raw data are in dn/dx, but we need dsigma/dx = dn/dx * sigma
        self.fragment_yields = np.concatenate([
            fragments_proton * csec_proton[None,:,None],
            fragments_neutron * csec_neutron[None,:,None]
        ])
