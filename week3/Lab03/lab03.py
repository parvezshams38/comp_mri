"""
Computational Magnetic Resonance Imaging (CMRI) 2024/2025 Winter semester

- Author          : Jinho Kim
- Email           : <jinho.kim@fau.de>
"""

import numpy as np
import utils
from skimage.filters import window


class Lab03_op:
    def __init__(self, PF):
        """
        PF: Partial Fourier factor. (float)
            This factor is used to calculate the original k-space size.
        """
        self.PF = PF

    def load_kdata_pf(self):
        mat = utils.load_data("kdata_phase_error_severe.mat")
        return mat["kdata"]

    def load_kdata_full(self):
        mat = utils.load_data("kdata1.mat")
        return mat["kdata1"]

    def get_half_zf_kdata(self, kdata: np.ndarray):
        """
        This function returns a half zero-filled kdata of its original shape.

        @param:
            kdata:          k-space data. (shape of [N, M])
        @return:
            zf_kdata:       Half zero-filled kdata. (shape of [N, N])
        """
        # Your code here ...

        zf_kdata = None

        return zf_kdata

    def hermitian_symmetry(self, zf_kdata: np.ndarray):
        """
        This function returns the Hermitian symmetry of the zero-filled k-space data without phase correction.

        @param:
            zf_kdata:       Zero-filled k-space data. (shape of [N, N])
        @return:
            hm_kdata:       Hermitian symmetric kdata. (shape of [N, N])
        """
        # Your code here ...

        hm_kdata = None

        return hm_kdata

    def estim_phs(self, kdata):
        """
        Phase estimation

        @Param:
            kdata:             asymmetric k-space data (shape of [N, M])
        @Return:
            estimated_phase:    estimated phase of the input kdata
        """
        # Your code here ...

        estimated_phase = None

        return estimated_phase

    def get_window(self, kdata, type="ramp"):
        """
        This function returns the window for the Hermitian symmetric extension

        @Param:
            kdata:          asymmetric k-space data
            type:           filter type ('ramp' or 'hamm')
        @Return:
            window_filter:  Window filter for the Hermitian symmetric extension
        """
        # Your code here ...

        window_filter = None

        return window_filter

    def pf_margosian(self, kdata, wtype, **kwargs):
        """
        Margosian reconstruction for partial Fourier (PF) MRI

        Param:
            kdata:      asymmetric k-space data
            wtype:      The type of window ('ramp' or 'hamm')
        Return:
            I: reconstructed magnitude image
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'estim_phs' and 'get_window' in this method if you need to used them instead of calling them by self.estim_phs and self.get_window.
        estim_phs = kwargs.get("estim_phs", self.estim_phs)
        get_window = kwargs.get("get_window", self.get_window)

        # Your code here ...

        I = None

        return I

    def pf_pocs(self, kdata, Nite, **kwargs):
        """
        POCS reconstruction for partial Fourier (PF) MRI

        Param:
            kdata:      asymmetric k-space data
            Nite:       number of iterations

        Return:
            I: reconstructed magnitude image
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'estim_phs' in this method if you need to used it instead of calling it by self.estim_phs.
        estim_phs = kwargs.get("estim_phs", self.estim_phs)

        # Your code here ...

        I = None

        return I


if __name__ == "__main__":
    # %% Load modules
    # This import is necessary to run the code cell-by-cell
    from lab03_solution import *

    # %% Define the lab03 object and load kdata.
    ## The partial Fourier factor is 9/16.
    PF = 9 / 16
    op = Lab03_op(PF)
    kdata = op.load_kdata_pf()
