"""
Computational Magnetic Resonance Imaging (CMRI) 2024/2025 Winter semester

- Author          : Jinho Kim
- Email           : <jinho.kim@fau.de>
"""

from typing import Tuple

import h5py
import numpy as np
import utils
from numpy.linalg import pinv


class Lab06_op:

    def load_data(self, dpath="data_brain_8coils.mat"):
        mat = utils.load_data(dpath)
        kdata = mat["d"]
        sens_maps = mat["c"]
        noise_maps = mat["n"]
        psi = np.cov(noise_maps.T)

        self.set_self(kdata)

        return kdata, sens_maps, noise_maps, psi

    def set_self(self, kdata):
        self.PE, self.RO, self.nCoil = kdata.shape

    def load_SESNE(self, R, dpath=None):
        if dpath is None:
            dpath = utils._get_root() / "SENSE_recons.h5"

        with h5py.File(dpath, "r") as f:
            sense_recon = f[f"SENSE_R{R}"][...]

        return sense_recon

    def get_acs(self, kdata: np.ndarray, nACS: int) -> np.ndarray:
        """
        Get ACS from the kspace

        @param kdata:       The kspace data [PE, RO, nCoil]
        @param nACS:        The number of ACS lines

        @return: acs:       The auto-calibration signal [nACS, RO, nCoil]
        """
        # Your code here ...

        acs = None
        return acs

    def get_block_h(self, kernel_size: Tuple[int, int], R: int) -> int:
        """
        Get the height of the GRAPPA block

        @param kernel_size:     The GRAPPA kernel size, Union[(2, 3), (4, 5), (6, 7)]
        @param R:               The acceleration factor

        @return:                The height of the block
        """
        # Your code here ...

        return None

    def get_block_w(self, kernel_size: Tuple[int, int]) -> int:
        """
        Get the width of the GRAPPA block

        @param kernel_size:     The GRAPPA kernel size, Union[(2, 3), (4, 5), (6, 7)]

        @return:                The width of the block
        """
        # Your code here ...

        return None

    def get_n_b(self, nACS: int, kernel_size: Tuple[int, int], R: int, **kwargs) -> int:
        """
        Get the number of blocks in the ACS

        @param nACS:            The number of ACS lines
        @param kernel_size:     The GRAPPA kernel size, Union[(2, 3), (4, 5), (6, 7)]
        @param R:               The acceleration factor

        @return: n_b:           The number of blocks in the ACS
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        get_block_h = kwargs.get("get_block_h", self.get_block_h)
        get_block_w = kwargs.get("get_block_w", self.get_block_w)

        # Your code here ...

        n_b = None

        return n_b

    def get_n_kc(self, kernel_size: Tuple[int, int]) -> int:
        """
        Get n_k * n_c. This is the number of coils times that of kernel size

        @param kernel_size:     The GRAPPA kernel size, Union[(2, 3), (4, 5), (6, 7)]

        @return: n_kc:          The number of coils and kernel size, n_kc in the lecture slides
        """
        # Your code here ...

        n_kc = None

        return n_kc

    def extract(
        self, acs: np.ndarray, nACS: int, kernel_size: Tuple[int, int], R: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract source and target points from ACS

        @param acs:             The auto-calibration signal [nACS, RO, nCoil]
        @param nACS:            The number of ACS lines
        @param kernel_size:     The GRAPPA kernel size, Union[(2, 3), (4, 5), (6, 7)]
        @param R:               The acceleration factor

        @return:
            src :               source points [n_b x n_kc]
            targ:               target points [R-1 x n_b x nCoil]
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        get_n_b = kwargs.get("get_n_b", self.get_n_b)
        get_n_kc = kwargs.get("get_n_kc", self.get_n_kc)
        get_block_h = kwargs.get("get_block_h", self.get_block_h)

        # Your code here ...
        src = None
        targ = None

        return src, targ

    def get_weights(self, src: np.ndarray, targ: np.ndarray) -> np.ndarray:
        """
        Get the weights for GRAPPA reconstruction.
        Use pinv function from numpy.linalg to get the pseudo-inverse.
        Use @ operator for matrix multiplication.

        @param src:         The source points [n_b x n_kc]
        @param targ:        The target points [R-1 x n_b x nCoil]

        @return:            The weights for GRAPPA reconstruction [R-1, n_kc, nCoil]
        """
        # Your code here ...

        weights = None
        return weights

    def get_mask(self, kdata: np.ndarray, R: int) -> np.ndarray:
        """
        Get undersampling mask, which keeps the original size, by taking every R-th line in the PE direction.
        Start to take the first line.

        @param kdata:       The kspace data [PE, RO, nCoil]
        @param R:           The acceleration factor

        @return:            Undersampling mask [PE, RO, nCoil]
        """
        # Your code here ...

        mask = None
        return mask

    def undersample(self, kdata: np.ndarray, R: int, acs: np.ndarray, nACS: int, **kwargs) -> np.ndarray:
        """
        Undersample the kdata along the PE direction while keeping the original size and the ACS.
        Apply the undersampling mask to the kdata and fill the ACS lines with the ACS data.

        @param kdata:       kdata [PE, RO, nCoil]
        @param R:           The acceleration factor
        @param acs:         The auto-calibration signal [nACS, RO, nCoil]
        @param nACS:        The number of ACS lines

        @return:            Undersampled kdata [PE, RO, nCoil]
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        get_mask = kwargs.get("get_mask", self.get_mask)

        # Your code here ...
        undersampled = None

        return undersampled

    def get_pad_PE_up(self, kernel_PE: int, R: int) -> int:
        """
        Get the padding size for the PE direction (up)

        @param kernel_PE:       The PE size of the kernel
        @param R:               The acceleration factor

        @return:                The padding size
        """
        # Your code here ...
        return None

    def get_pad_PE_down(self, kernel_PE: int, R: int) -> int:
        """
        Get the padding size for the PE direction (down)

        @param kernel_PE:       The PE size of the kernel
        @param R:               The acceleration factor

        @return:                The padding size
        """
        # Your code here ...
        return None

    def get_pad_RO_left(self, kernel_RO: int) -> int:
        """
        Get the padding size for the RO direction (left)

        @param kernel_RO:       The RO size of the kernel

        @return:                The padding size of the left readout direction
        """
        # Your code here ...
        return None

    def get_pad_RO_right(self, kernel_RO: int) -> int:
        """
        Get the padding size for the RO direction (right)

        @param kernel_RO:       The RO size of the kernel

        @return:                The padding size of the right readout direction
        """
        # Your code here ...
        return None

    def zero_padding(self, input_k: np.ndarray, kernel_size: Tuple[int, int], R: int, **kwargs) -> np.ndarray:
        """
        Zero padding for interpolation using kernels

        @param input_k:         Input kdata [PE, RO, nCoil]
        @param kernel_size:     The GRAPPA kernel size, Union[(2, 3), (4, 5), (6, 7)]
        @param R:               Acceleration factor

        @return:
            zp_kdata:           Zero-padded kspace [PE + pad_PE_up + pad_PE_down, RO + pad_RO_left + pad_RO_right, nCoil]
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        get_pad_PE_up = kwargs.get("get_pad_PE_up", self.get_pad_PE_up)
        get_pad_PE_down = kwargs.get("get_pad_PE_down", self.get_pad_PE_down)
        get_pad_RO_left = kwargs.get("get_pad_RO_left", self.get_pad_RO_left)
        get_pad_RO_right = kwargs.get("get_pad_RO_right", self.get_pad_RO_right)

        # Your code here ...
        zp_kdata = None

        return zp_kdata

    def grappa_core(
        self, kdata_us: np.ndarray, weights: np.ndarray, kernel_size: Tuple[int, int], R: int, **kwargs
    ) -> np.ndarray:
        """
        This method interpolates missing kdata coefficients using the GRAPPA weights

        @param kdata_us:        The undersampled kspace data [PE, RO, nCoil]
        @param weights:         The GRAPPA weights [R-1, n_kc, nCoil]
        @param kernel_size:     The GRAPPA kernel size, Union[(2, 3), (4, 5), (6, 7)]
        @param R:               The acceleration factor

        @return:                The GRAPPA reconstructed kspace [PE + pad_PE_up + pad_PE_down, RO + pad_RO_left + pad_RO_right, nCoil]
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        get_block_h = kwargs.get("get_block_h", self.get_block_h)
        zero_padding = kwargs.get("zero_padding", self.zero_padding)

        # Your code here ...
        grappa_k = None

        return grappa_k

    def crop2original(self, padded_input_k: np.ndarray, kernel_size: Tuple[int, int], R: int, **kwargs) -> np.ndarray:
        """
        Crop the padded kspace to its original size

        @param padded_input_k:      Padded kspace [PE + pad_PE_up + pad_PE_down, RO + pad_RO_left + pad_RO_right, nCoil]
        @param kernel_size:         The GRAPPA kernel size, Union[(2, 3), (4, 5), (6, 7)] **##### kernel size is always even * odd
        @param R:                   Acceleration factor

        @return:                    Cropped kspace [PE, RO, nCoil]
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        get_pad_PE_up = kwargs.get("get_pad_PE_up", self.get_pad_PE_up)
        get_pad_RO_left = kwargs.get("get_pad_RO_left", self.get_pad_RO_left)

        # Your code here ...
        padded_input_k = None

        return padded_input_k

    def data_consistency(self, input_k: np.ndarray, kdata_us: np.ndarray) -> np.ndarray:
        """
        Data consistency step for GRAPPA.
        This method keep the original signal from the undersampled kspace and fill only the missing coefficients with the GRAPPA reconstructed kspace

        @param input_k:         GRAPPA reconstructed kspace [PE, RO, nCoil]
        @param kdata_us:        Undersampled kspace [PE, RO, nCoil]

        @return:                GRAPPA reconstructed kspace with the original signal [PE, RO, nCoil]
        """
        # Your code here ...
        dc_k = None

        return dc_k

    def run_grappa(self, kdata: np.ndarray, nACS: int, kernel_size: Tuple[int, int], R: int) -> np.ndarray:
        """
        Running GRAPPA algorithm for the undersampled kdata at the acceleration factor R
        The GRAPPA algorithm is applyed with nACS ACS lines and kernel size.

        @param kdata:           The kspace data [PE, RO, nCoil]
        @param nACS:            The number of ACS lines
        @param kernel_size:     The GRAPPA kernel size, Union[(2, 3), (4, 5), (6, 7)]
        @param R:               The acceleration factor

        @return:                GRAPPA reconstructed kspace [PE, RO, nCoil]
        """
        acs = self.get_acs(kdata, nACS)
        src, targ = self.extract(acs, nACS, kernel_size, R)
        weights = self.get_weights(src, targ)

        kdata_us = self.undersample(kdata, R, acs, nACS)

        grappa_k_zp = self.grappa_core(kdata_us, weights, kernel_size, R)
        grappa_k_cropped = self.crop2original(grappa_k_zp, kernel_size, R)
        grappa_k = self.data_consistency(grappa_k_cropped, kdata_us)

        return grappa_k


if __name__ == "__main__":
    # %% Load modules
    # This import is necessary to run the code cell-by-cell
    from lab06 import *

    # %%
    op = Lab06_op()
    kdata, sens_maps, noise_maps, psi = op.load_data()
