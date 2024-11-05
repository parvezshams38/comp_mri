"""
Computational Magnetic Resonance Imaging (CMRI) 2024/2025 Winter semester

- Author          : Jinho Kim
- Email           : <jinho.kim@fau.de>
"""

import matplotlib.pyplot as plt
import numpy as np
import utils
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.filters import window


class Lab02_op:
    def load_kdata(self):
        mat = utils.load_data("kdata1.mat")
        return mat["kdata1"]

    def load_kdata_os(self):
        mat = utils.load_data("kdata2.mat")
        return mat["kdata2"]

    def fft2c(self, x, axes=(-2, -1)):
        """
        This function calculates the 2D Fourier Transform of the input array x.
        Since the center of the kdata is in the middle of the matrix, we need to shift the center to the top-left corner.
        Do not forget to preserve the energy of the signal (Keep orthonormality).

        args:
            x:      Input array. (shape of [N, N])
            axes:   Axes to perform the Fourier Transform. (tuple of int)

        return:
            The 2D Fourier Transform of the input array x. (shape of [N, N])
        """
        # Your code here ...
        if x.ndim == 1:
            temp = fftshift(x)
            temp = np.fft.fft(temp, norm="ortho")
            temp = fftshift(temp)
        else:
            temp = fftshift(x)
            temp = fft2(temp, norm = 'ortho')
            temp = fftshift(temp)
        return temp

    def ifft2c(self, x, axes=(-2, -1)):
        """
        This function calculates the 2D inverse Fourier Transform of the input array x.
        Since the center of the kdata is in the middle of the matrix, we need to shift the center to the top-left corner.
        Do not forget to preserve the energy of the signal (Keep orthonormality).

        args:
            x:      Input array. (shape of [N, N])
            axes:   Axes to perform the inverse Fourier Transform. (tuple of int)

        return:
            The 2D inverse Fourier Transform of the input array x. (shape of [N, N])

        """
        # Your code here ...

        if x.ndim == 1:
            temp = ifftshift(x)
            temp = np.fft.ifft(temp, norm="ortho")
            temp = ifftshift(temp)
        else:
            temp = ifftshift(x)
            temp = ifft2(temp,norm="ortho")
            temp = ifftshift(temp)
        return temp

    def get_square_filter(self, kdata: np.ndarray, mask_size: int):
        """
        This function returns the zero-filled square filter of the same size as kdata.
        The outer values of the filter are zero and the center values of [mask_size x mask_size] are one.

        @param:
            kdata:          k-space data. (shape of [N, N])
            mask_size:      size of the filter. (int)
        @return:
            filter:         Filter. (shape of [N, N])
        """
        # Your code here ...
        filter = np.zeros_like(kdata)
        n = kdata.shape[0]
        for i in range(n//2-mask_size//2,n//2+mask_size//2):
            for j in range(n//2-mask_size//2,n//2+mask_size//2):
                filter[i, j] = 1

        return filter

    def filtering(self, kdata: np.ndarray, filter: np.ndarray):
        """
        This function applies the filter to the kspace

        @param:
            kdata:              k-space data. (shape of [N, N])
            filter:             Cropping filter (shape of [N, N])
        @return:
            filtered_kdata:     Filtered k-space data. (shape of [N, N])
        """
        # Your code here ...

        filtered_kdata = np.multiply(kdata, filter)
        return filtered_kdata

    def get_psf_1d_square(self, kdata: np.ndarray, mask_size: int, **kwargs):
        """
        This function returns the 1D point spread function (PSF) of the square filter.
        For the 1D filter, you take the middle row of the 2D filter.
        The 1D PSF is the absolute value of the inverse Fourier Transform of the filter.

        @param:
            kdata:          K-space data. (shape of [N, N])
            mask_size:      Size of the filter. (int)
        @return:
            psf:            Point spread function. (shape of [N])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        get_square_filter = kwargs.get("get_square_filter", self.get_square_filter)
        ifft2c = kwargs.get("ifft2c", self.ifft2c)

        # Your code here ...
        n = kdata.shape[0]
        square_filter_2d = self.get_square_filter(kdata, mask_size)
        square_filter_1d = square_filter_2d[n//2]
        psf = np.abs(ifft2c(square_filter_1d))

        return psf

    def get_fwhm(self, psf: np.ndarray):
        """
        This function calculates the Full Width Half Maximum of the given 1D PSF.

        @param:
            psf:            1D point spread function. (shape of [N])
        @return:
            fwhm:           Full Width Half Maximum. (int)
        """
        # Your code here ...
        max_amp = np.max(psf)
        half_max_amp = max_amp / 2
        indices = np.where(psf >= half_max_amp)
        #print(indices)
        fwhm = np.max(indices)-np.min(indices)
        return fwhm

    def get_hamming_filter(self, kdata: np.ndarray, mask_size: int):
        """
        This function returns the zero-filled Hamming filter of the same size as kdata.
        The outer values of the filter are zero and the center of [mask_size x mask_size] is a Hamming filter.
        To generate the Hamming filter, use the 'window' function from skimage.filters.

        @param:
            kdata:          k-space data. (shape of [N, N])
            mask_size:      size of the filter. (int)
        @return:
            filter:         Filter. (shape of [N, N])
        """
        # Your code here ...
        filter_h = window("hamming", (mask_size, mask_size))

        filter = np.zeros_like(kdata)
        n, m = kdata.shape[0], kdata.shape[1]
        k, l = 0, 0
        for i in range(n // 2 - mask_size // 2, n // 2 - mask_size // 2 + mask_size):
            l = 0
            for j in range(m // 2 - mask_size // 2, m // 2 - mask_size // 2 + mask_size):
                filter[i, j] = filter_h[k, l]
                l += 1
            k += 1

        return filter

    def get_psf_1d_hamming(self, kdata: np.ndarray, mask_size: int, **kwargs):
        """
        This function returns the 1D point spread function (PSF) of the hamming filter.
        For the 1D filter, you take the middle row of the 2D filter.
        The 1D PSF is the absolute value of the inverse Fourier Transform of the filter.

        @param:
            kdata:          K-space data. (shape of [N, N])
            mask_size:      Size of the filter. (int)
        @return:
            psf:            Point spread function. (shape of [N])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        get_hamming_filter = kwargs.get("get_hamming_filter", self.get_hamming_filter)
        ifft2c = kwargs.get("ifft2c", self.ifft2c)

        # Your code here ...

        n = kdata.shape[0]
        hamming_filter_2d = self.get_hamming_filter(kdata, mask_size)
        hamming_filter_1d = hamming_filter_2d[n // 2]

        psf = np.abs(ifft2c(hamming_filter_1d))

        return psf

    def cut_oversampled(self, kdata_os, os_factor, **kwargs):
        """
        This function removes the oversampling by cropping the frequency-encoding dimension in the image domain.

        @param:
            kdata_os:           Oversampled k-space data. (shape of [N, M])
            os_factor:          Oversampling factor. (int)
        @return:
            cropped_img:        Cropped image. (shape of [N, M//os_factor])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        ifft2c = kwargs.get("ifft2c", self.ifft2c)

        # Your code here ...
        actual_image = ifft2c(kdata_os)
        n, m = actual_image.shape[0], actual_image.shape[1]

        col_to_discard = m - m//os_factor                          #m-m/of________
        left = col_to_discard // 2
        right = left + m//os_factor
        #print(left, right)
        cropped_img = actual_image[:,left:right]
        return cropped_img
    
if __name__ == "__main__":
    # %% Load modules
    # This import is necessary to run the code cell-by-cell
    from lab02 import *

    # %% Define the lab02 object and load kdata and kdata_os
    op = Lab02_op()
    kdata = op.load_kdata()
    kdata_os = op.load_kdata_os()
