"""
Computational Magnetic Resonance Imaging (CMRI) 2024/2025 Winter semester

- Author          : Jinho Kim
- Email           : <jinho.kim@fau.de>
"""

import numpy as np
import utils
from skimage.filters import window
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt


class Lab03_op:
    def __init__(self, PF):
        """
        PF: Partial Fourier factor.(float)
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
        N,M = np.shape(kdata)[0], np.shape(kdata)[1]
        temp = np.zeros([N, N], dtype=complex)
        temp[:,0:N//2] = kdata[:,0:N//2]

        zf_kdata = temp
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
        N = np.shape(zf_kdata)[0]
        hm_kdata = np.zeros([N,N], dtype=complex)
        for i in range (N):
            for j in range (N):
                if j < N//2:
                    hm_kdata[i,j] = zf_kdata[i,j]
                else:
                    hm_kdata[i,j] = np.conjugate(zf_kdata[N-i-1, N-j-1])
        
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
        N,M = np.shape(kdata)
        temp = np.zeros([N,N], dtype=complex)
        f = window("hamming", (N, 2*M-N))

        for i in range (N):
            k = 0
            for j in range (N-M,M):
                temp [i,j] = kdata [i,j] * f [i,k]
                k += 1

        estimated_phase = np.angle(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(temp))))
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
        N, M = np.shape(kdata)
        width = 2 * M - N
        if type == "hamming":
            window_1d = window("hamming", shape=(width,width))
            window_1d = window_1d[width//2,:]
            window_1d = np.reshape(window_1d,(1,width))
            window_2d = np.dot(np.ones([N,1]),window_1d)
            window_2d /= np.max(window_2d)
        else:
            window_1d = np.linspace(1, 0, width)
            window_1d = np.reshape(window_1d,(1,width))
            window_2d = np.dot(np.ones([N,1]),window_1d)
            window_2d /= np.max(window_2d)

        window_filter = np.zeros([N,N])
        window_filter[:,N-M:M] = window_2d
        if type == "hamming":
            window_filter[:,0:N//2] = 1
        else:
            window_filter[:,0:N-M] = 1
        
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
        N,M = np.shape(kdata)
        kdata_squared_window = np.zeros([N,N],dtype=complex)
        kdata_squared_window[:,0:M] = kdata

        kdata_filtered_squared = kdata_squared_window * get_window(kdata, wtype)
        kdata_im = ifftshift(ifft2(fftshift(kdata_filtered_squared), norm="ortho"))
        
        estimated_phase = estim_phs(kdata)
        estimated_image = kdata_im * np.exp(-(1j)*estimated_phase)
        I = np.real(estimated_image)

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
        estimated_phase = estim_phs(kdata)        
        N,M = np.shape(kdata)
        kdata_squared_window = np.zeros([N,N],dtype=complex)

        for _ in range(Nite):
            kdata_squared_window[:,0:M] = kdata
            kdata_im = utils.ifft2c(kdata_squared_window)
            I = np.abs(kdata_im) * np.exp(1j*estimated_phase)
            kdata_squared_window = utils.fft2c(I)

        return I


if __name__ == "__main__":
    # %% Load modules
    # This import is necessary to run the code cell-by-cell
    #from lab03_solution import *

    # %% Define the lab03 object and load kdata.
    ## The partial Fourier factor is 9/16.
    PF = 9 / 16
    op = Lab03_op(PF)
    kdata = op.load_kdata_pf()

    half_data = op.get_half_zf_kdata(kdata)
    half_her = op.hermitian_symmetry(half_data)
    full_kdata = op.load_kdata_full()
    
    orginial_im = utils.normalization(utils.ifft2c(full_kdata))
    im_ramp = utils.normalization(op.pf_margosian(kdata, "ramp"))
    im_ham = utils.normalization(op.pf_margosian(kdata, "hamming"))
    im_pocs = utils.normalization(op.pf_pocs(kdata, 8))

    #utils.imshow([orginial_im-orginial_im,im_ramp-orginial_im,im_ham-orginial_im,im_pocs-orginial_im],num_rows=2,norm=.5)

    er_ramp = orginial_im-im_ramp
    er_ham = orginial_im-im_ham
    er_pocs = orginial_im-im_pocs

    e1 = np.sum(np.abs(er_ramp)*np.abs(er_ramp))
    e2 = np.sum(np.abs(er_ham)*np.abs(er_ham))
    e3 = np.sum(np.abs(er_pocs)*np.abs(er_pocs))

    print(e1,e2,e3)


    snr1 = np.mean(np.abs(er_ramp))/np.std(np.abs(er_ramp))
    snr2 = np.mean(np.abs(er_ham))/np.std(np.abs(er_ham))
    snr3 = np.mean(np.abs(er_pocs))/np.std(np.abs(er_pocs))

    print(np.abs(snr1),np.abs(snr2),np.abs(snr3))