"""
Computational Magnetic Resonance Imaging (CMRI) 2024/2025 Winter semester

- Author          : Jinho Kim
- Email           : <jinho.kim@fau.de>
"""

from typing import Tuple

import numpy as np
import torch
import torchkbnufft as tkbn
import utils
from grid import grid

import matplotlib.pyplot as plt


class Lab04_op:

    def __init__(self, device_num=0):
        self.PI = np.pi
        self.GA = 111.246117975 * self.PI / 180  # Golden angle

        # For NUFFT operator
        self.device = torch.device(f"cuda:{device_num}") if torch.cuda.is_available() else torch.device("cpu")

    def load_kdata(self):
        mat = utils.load_data("radial_data.mat")
        k_radial = mat["k"]
        return k_radial

    def get_traj(self, k_radial):
        """
        This method returns the trajectory of the radial k-space data.

        Args:
            k_radial:           k-space data. (shape of [Readout, Spokes])

        Return:
            traj:               trajectory of the k-space data. (shape of [Readout])
        """
        # Your code here ...
        m,n=np.shape(k_radial)
        traj = np.zeros_like(k_radial).astype(complex)
        mag = np.linspace(-0.5,0.5,m)
        for i in range(m):
            for k in range(n):
                ang = self.PI/2+k*self.GA
                traj[i,k] = mag[i]*np.exp(1j*ang)
        
        return traj

    def calc_nyquist(self, k_radial) -> int:
        """
        This method calculates the minimum number of spokes that satisfy Nyquist sampling theorm.

        Args:
            k_radial:        k-space data. (shape of [Readout, Spokes])

        Returns:
            spokes_nyq:      minimum number of spokes that satisfy Nyquist sampling theorem. (Ceilling up integer)
        """
        # Your code here ...

        spokes_nyq = np.ceil((self.PI/2)*np.shape(k_radial)[0])

        return int(spokes_nyq)

    def grid_radial(self, k_radial, traj):
        """
        This method maps the radial k-space data to Cartesian k-data using the triangular gridding kernel of width 2.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])
            traj:           trajectory of the k-space data. (shape of [Readout])

        Returns:
            k_cart:         Cartesian k-space data. (shape of [Readout, Readout])
        """
        # Your code here ...
        n,m = np.shape(k_radial)

        k_cart = grid(k_radial,traj,n)

        return k_cart

    def get_ramp(self, k_radial):
        """
        This method returns a ramp filter that can be used to weight the kspace data
        for gridding.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])

        Returns:
            ramp:           ramp filter (shape of [Readout, 1])
        """
        # Your code here ...
        r, s = np.shape(k_radial)
        ramp = np.zeros(r).astype(complex)
        ramp[0:r//2] = np.linspace(1,1/s,r//2)
        ramp[r//2:r] = np.linspace(1/s,1,r-r//2)
        return np.reshape(ramp, [r,1])

    def grid_radial_ds(self, k_radial, traj, **kwargs):
        """
        This method maps the density compensatied radial k-space data to Cartesian k-data using the triangular gridding kernel of width 2.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])
            traj:           trajectory of the k-space data. (shape of [Readout])

        Returns:
            k_cart_ds:      Densitiy compensated Cartesian k-space data. (shape of [Readout, Readout])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'get_ramp' in this method if you need to used it instead of calling it by self.get_ramp.
        get_ramp = kwargs.get("get_ramp", self.get_ramp)

        # Your code here ...
        n,m = np.shape(k_radial)
        ramp = np.reshape(get_ramp(k_radial),(n,))
        k_radial_temp = np.copy(k_radial)
        for j in range(m):
            k_radial_temp[:,j] *= ramp
        k_cart_ds = grid(k_radial_temp,traj,n)

        return k_cart_ds

    def grid_radial_ds_os(self, k_radial, traj, os_rate: int, **kwargs):
        """
        This method maps the density compensatied radial k-space data to Cartesian k-data using the triangular gridding kernel of width 2.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])
            traj:           trajectory of the k-space data. (shape of [Readout, Spokes])
            os_rate:        oversampling rate

        Returns:
            k_cart_ds_os:   Densitiy compensated oversampled Cartesian k-space data. (shape of [Readout * os_rate, Readout * os_rate])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'get_ramp' in this method if you need to used it instead of calling it by self.get_ramp.
        get_ramp = kwargs.get("get_ramp", self.get_ramp)

        # Your code here ...
        r,s = np.shape(k_radial)
        ramp = np.reshape(get_ramp(k_radial),(r,))
        k_radial_temp = np.copy(k_radial)
        for i in range(s):
            k_radial_temp[:,i] *= ramp 

        k_cart_ds_os = grid(k_radial_temp,traj,r*os_rate)

        return k_cart_ds_os

    def decimate2d(self, ov_image, target_shape: Tuple[int, int]):
        """
        This method crops the image to the desired size.

        Args:
            ov_image (np.ndarray):          Oversmapled image (shape of [Oversampled_Readout, Oversampled_Readout])
            target_shape (tuple):           Desired shape of the image

        Returns:
            image_crop (np.ndarray):        cropped image (shape of target_shape)
        """
        # Your code here ...
        os_r = np.shape(ov_image)[0]
        m,n = target_shape

        image_crop = ov_image[(os_r-m)//2:(os_r-m)//2+m, (os_r-n)//2:(os_r-n)//2+n]

        return image_crop

    def get_c(self, k_radial, traj, os_rate):
        """
        This method calculates the deapodization factor c(x,y) for the deapodization.
        Think about how to utilize the grid function to calculate c(x,y).
        If you want to use delta function, set 1 of the delta function at the center of the k-space. (ex. at Readout//2)

        Args:
            k_radial (np.ndarray):      k-space data. (shape of [Readout, Spokes])
            traj (np..ndarray):         trajectory of the k-space data. (shape of [Readout, Spokes])
            os_rate (int):              oversampling rate

        Returns:
            c:                          deapodization factor c(x,y) (shape of [Oversampled_Readout, Oversampled_Readout])
        """
        # Your code here ...
        r,s = np.shape(k_radial)
        delta = np.zeros([r,s])
        delta[r//2,s//2] = 1

        delta_response = grid(delta,traj,r*os_rate)
        c = utils.ifft2c(delta_response)
        return c

    def deapodization(self, k_radial, traj, os_rate, **kwargs):
        """
        This method performs deapodization on the k-space data.
            1. Get the gridding kernel using get_c method.
            2. Grid the k-space data with opversampling and density compensation.
            3. Reconstruction the grid k-space data.
            4. Deapodize the reconstructed k-space data.
            5.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])
            traj:           trajectory of the k-space data. (shape of [Readout])
            os_rate:        oversampling rate

        Returns:
            deapod_recon:   De-apodized reconstruction image (shape of [Readout, Readout])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'grid_radial_ds_os' and 'decimate2d' in this method if you need to used them instead of calling them by self.grid_radial_ds_os and self.decimate2d.
        grid_radial_ds_os = kwargs.get("grid_radial_ds_os", self.grid_radial_ds_os)
        decimate2d = kwargs.get("decimate2d", self.decimate2d)
        get_c = kwargs.get("get_c", self.get_c)

        a = 1

        # Your code here ...
        c = get_c(k_radial,traj,os_rate)
        apodized_oversampled_kdata = grid_radial_ds_os(k_radial,traj,os_rate)
        apodized_oversampled_image_data = utils.ifft2c(apodized_oversampled_kdata)
        oversampled_image_data = apodized_oversampled_image_data / (c + a*np.ones_like(c))
        deapod_recon = decimate2d(oversampled_image_data,(np.shape(k_radial)[0],np.shape(k_radial)[0]))
        return deapod_recon

    def nufft_traj(self, k_radial):
        """
        This method returns the trajectory of the radial k-space data for NUFFT.
        - The lenght of the trajectory spock for NUFFT operator is 2 * PI (-PI ~ PI) and the first spock starts at the angle of 90 degrees.
        - Trajectory for the NUFFT operator needs to be in shape of (2, x) where 2 is the separate channels of real and imaginary part of the trajectory.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])

        Returns:
            ktraj:          trajectory of the k-space data for NUFFT. (shape of [2, Readout * Spokes])
        """
        # Your code here ...

        spokelength, nspokes = k_radial.shape
        angles = np.arange(nspokes) * self.GA 

  
        ky = np.linspace(-np.pi, np.pi, spokelength)[:, None] * np.cos(angles)  # Y component
        kx = np.linspace(-np.pi, np.pi, spokelength)[:, None] * (-np.sin(angles))  # X component

        ktraj = np.stack((kx.flatten(), ky.flatten()), axis=0)
        return ktraj

    def nufft_kdata(self, k_radial, **kwargs):
        """
        This method prepares the k-space data for NUFFT.
        - Use density commpenstated k-space data for NUFFT.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])

        Returns:
            nufft_kdata:    NUFFT k-space data. (shape of [Batch(1), Coil(1), Readout*Spokes])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'get_ramp' in this method if you need to used it instead of calling it by self.get_ramp.
        get_ramp = kwargs.get("get_ramp", self.get_ramp)

        # Your code here ...
        k_image = get_ramp(k_radial) * k_radial
        k_image = k_image.flatten()
        nufft_kdata = np.reshape(k_image,(1,1,-1))

        return torch.tensor(nufft_kdata, dtype=torch.complex64, device=self.device)

    def nufft_recon(self, k_radial, im_size: Tuple[int, int], **kwargs):
        """
        This method reconstructs the image from the k-space data using NUFFT.
        - Define trajectories and k-data for a NUFFT operator.
        - Define a Adjoint NUFFT operator with im_size. Do not forget to set the self.device.

        Args:
            k_radial:       k-space data. (shape of [Readout, Spokes])
            im_size:        size of the desired reconstruction image.

        Returns:
            nufft_recon:    NUFFT reconstructed image. (shape of im_size)
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        # Use 'nufft_traj' and 'nufft_kdata' in this method if you need to used them instead of calling them by self.nufft_traj and self.nufft_kdata.
        nufft_traj = kwargs.get("nufft_traj", self.nufft_traj)
        nufft_kdata = kwargs.get("nufft_kdata", self.nufft_kdata)

        # Your code here ...

        ktraj = torch.tensor(nufft_traj(k_radial), dtype=torch.float32, device=self.device)
        kdata = nufft_kdata(k_radial)

        adjoint_nufft = tkbn.KbNufftAdjoint(im_size=im_size).to(self.device)
        nufft_recon = adjoint_nufft(kdata, ktraj)
        
        return nufft_recon.cpu().numpy()[0][0]


if __name__ == "__main__":
    # %% Load modules
    # This import is necessary to run the code cell-by-cell
    from lab04 import *

    # %%
    op = Lab04_op()
    k_radial = op.load_kdata()
    #utils.imshow([k_radial], norm=0.3)
    #print(np.shape(k_radial))
    #print(op.calc_nyquist(k_radial))

    traj = op.get_traj(k_radial)
    utils.plot_spokes(traj, 1)

    k_cart_ds_os_recon_deapod_cropped = op.deapodization(k_radial,op.get_traj(k_radial),2)
    utils.imshow([k_cart_ds_os_recon_deapod_cropped])

    nufft_recon = op.nufft_recon(k_radial, (k_radial.shape[0], k_radial.shape[0]))
    utils.imshow(
        [
            k_cart_ds_os_recon_deapod_cropped,
            nufft_recon,
        ],
        titles=[
            "De-apodization",
            "NUFFT",
        ],
        num_rows=1,
        pos=[1, 1],
        norm=0.6,
    ) 
