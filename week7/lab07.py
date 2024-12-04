"""
Computational Magnetic Resonance Imaging (CMRI) 2024/2025 Winter semester

- Author          : Jinho Kim
- Email           : <jinho.kim@fau.de>
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchkbnufft as tkbn
import utils
from torchkbnufft.modules.kbnufft import KbNufft as NUFFT_Forward_OP
from torchkbnufft.modules.kbnufft import KbNufftAdjoint as NUFFT_Adjoint_OP
from tqdm.auto import tqdm


class Lab07_op:

    def __init__(self):
        self.PI = np.pi

        # For NUFFT operator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        mat = utils.load_data("data_radial_brain_4ch.mat")
        kspace = mat["kdata"]
        sens_maps = mat["c"]
        traj = mat["k"]  # range of traj: -0.5~0.5
        dc_weights = mat["w"]
        gt = mat["img_senscomb"]

        return kspace, sens_maps, traj, dc_weights, gt

    def get_nufft_ob(self, im_size: Tuple[int, int], grid_size: Tuple[int, int]) -> tkbn.KbNufft:
        """
        Define a NUFFT operator with the given parameters.

        Args:
            im_size:            Size of image with length being the number of dimensions.
            grid_size:          Size of grid to use for interpolation,
                                typically 1.25 to 2 times ``im_size``.

        Returns:
            nufft_ob:           NUFFT operator
        """
        # Your code here ...
        nufft_ob = None
        return nufft_ob

    def get_nufft_adj_ob(self, im_size: Tuple[int, int], grid_size: Tuple[int, int]) -> tkbn.KbNufftAdjoint:
        """
        Define a adjoint NUFFT operator with the given parameters.
        Don't forget to set the device to self.device.


        Args:
            im_size:            Size of image with length being the number of dimensions.
            grid_size:          Size of grid to use for interpolation,
                                typically 1.25 to 2 times ``im_size``.

        Returns:
            nufft_adj_ob:       Adjoint NUFFT operator
        """
        # Your code here ...
        nufft_adj_ob = None
        return nufft_adj_ob

    def get_nufft_kdata(self, kdata: np.ndarray, dc_weights: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        This method converts the k-space data to the format that NUFFT operator can use.
        Don't forget to set the device to self.device.

        Args:
            kdata: k-space data. (shape of [Readout, Spokes, Channels])
            dc_weights: [Optional] Density compensation weights. When dc_weights is given, apply it to kdata. (shape of [Readout, Spokes])

        Returns:
            kdata_nufft: k-space data in the format that NUFFT operator can use. (shape of [Batch(1), Channels, Readout*Spokes])
        """
        kdata_nufft = kdata.copy()

        # Your code here ...
        if dc_weights is not None:
            kdata_nufft = None

        kdata_nufft = None

        return kdata_nufft

    def get_nufft_traj(self, traj: np.ndarray) -> torch.Tensor:
        """
        This method converts the trajectory to the format that NUFFT operator can use.
        traj in the provided .mat file is in the range of [-0.5 ~ 0.5], while NUFFT operator requires the range of [-PI ~ PI].
        Don't forget to set the device to self.device.

        Args:
            traj: Trajectory of the k-space data. (shape of [Readout, Spokes])

        Returns:
            traj_stack: Trajectory of the k-space data in the format that NUFFT operator can use. (shape of [2, Readout*Spokes], two channels for real and imaginary parts)
        """
        # Your code here ...

        # To make the range of traj to [-PI ~ PI]
        traj_nufft = None
        traj_stack = None

        return traj_stack

    def get_nufft_sens_maps(self, sens_maps: np.ndarray) -> torch.Tensor:
        """
        This method converts the sensitivity maps to the format that NUFFT operator can use.
        Don't forget to set the device to self.device.

        Args:
            sens_maps: Sensitivity maps of the k-space data. (shape of [Readout//2, Readout // 2, Channels])

        Returns:
            sens_maps_nufft: Sensitivity maps of the k-space data in the format that NUFFT operator can use. (shape of [Batch(1), Channels, Readout // 2, Readout // 2])
        """
        # Your code here ...
        sens_maps_nufft = None

        return sens_maps_nufft

    def nufft_recon(
        self,
        kdata: np.ndarray,
        traj: np.ndarray,
        sens_maps: np.ndarray,
        dc_weights: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        This method reconstructs the image from the k-space data using the NUFFT operator.
        Apply the dc_weights to kdata.
        Reconstruction is done by the adjoint NUFFT operator with orthonormal normalization.

        Args:
            kdata: k-space data. (shape of [Readout, Spokes, Channels])
            traj: Trajectory of the k-space data. (shape of [Readout, Spokes])
            sens_maps: Sensitivity maps of the k-space data. (shape of [Readout//2, Readout // 2, Channels])
            dc_weights: Density compensation weights. (shape of [Readout, Spokes])

        Returns:
            recon_NUFFT: Reconstructed image from the k-space data using the NUFFT operator. (absolute valued np.ndarray, shape of [Readout // 2, Readout // 2])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        get_nufft_adj_ob = kwargs.get("get_nufft_adj_ob", self.get_nufft_adj_ob)
        get_nufft_traj = kwargs.get("get_nufft_traj", self.get_nufft_traj)
        get_nufft_sens_maps = kwargs.get("get_nufft_sens_maps", self.get_nufft_sens_maps)
        get_nufft_kdata = kwargs.get("get_nufft_kdata", self.get_nufft_kdata)

        # Your code here ...
        im_size = None
        grid_size = None

        nufft_adj_ob = None

        nufft_kdata = None
        nufft_traj = None
        nufft_sm = None

        # Reconstruction
        recon_NUFFT = None

        return recon_NUFFT

    def calc_grad(
        self,
        nufft_ob: NUFFT_Forward_OP,
        nufft_adj_ob: NUFFT_Adjoint_OP,
        nufft_traj: torch.Tensor,
        nufft_sm: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """
        This method calculates the gradient of the objective function for the NUFFT operator.
        Preserve energy by using the orthonormal normalization between transforms.

        Args:
            nufft_ob: NUFFT operator
            nufft_adj_ob: Adjoint NUFFT operator
            nufft_traj: Trajectory of the k-space data for NUFFT operator. (shape of [2, Readout*Spokes])
            nufft_sm: Sensitivity maps of the k-space data for NUFFT operator. (shape of [Batch(1), Channels, Readout // 2, Readout // 2])
            u: Image to be reconstructed. (shape of [Batch(1), Channels, Readout // 2, Readout // 2])
            g: k-space data. (shape of [Batch(1), Channels, Readout*Spokes])

        Returns:
            grad: Gradient of the objective function for the NUFFT operator. (shape of [Batch(1), Channels, Readout // 2, Readout // 2])
        """
        # Your code here ...
        grad = None

        return grad

    def nufft_gd_recon(
        self,
        gt: np.ndarray,
        kdata: np.ndarray,
        traj: np.ndarray,
        sens_maps: np.ndarray,
        iter: Optional[int] = 100,
        step_size: Optional[int] = 0.01,
        **kwargs,
    ) -> Tuple[list, list, list]:
        """
        This method reconstructs the image from the k-space data using the NUFFT operator with the gradient descent method.
        See the lecture note p.15 for the gradient descent algorithm.

        Args:
            gt: Ground truth image to calculate RMSE. (shape of [Readout // 2, Readout // 2])
            kdata: k-space data. (shape of [Readout, Spokes, Channels])
            traj: Trajectory of the k-space data. (shape of [Readout, Spokes])
            sens_maps: Sensitivity maps of the k-space data. (shape of [Readout//2, Readout // 2, Channels])
            iter: Number of iterations for the gradient descent method. Default: 100
            step_size: Step size for the gradient descent method. Default: 0.01

        Returns:
            recons_gd: Reconstructed images from the k-space data using the NUFFT operator with the gradient descent method. (List of absolute valued np.ndarray, shape of [Readout // 2, Readout // 2])
            nmse_gd: Normalized mean squared error of the reconstructed images at each iteration. (shape of [iter])
            grad_norms: L2 norm of the gradient at each iteration. (shape of [iter])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        get_nufft_ob = kwargs.get("get_nufft_ob", self.get_nufft_ob)
        get_nufft_adj_ob = kwargs.get("get_nufft_adj_ob", self.get_nufft_adj_ob)
        get_nufft_traj = kwargs.get("get_nufft_traj", self.get_nufft_traj)
        get_nufft_sens_maps = kwargs.get("get_nufft_sens_maps", self.get_nufft_sens_maps)
        get_nufft_kdata = kwargs.get("get_nufft_kdata", self.get_nufft_kdata)
        calc_grad = kwargs.get("calc_grad", self.calc_grad)

        # Your code here ...
        N_ro, *_ = kdata.shape
        im_size = (N_ro // 2, N_ro // 2)
        grid_size = (N_ro, N_ro)

        nufft_ob = None
        nufft_adj_ob = None

        nufft_traj = None
        nufft_sm = None

        u = None
        g = None

        nmse_gd = []
        grad_norms = []
        recons_gd = []
        with tqdm(total=iter, unit="iter", leave=True) as pbar:
            for ii in range(iter):
                grad = None
                u = None

                grad_norm = torch.norm(grad).cpu().numpy()
                abs_u = np.abs(u.squeeze().cpu().numpy())
                nmse_gd.append(utils.calc_nmse(np.abs(gt), abs_u))
                grad_norms.append(grad_norm)

                pbar.set_description(desc=f"Iteration {ii: 3d}")
                pbar.set_postfix({"L2 norm": f"{grad_norm: .5f}"})
                pbar.update()
                recons_gd.append(abs_u)

        return recons_gd, nmse_gd, grad_norms

    def nufft_cg(
        self,
        nufft_ob: NUFFT_Forward_OP,
        nufft_adj_ob: NUFFT_Adjoint_OP,
        nufft_kdata: torch.Tensor,
        nufft_traj: torch.Tensor,
        nufft_sm: torch.Tensor,
        im_size: tuple,
        iter: int,
        tol: float,
        gt: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, list, list]:
        """
        Conjugate Gradient method using NUFFT operator for non-Cartesian k-space data.
        Preserve energy by using the orthonormal normalization between transforms.
        See the Appendix 1 in the exercise description for the conjugate gradient algorithm.

        Args:
            nufft_ob: NUFFT operator
            nufft_adj_ob: Adjoint NUFFT operator
            nufft_kdata: k-space data in the format that NUFFT operator can use. (shape of [Batch(1), Channels, Readout*Spokes])
            nufft_traj: Trajectory of the k-space data for NUFFT operator. (shape of [2, Readout*Spokes])
            nufft_sm: Sensitivity maps of the k-space data for NUFFT operator. (shape of [Batch(1), Channels, Readout // 2, Readout // 2])
            im_size: Size of image with length being the number of dimensions.
            iter: Number of iterations for the conjugate gradient method.
            tol: Tolerance for the residual norm to stop the iteration.
            gt: Ground truth image to calculate RMSE. (shape of [Readout // 2, Readout // 2])

        Returns:
            x_np: Reconstructed image from the k-space data using the NUFFT operator with the conjugate gradient method. (absolute valued np.ndarray, shape of [Readout // 2, Readout // 2])
            gifs: List of reconstructed images at each iteration. (List of absolute valued np.ndarray, shape of [Readout // 2, Readout // 2])
            nmse: Normalized mean squared error of the reconstructed images at each iteration. (shape of [iter])
        """
        # Your code here ...
        x = None
        r = None
        p = None
        rt_r = None

        gifs, nmse = [], []
        with tqdm(total=iter, unit="iter", leave=True) as pbar:
            for ii in range(iter):
                Ap = None
                alpha = None
                x = None
                r = None
                if torch.sqrt(rt_r_new) < tol:
                    break
                rt_r_new = None
                beta = None
                p = None

                # update rt_r for the next iteration
                rt_r = rt_r_new

                pbar.set_description(desc=f"Iteration {ii: 3d}")
                pbar.set_postfix({"residual norm": np.sqrt(rt_r.cpu().numpy())})
                pbar.update()

                x_np = np.abs(x.squeeze().cpu().numpy())
                gifs.append(x_np)
                if gt is not None:
                    nmse.append(utils.calc_nmse(abs(gt), x_np))

        return x_np, gifs, nmse

    def cg_sense(
        self,
        kdata: np.ndarray,
        traj: np.ndarray,
        sens_maps: np.ndarray,
        gt: Optional[np.ndarray] = None,
        iter: Optional[int] = 50,
        tol: Optional[float] = 1e-6,
        title: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, list]:
        """
        Reconstruct subsampled PMRI data using CG SENSE [1]
        uses M. Muckley's torchkbnufft: https://github.com/mmuckley/torchkbnufft

        Args:
            kdata: k-space data. (shape of [Readout, Spokes, Channels])
            traj: Trajectory of the k-space data. (shape of [Readout, Spokes])
            sens_maps: Sensitivity maps of the k-space data. (shape of [Readout//2, Readout // 2, Channels])
            gt: Ground truth image to calculate RMSE. (shape of [Readout // 2, Readout // 2])
            iter: Number of iterations for the conjugate gradient method. Default: 50
            tol: Tolerance for the residual norm to stop the iteration. Default: 1e-6
            title: Title for the gif. Default: None

        Returns:
            x: Reconstructed image from the k-space data using the CG SENSE method. (absolute valued np.ndarray, shape of [Readout // 2, Readout // 2])
            nmse: Normalized mean squared error of the reconstructed images at each iteration. (shape of [iter])
        """
        # PLEASE IGNORE HERE AND DO NOT MODIFY THIS PART.
        get_nufft_ob = kwargs.get("get_nufft_ob", self.get_nufft_ob)
        get_nufft_adj_ob = kwargs.get("get_nufft_adj_ob", self.get_nufft_adj_ob)
        get_nufft_traj = kwargs.get("get_nufft_traj", self.get_nufft_traj)
        get_nufft_sens_maps = kwargs.get("get_nufft_sens_maps", self.get_nufft_sens_maps)
        get_nufft_kdata = kwargs.get("get_nufft_kdata", self.get_nufft_kdata)
        nufft_cg = kwargs.get("nufft_cg", self.nufft_cg)

        N_ro, *_ = kdata.shape
        im_size = (N_ro // 2, N_ro // 2)
        grid_size = (N_ro, N_ro)

        nufft_ob = get_nufft_ob(im_size, grid_size)
        nufft_adj_ob = get_nufft_adj_ob(im_size, grid_size)

        nufft_traj = get_nufft_traj(traj)
        nufft_sm = get_nufft_sens_maps(sens_maps)
        nufft_kdata = get_nufft_kdata(kdata)

        x, gifs, nmse = nufft_cg(
            nufft_ob=nufft_ob,
            nufft_adj_ob=nufft_adj_ob,
            nufft_kdata=nufft_kdata,
            nufft_traj=nufft_traj,
            nufft_sm=nufft_sm,
            im_size=im_size,
            iter=iter,
            tol=tol,
            gt=gt,
        )

        if title:
            utils.create_gif(gifs, title)

        return x, nmse


if __name__ == "__main__":
    # %% Load modules
    # This import is necessary to run the code cell-by-cell
    from lab07 import *

    # %%
    op = Lab07_op()
    kdata, sens_maps, traj, dc_weights, gt = op.load_data()
