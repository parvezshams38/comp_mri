"""
Computational Magnetic Resonance Imaging (CMRI) 2024/2025 Winter semester

- Author          : Jinho Kim
- Email           : <jinho.kim@fau.de>
"""

import numpy as np
import utils


class Lab01_op:
    """
    label: Label map of the digital brain phantom
        - 1: Cerebrospinal fluid (CSF)
        - 2: Gray matter (GM)
        - 3: White matter (WM)
    T1_map: Predefined T1 Values
    T2_map: Predefined T2 Values
    PD_map: Proton Density Values
    """

    def __init__(self):
        # Define TR and TE as a pair in a list [TR, TE]
        self.PDw_TRTE = [2000, 50]  # Task 2.2
        self.T1w_TRTE = [300, 50]  # Task 2.3
        self.T2w_TRTE = [2000,100]  # Task 2.4

    def load_data(self, path="digital_brain_phantom.mat"):
        mat = utils.load_data(path)
        self.label = mat["ph"]["label"][0][0]  # (128, 128)
        self.T1_map = mat["ph"]["t1"][0][0]  # (128, 128)
        self.T2_map = mat["ph"]["t2"][0][0]  # (128, 128)
        self.PD_map = mat["ph"]["sd"][0][0]  # (128, 128)

    def get_csf_mask(self):
        """
        Get CSF mask from the label map
        Return:
            A mask of CSF, shape: (128, 128)
        """
        mask = np.where(self.label == 1, 1, 0)
        return mask

    def get_gm_mask(self):
        """
        Get GM mask from the label map
        Return:
            A mask of GM, shape: (128, 128)
        """
        # Your code here ...

        return np.where(self.label==2,1,0)

    def get_wm_mask(self):
        """
        Get WM mask from the label map
        Return:
            A mask of WM, shape: (128, 128)
        """
        # Your code here ...

        return np.where(self.label==3,1,0)

    def get_T1(self, target):
        """
        Returns the T1 value of the target region

        Args:
            target: A mask of the target region (CSF, GM, WM), shape: (128, 128)
        Return:
            T1 value of the target region in [ms]
        """
        # Your code here ...
        # Find the indices of non-zero elements
        non_zero_indices = np.nonzero(target)

        # Get the index of the first non-zero element

        T1_ms = self.T1_map[non_zero_indices[0][0],non_zero_indices[1][0]]
        return T1_ms

    def get_T2(self, target):
        """
        Returns the T2 value of the target region

        Args:
            target: A mask of the target region (CSF, GM, WM), shape: (128, 128)
        Return:
            T2 value of the target region in [ms]
        """
        # Your code here ...
        non_zero_indices = np.nonzero(target)

        # Get the index of the first non-zero element

        T2_ms = self.T2_map[non_zero_indices[0][0], non_zero_indices[1][0]]
        return T2_ms

    def get_PD(self, target):
        """
        Returns the PD value of the target region

        Args:
            target: A mask of the target region (CSF, GM, WM), shape: (128, 128)
        Return:
            PD value of the target region
        """
        # Your code here ...
        non_zero_indices = np.nonzero(target)

        # Get the index of the first non-zero element

        PD = self.PD_map[non_zero_indices[0][0], non_zero_indices[1][0]]
        return PD

    def spin_echo(self, TR, TE):
        """
        Simulate a spin echo sequence

        Args:
            TR: Repetition time in [ms]
            TE: Echo time in [ms]
        Return:
            A 2D image of the spin echo sequence, shape: (128, 128)
        """
        # Your code here ...

        SE = np.zeros((128,128))
        for i in range(128):
            for j in range(128):
                T1 = self.T1_map[i,j]
                T2 = self.T2_map[i,j]
                m = self.PD_map[i,j]
                SE[i,j] = m*(1-2*np.exp(-(TR-TE/2)/T1)+np.exp(-TR/T1))*np.exp(-TE/T2)
        return SE


if __name__ == "__main__":
    # %% Load modules
    # This import is necessary to run the code cell-by-cell
    from lab01 import *

    # %% Define the object
    op = Lab01_op()

    # %% 1.1.	Load the file digital_brain_phantom.mat by calling load_data method.
    op.load_data()
    csf_mask = op.get_csf_mask()
    gm_mask = op.get_gm_mask()
    wm_mask = op.get_wm_mask()
    #utils.imshow([csf_mask, gm_mask, wm_mask])

    print(op.get_T1(csf_mask), op.get_T1(gm_mask), op.get_T1(wm_mask))
    print(op.get_T2(csf_mask), op.get_T2(gm_mask), op.get_T2(wm_mask))
    print(op.get_PD(csf_mask), op.get_PD(gm_mask), op.get_PD(wm_mask))


    t1 = op.spin_echo(op.T1w_TRTE[0], op.T1w_TRTE[1])
    t2 = op.spin_echo(op.T2w_TRTE[0], op.T2w_TRTE[1])
    t3 = op.spin_echo(op.PDw_TRTE[0], op.PDw_TRTE[1])

    utils.imshow([t3, t1, t2])
