""" Performs unit tests on my_ib_color_model methods."""

# Imports
import unittest
# import pytest
import copy
import numpy as np
from ib_from_scratch.my_ib_color_model import klDiv, IBModel


class TestIBModel(unittest.TestCase):

    def test_klDiv_function(self, p, q):

        # Get KL-Divergence using the old (slow) method
        true_tot = 0.0
        for i in range(len(p)):
            p1 = p[i]
            p2 = q[i]
            val = p1*np.log2(p1/p2)
            true_tot += val

        # Get KL-Divergence using the updated function
        tot = klDiv(p, q)

        # Check function results against baseline
        self.assertEqual(tot, true_tot)


    def test_random_init_method(self, u_vals, u_coords, meanings, words):

        # Initialize ib model
        ibm = IBModel(observations=u_vals, coordinates=u_coords, meanings=meanings, words=words)

        # Get fitted encoder values using (slow) baseline method
        true_fitted_encoder = copy.deepcopy(ibm.fitted_encoder)
        for i in range(true_fitted_encoder.shape[0]):
            true_fitted_encoder[i, :] = true_fitted_encoder[i, :] / np.sum(true_fitted_encoder[i, :])

        # Get the fitted decoder values using (slow) baseline method
        true_fitted_decoder = copy.deepcopy(ibm.fitted_decoder)
        for wi in range(ibm.words.shape[0]):
            for mi in range(ibm.meanings.shape[0]):
                factor1 = true_fitted_encoder[mi, wi]
                factor2 = ibm.p_m[mi]
                divisor = np.sum(ibm.p_m * true_fitted_encoder[:, wi])
                true_fitted_decoder[wi, mi] = (factor1 * factor2) / divisor

        # Get the p(w) values using (slow) baseline method
        true_p_w = copy.deepcopy(ibm.p_w)
        for wi in range(ibm.words.shape[0]):
            qvec = true_fitted_encoder[:, wi]
            prob = np.sum(qvec*ibm.p_m)
            true_p_w[wi] = prob

        # Get the m_w(u) values using (slow) baseline method
        true_fitted_mu_w = copy.deepcopy(ibm.fitted_mu_w)
        for wi in range(ibm.words.shape[0]):
            for ui in range(ibm.u_vals.shape[0]):
                u_tot = 0.0
                for mi in range(ibm.meanings.shape[0]):
                    factor1 = ibm.meanings[mi,ui]
                    factor2 = true_fitted_decoder[wi,mi]
                    u_tot += factor1 * factor2
                true_fitted_mu_w[wi, ui] = u_tot

        # Get ib model results
        ibm.randomly_initialize(rseed=42, uniform=False)

        # Check results against baseline values
        self.assertEqual(ibm.fitted_encoder, true_fitted_encoder)
        self.assertEqual(ibm.fitted_decoder, true_fitted_decoder)
        self.assertEqual(ibm.p_w, true_p_w)
        self.assertEqual(ibm.fitted_mu_w, true_fitted_mu_w)

    def test_single_update_step(self, u_vals, u_coords, meanings, words, beta):

        # Initialize ib model
        ibm = IBModel(observations=u_vals, coordinates=u_coords, meanings=meanings, words=words)
        ibm.randomly_initialize(rseed=42, uniform=False)

        # Get initial values for distributions
        true_p_w = copy.deepcopy(ibm.p_w)
        true_fitted_mu_w = copy.deepcopy(ibm.fitted_mu_w)
        true_fitted_encoder = copy.deepcopy(ibm.fitted_encoder)
        true_fitted_decoder = copy.deepcopy(ibm.fitted_decoder)

        # Get p_w update using baseline
        for wi in range(ibm.words.shape[0]):
            qvec = ibm.fitted_encoder[:,wi]
            prob = np.sum(qvec*ibm.p_m)
            true_p_w[wi] = prob
        # Get p_w update using updated method
        ibm.update_p_w()

        # Get mu_w update using baseline
        for wi in range(ibm.words.shape[0]):
            for ui in range(ibm.u_vals.shape[0]):
                u_tot = 0.0
                for mi in range(ibm.meanings.shape[0]):
                    factor1 = ibm.meanings[mi, ui]
                    factor2 = ibm.fitted_decoder[wi, mi]
                    u_tot += factor1 * factor2
                true_fitted_mu_w[wi, ui] = u_tot
        # Get mu_w update using updated method
        ibm.update_mu_w()

        # Get encoder update using only method
        ibm.update_encoder(beta)

        # Get decoder update using baseline
        for wi in range(ibm.words.shape[0]):
            for mi in range(ibm.meanings.shape[0]):
                factor1 = ibm.fitted_encoder[mi, wi]
                factor2 = ibm.p_m[mi]
                divisor = np.sum(ibm.p_m * ibm.fitted_encoder[:, wi])
                true_fitted_decoder[wi, mi] = (factor1 * factor2) / divisor
        # Get decoder update using updated method
        ibm.update_decoder()

        # Check results against baseline values
        self.assertEqual(ibm.p_w, true_p_w)
        self.assertEqual(ibm.fitted_mu_w, true_fitted_mu_w)
        self.assertEqual(ibm.fitted_decoder, true_fitted_decoder)


if __name__ == '__main__':
    unittest.main()

    # beta = 1.05
    # u_vals = np.random.Generator.integers(low=1, high=11, size=10)
    # u_coords = np.random.Generator.random(size=(10, 3))
    # words = np.array(['w1', 'w2', 'w3'])
    # meanings = generate_WCS_meanings(perceptual_variance=64)?????
