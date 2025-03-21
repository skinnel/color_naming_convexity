"""My updated implementation of the IB color model, which borrows some ideas from the Embo repository
(https://gitlab.com/epiasini/embo) in order to deal with convergence issues introduced by numerical sensitivity, which
can cause the IB algorithm to converge on suboptimal local minima. Difference with previous attempts is that this
attempt is only capable of running in sequence and uses beta annealing when finding solutions.

@authors Lindsay Skinner (skinnel@uw.edu)
"""

import numpy as np
import pickle as pkl
import pandas as pd
import multiprocessing as mp
from typing import Tuple, Optional, Union, List, Dict
from ib_from_scratch.ib_utils import (get_entropy, get_kl_divergence, get_mutual_information, compute_upper_bound,
                                      get_joint_dist, joint, marginal)
from copy import deepcopy


# Define helper functions to update probability distributions during IB iterations
# Idea to separate these from the class comes from Embo

# Calculate the log-loss value across each element
def element_log_loss(qw: np.array, pu_m: np.array, mwu: np.array, beta: float) -> float:
    """Calculates the log-loss value for each observation used to define the distribution.

    Parameters
    ----------
    qw
        q_beta(w), the marginal distribution of word w. This is pm in embo.
    pu_m
        p(u|m), the distribution of u (colors) conditioned on meaning m. This is pyx_c in embo.
    mwu
        m^_w(u), the distribution of u (colors) conditioned on word w. This is pym_c in embo.
    beta
        The beta value used to weight the loss calculation.

    Returns
    -------
    ll_val
        The log-loss value.
    """

    ll_val = (np.log(qw[:, np.newaxis]) - beta * get_kl_divergence(p=pu_m[:, np.newaxis, :], q=mwu[:, :, np.newaxis]))
    #ll_val = (np.log(qw) - beta * get_kl_divergence(p=pu_m, q=mwu))

    return ll_val


def update_encoder(qw: np.array, pu_m: np.array, mwu: np.array, beta: float) -> Tuple[np.array, float]:
    """Updates the encoder in the IB iteration process. This is p_mx_c in embo.

    Parameters
    ----------
    qw
        q_beta(w), the marginal distribution of word w. This is pm in embo.
    pu_m
        p(u|m), the distribution of u (colors) conditioned on meaning m. This is pyx_c in embo.
    mwu
        m^_w(u), the distribution of u (colors) conditioned on word w. This is pym_c in embo.
    beta
        The beta value used to weight the loss calculation.

    Returns
    -------
    encoder
        The encoder, q(w|m), updated during the most recent iteration.
    z
        The normalization factor applied to the encoder.
    """
    encoder = np.exp(element_log_loss(qw, pu_m, mwu, beta))
    z = encoder.sum(axis=0)
    encoder /= z  # normalize

    return encoder, z


# Update q(m^|w) - Not necessary during the iteration process, so just do this once convergence has occurred.
def update_decoder(qw: np.array, pm: np.array, encoder: np.array) -> np.array:
    """Updates the decoder q(m|w).

    Parameters
    ----------
    qw
        q_beta(w), the marginal distribution of word w. This is pm in embo.
    pm
        p(m), the marginal distribution of meaning m. This is px in embo.
    encoder
        q_beta(w|m), the distribution of w (words) conditioned on meaning m. This is pmx_c in embo.

    Returns
    -------
     decoder
        The decoder derived from the provided encoder and the assumption of an ideal Bayesian listener.
    """
    # num = pm[:, np.newaxis] @ encoder
    # qmw = num / qw[:, np.newaxis]
    # return qmw

    puw = pm[np.newaxis, :] @ encoder.T
    decoder = puw / qw[np.newaxis, :]

    # # TODO: adjust this with the below (SI pg 3)
    # puw = encoder @ pm[np.newaxis, :]
    # decoder = puw / qw[:, np.newaxis]

    return decoder


def update_mwu(qw: np.array, pm: np.array, pu_m: np.array, encoder: np.array) -> np.array:
    """Updates m^_w(u) in the IB iteration process. This is p_ym_c in embo.

    Parameters
    ----------
    qw
        q_beta(w), the marginal distribution of word w. This is pm in embo.
    pm
        p(m), the marginal distribution of meaning m. This is px in embo.
    pu_m
        p(u|m), the distribution of u (colors) conditioned on meaning m. This is pyx_c in embo.
    encoder
        q_beta(w|m), the distribution of w (words) conditioned on meaning m. This is pmx_c in embo.

    Returns
    -------
     mwu
        The beta-specific distribution, m^_w(w), updated during the most recent iteration.
    """
    # TODO: try switching this out so its sum_m decoder * m(u) (SI pg 3)
    # note that to get m(u) we do p(u|m)*p(m)
    mu = pu_m * pm[np.newaxis, :]
    puw = mu @ encoder.T
    #puw = pu_m * pm[np.newaxis, :] @ encoder.T
    mwu = puw / qw[np.newaxis, :]

    # # note that to get m(u) we do p(u|m)*p(m)
    # mu = pu_m * pm[np.newaxis, :]
    # puw = mu @ decoder
    # mwu = puw / qw[np.newaxis, :]

    # pym = pyx_c*px[np.newaxis,:] @ pmx_c.T
    # pym_c = pym / pm[np.newaxis,:]


    return mwu


def update_qw(pm: np.array, encoder: np.array) -> np.array:
    """Updates q_beta(w) in the IB iteration process. This is p_m in embo.

    Parameters
    ----------
    pm
        p(m), the marginal distribution of meaning m. This is px in embo.
    encoder
        q_beta(w|m), the distribution of words (w) conditioned on meaning m. This is pmx_c in embo.

    Returns
    -------
    qw
        The updated distribution of words, w. .
    """
    qw = encoder @ pm

    return qw

# TODO: may have wrong definition of p(u) - test
# def update_pu(mwu, qw, encoder, meanings):
#
#     # # Get denominator
#     # # TODO: check if we need pm in here to weight things? Issues may not show with pm uniform
#     # denom = encoder @ meanings
#     #
#     # # Get numerator
#     # # num = mwu * qw[:, np.newaxis]
#     # num = mwu * qw[np.newaxis, :]
#     #
#     # pu = num / denom.T
#     # pu = pu.sum(axis=1)
#
#     # pwu = joint(pY_X=mwu.T, pX=qw)
#     # pu = marginal(pXY=pwu, axis=0)  # 0=y, 1=x
#
#     # get p(u) from meanings i.e. m(u) = p(u|m)
#     pu = np.sum(meanings, axis=0)
#     pu /= pu.sum()
#
#     return pu



# Function to remove unused dimensions, which can cause issues with NaNs and infinite values
def drop_unused_dimensions(qw, encoder, mwu, eps: float = 0.0):
    """ Taken directly from embo. Removes bottleneck dimensions that are not in use anymore.

    This is in general safe to do as once a dimension is excluded by
    the algorithm, it is never brought back. It is necessary to use
    this function particularly when using the generalized bottleneck,
    where dimensions are discarded very aggressively, and if keeping
    them we would need to put special safeguards in place to avoid
    NaNs and infinities in the Dkl and when computing p(y|m).

    """
    unused_qw = qw <= eps
    unused_encoder = np.all(encoder <= eps, axis=1)
    unused_mwu = np.all(mwu <= eps, axis=0)
    unused = np.any(np.vstack([unused_qw, unused_encoder.T, unused_mwu]), axis=0)
    in_use = ~unused
    return qw[in_use], encoder[in_use, :], mwu[:, in_use]


class IBModel:

    def __init__(self, observations, coordinates, meanings, words, pu=None):

        self.u_vals: np.array = observations
        self.u_coords: np.array = coordinates
        self.meanings: np.array = meanings
        self.words: np.array = words

        # Flag to indicate whether IB annealing has already been performed
        self.fitted = False

        # Initialize accuracy, complexity and ib objective function values
        self.accuracy: np.array = None
        self.complexity: np.array = None
        self.beta_vals: np.array = None
        self.obj_fun_val: np.array = None

        # TODO: explore other linguistic need options - also is this the same pm as below?
        # Assumes uniform distribution for linguistic need
        self.pm: np.array = np.ones(meanings.shape[0]) / meanings.shape[0]

        #Assumed to be a 1D array of length |words|
        #q_beta(w)
        # self.qw = np.zeros(words.shape[0])
        self.qw = np.ones(words.shape[0]) / words.shape[0]

        #Assumed to be 2D with 'meaning' on axis 0 and 'word' on axis 1
        #q(w|m)
        # self.encoder = np.zeros((meanings.shape[0], words.shape[0]))
        self.encoder = np.ones((meanings.shape[0], words.shape[0])) / words.shape[0]

        #Assumed to be 2D with 'word' on axis 0 and 'meaning' on axis 1
        #q(m^|w)
        # self.decoder = np.zeros((words.shape[0], meanings.shape[0]))
        self.decoder = np.ones((words.shape[0], meanings.shape[0])) / meanings.shape[0]

        #Assumed to be 2D with 'word' on axis 0 and 'u_vals' on axis 1
        #m^_w(u)
        # self.mwu = np.zeros((words.shape[0], observations.shape[0]))
        self.mwu = np.ones((words.shape[0], observations.shape[0])) / observations.shape[0]

        # TODO: check if meanings should be joint dist of u and m or just dist over m
        #Needed for update steps
        # self.pum_joint = get_joint_dist(self.meanings, self.u_vals, axis=0)
        # self.pm = self.pum_joint.sum(axis=1)
        # self.pu = self.pum_joint.sum(axis=0)
        # self.pu_m = self.pum_joint.T / self.pm
        #
        # TODO: pu should be distribution over observation space - try uniform at first
        #self.pum_joint = meanings / meanings.sum()
        #
        # self.pu = self.pum_joint.sum(axis=0)
        #
        # self.pu = self.mwu.sum(axis=1)
        # # Notes: mwu = [words, uvals]; qw = [words]
        # # self.pu = self.mwu.T @ self.qw
        # # self.pu = self.qw @ self.mwu.T
        # #self.pu = np.ones(observations.shape[0]) / observations.shape[0]
        # self.pu = update_pu(self.mwu, self.qw, self.encoder, self.meanings)
        #
        # pu = np.sum(meanings, axis=0)
        # self.pu = pu / pu.sum()
        #
        if pu is None:
            self.pu = np.ones(observations.shape[0]) / observations.shape[0]
        else:
            self.pu = pu.flatten()
        #
        #self.pu = self.pum_joint.sum(axis=0)
        #self.pu = self.pu / self.pu.sum()
        #self.pu_m = self.pum_joint.T / self.pm
        self.pu_m = meanings

        # Saves the encoders learned during the iterative process
        self.saved_encoders = {'beta': [],
                               'ib_val': [],
                               'encoder': []}

        # Save the best version of model distributions for reference during annealing steps
        self.best_vals = {
            'beta': None,
            'IB': None,
            'encoder': None,
            'qw': None,
            'mwu': None,
            'pu': None,
            #'decoder': None
        }

    def randomly_initialize(self):
        """Assigns random distributions to q_beta(w) and m^_w(u) in order to initialize the iteration process that
        optimizes the IB tradeoff.
        """

        # Initialize q_beta(w)
        self.qw = np.random.rand(self.qw.shape[0]) + 1
        self.qw /= self.qw.sum()

        # Initialize m^_w(u)
        self.mwu = np.random.rand(self.mwu.shape[0], self.mwu.shape[1]) + 1
        self.mwu /= self.mwu.sum(axis=0)

        # TODO: confirm removal of this with testing - thinking p(u) shouldn't change
        # # Update pu accordingly
        # # self.pu = self.mwu.sum(axis=1)
        # # self.pu = self.mwu.T @ self.qw
        # # self.pu = self.qw @ self.mwu.T
        # self.pu = update_pu(self.mwu, self.qw, self.encoder, self.meanings)

    def randomly_initialize_existing_start(self):
        """Assigns random noise to the existing q_beta(w) and m^_w(u) distributions in order to initialize the iteration
        process that optimizes the IB tradeoff when using annealing.
        """

        # Initialize q_beta(w)
        qw_delta = np.random.rand(self.qw.shape[0]) + 1
        qw_delta = qw_delta / (100 * self.qw.mean())
        self.qw = self.qw + qw_delta
        self.qw /= self.qw.sum()

        # Initialize m^_w(u)
        mwu_delta = np.random.rand(self.mwu.shape[0], self.mwu.shape[1]) + 1
        mwu_delta = mwu_delta / (100 * self.mwu.mean())
        self.mwu = self.mwu + mwu_delta
        self.mwu /= self.mwu.sum(axis=0)

        # TODO: confirm removal of this with testing - thinking p(u) shouldn't change
        # # Update pu accordingly
        # # self.pu = self.mwu.sum(axis=1)
        # # self.pu = self.mwu.T @ self.qw
        # # self.pu = self.qw @ self.mwu.T
        # self.pu = update_pu(self.mwu, self.qw, self.encoder, self.meanings)

    # Single iteration update
    def run_single_annealing_iteration(self, beta: float = 1.0):
        """ Performs a single step of the iteration process used to find the optimal complexity-informativeness tradeoff
        for a given beta value.

        Parameters
        ----------
        beta
            The beta value for which the iterative process is being run

        """

        # Step 1: update encoder
        self.encoder, z = update_encoder(qw=self.qw, pu_m=self.pu_m, mwu=self.mwu, beta=beta)

        # (from embo) remove unused dimensions
        self.qw, self.encoder, self.mwu = drop_unused_dimensions(self.qw, self.encoder, self.mwu, eps=0.00001)

        # Step 2: update q_beta(w)
        self.qw = update_qw(pm=self.pm, encoder=self.encoder)

        # (from embo) remove unused dimensions
        self.qw, self.encoder, self.mwu = drop_unused_dimensions(self.qw, self.encoder, self.mwu, eps=0.00001)

        # Step 3: update m^_w(u)
        self.mwu = update_mwu(qw=self.qw, pm=self.pm, pu_m=self.pu_m, encoder=self.encoder)
        #self.mwu = update_mwu(qw=self.qw, pm=self.pm, pu_m=self.pu_m, decoder=self.decoder)

        # (from embo) again, remove unused dimensions
        self.qw, self.encoder, self.mwu = drop_unused_dimensions(self.qw, self.encoder, self.mwu, eps=0.00001)

        # TODO: confirm removal of this with testing - thinking p(u) shouldn't change
        # # Update pu accordingly
        # # self.pu = self.mwu.sum(axis=1)
        # # self.pu = self.mwu.T @ self.qw
        # # self.pu = self.qw @ self.mwu.T
        # self.pu = update_pu(self.mwu, self.qw, self.encoder, self.meanings)

        # Update decoder
        #self.decoder = update_decoder(self.qw, self.pm, self.encoder)

    # Beta-specific iteration with annealing
    def run_beta_specific_iterations_with_annealing(
            self, beta: float, attempts: int, max_iter: int, rtol: float = 1e-3, atol: float = 0.0,
            return_best: bool = True, verbose: bool = False, save_encoders: bool = True
    ) -> Union[List[float], List[Dict[str, float]]]:
        """Finds the optimal IB tradeoff for the provided beta value.

        Parameters
        ----------
        beta
            The beta value for which the iterative process is being run.
        attempts
            The number of randomly initialized starting points used in training. We use multiple starting points to
            lower the risk of becoming trapped in a local minima.
        max_iter
            The maximum number of iterations to run for a given attempt.
        rtol
            The relative difference tolerance used to determine whether the iterative process has converged or not.
        atol
            The absolute difference tolerance used to determine whether the iterative process has converged or not.
        return_best
            If true the single best attempt is returned. If false, the results for all starting attempts are returned.
        verbose
            If true then indicators for the start of each attempt and a status update indicating whether the attempt
            converged a printed to standard output.
        save_encoders
            If 'all' then the encoders, beta values and ib values from all attempts are saved on the model. If 'best'
            then only those values and encoders from the best run are saved. If None then no values are saved.

        Returns
        -------
        best_res
            The single best result observed from the various starting attempts for a given beta value. Includes
            MI(meanings), MI(observations), Entropy, IB value and Beta.
        """

        # Save initial starting state, to be used in each attempt
        original_encoder = deepcopy(self.encoder)
        original_qw = deepcopy(self.qw)
        original_mwu = deepcopy(self.mwu)
        #original_pu = deepcopy(self.pu)
        #original_decoder = deepcopy(self.decoder)

        # Restart with a new random initialization for the specified number of attempts
        attempt_results = []
        if save_encoders == 'best':
            attempt_encoders = []
        for a in range(attempts):

            # Print attempt info
            if verbose:
                print(f'Processing beta={beta} attempt number {a}')

            # Use original state of distributions for start of each attempt
            if a > 0:
                self.encoder = deepcopy(original_encoder)
                self.qw = deepcopy(original_qw)
                self.mwu = deepcopy(original_mwu)
                #self.pu = deepcopy(original_pu)
                #self.decoder = deepcopy(original_decoder)
            # Randomize the starting points - keep near the starting solutions
            self.randomly_initialize_existing_start()

            # Iterate until convergence or maximum allotment of iterations is reached
            i = 0
            converged = False
            old_ib_val = None  # Should be unnecessary, but included just in case
            while i < max_iter and not converged:

                try:
                    # Perform iteration
                    # TODO: double check this
                    self.run_single_annealing_iteration(beta)
                    #ib_val = get_entropy(self.qw) - beta * get_mutual_information(self.pu, self.qw, self.mwu)
                    ib_val = (get_mutual_information(self.qw, self.pm, self.encoder) -
                              beta * get_mutual_information(self.pu, self.qw, self.mwu))
                    # ib_val = (get_mutual_information(self.qw, self.pm, self.encoder) -
                    #           beta * get_mutual_information(self.mwu, self.qw, self.decoder))
                    #
                    # mi1 = get_mutual_information(self.qw, self.pm, self.encoder)
                    # mi2_joint = joint(self.mwu, self.qw)
                    # mi2 = MI(mi2_joint)
                    # ib_val = mi1 - beta * mi2

                    # Check for convergence
                    if i > 0 and np.allclose(ib_val, old_ib_val, rtol, atol):
                        converged = True

                    # Update IB function results and iterator
                    old_ib_val = ib_val
                    i += 1

                    # Save results, if applicable
                    if save_encoders == 'all':
                        self.saved_encoders['beta'].append(beta)
                        self.saved_encoders['ib_val'].append(ib_val)
                        self.saved_encoders['encoder'].append(self.encoder.copy())

                except:
                    self.randomly_initialize_existing_start()
                    print(f'Error encountered for beta={beta} attempt {a} iteration {i}')
                    i += 1


            # Add converged results to list
            attempt_results.append({
                'MI meanings': get_mutual_information(self.qw, self.pm, self.encoder),
                'MI obs': get_mutual_information(self.pu, self.qw, self.mwu),
                #'MI obs': get_mutual_information(self.mwu, self.qw, self.decoder),
                'Entropy': get_entropy(self.qw),
                'IB': ib_val
            })

            # Add encoder results to list
            if save_encoders == 'best':
                attempt_encoders.append({
                    'Encoder': self.encoder.copy(),
                    'IB': ib_val
                })

            # Save results if best for this beta
            if self.best_vals['beta'] == beta:
                if self.best_vals['IB'] > ib_val:
                    self.best_vals['IB'] = ib_val
                    self.best_vals['encoder'] = deepcopy(self.encoder)
                    self.best_vals['qw'] = deepcopy(self.qw)
                    self.best_vals['mwu'] = deepcopy(self.mwu)
                    #self.best_vals['pu'] = deepcopy(self.pu)
                    #self.best_vals['decoder'] = deepcopy(self.decoder)
            else:
                self.best_vals['beta'] = beta
                self.best_vals['IB'] = ib_val
                self.best_vals['encoder'] = deepcopy(self.encoder)
                self.best_vals['qw'] = deepcopy(self.qw)
                self.best_vals['mwu'] = deepcopy(self.mwu)
                #self.best_vals['pu'] = deepcopy(self.pu)
                #self.best_vals['decoder'] = deepcopy(self.decoder)

            # Indicate whether results converged
            if verbose and converged:
                print(f'Attempt {a} for beta={beta} converged in {i} iterations')

        # Save results, if applicable
        if save_encoders == 'best':
            enc_res = min(attempt_encoders, key=lambda x: x['IB'])
            self.saved_encoders['beta'].append(beta)
            self.saved_encoders['ib_val'].append(enc_res['IB'])
            self.saved_encoders['encoder'].append(enc_res['Encoder'])

        # Ensure best distributions are retained for annealing purposes
        self.encoder = deepcopy(self.best_vals['encoder'])
        self.qw = deepcopy(self.best_vals['qw'])
        self.mwu = deepcopy(self.best_vals['mwu'])
        #self.pu = deepcopy(self.best_vals['pu'])
        #self.decoder = deepcopy(self.best_vals['decoder'])

        # If specified, pick best of attempted results
        if return_best:
            best_res = min(attempt_results, key=lambda x: x['IB'])
            m_mi = best_res['MI meanings']
            u_mi = best_res['MI obs']
            h_m = best_res['Entropy']
            ib = best_res['IB']

            # TODO: remove this after testing
            best_ib = self.best_vals['IB']
            best_enc = self.best_vals['encoder']
            best_qw = self.best_vals['qw']
            best_mwu = self.best_vals['mwu']
            #best_pu = self.best_vals['pu']
            #best_decoder = self.best_vals['decoder']
            check_best_res = {
                'MI meanings': get_mutual_information(best_qw, self.pm, best_enc),
                #'MI obs': get_mutual_information(best_pu, best_qw, best_mwu),
                'MI obs': get_mutual_information(self.pu, best_qw, best_mwu),
                #'MI obs': get_mutual_information(best_mwu, best_qw, best_decoder),
                'Entropy': get_entropy(best_qw),
                'IB': best_ib
            }
            for k, v in check_best_res.items():
                print(f'{k}: {v}')

            return [m_mi, u_mi, h_m, ib, beta]

        # Otherwise, return all results
        else:
            return attempt_results


    # Perform IB Annealing across all beta values
    def run_ib(self, n_beta: int, min_beta: float, max_beta: float, n_restarts: Optional[int], rtol: Optional[float],
               max_iter: Optional[int], ensure_monotonic: bool = False, verbose: bool = True,
               save_encoders: str = 'best'
               ) -> Tuple[np.array, np.array, np.array, np.array, np.array, float, float]:
        """Performs IB annealing until convergence across all beta values.

        Current approach is taken from embo. Note that this process randomly initializes the results for each beta
        value; results are not pre-seeded with the optimal results from a neighboring beta, as is specified in the IB
        color paper.

        Arguments:
        -----------
        n_beta
            The number of beta values over which to perform the annealing process. The beta values will be evenly spaced
            between the specified min and max beta value.
        min_beta
            The smalled beta value for which to perform the annealing process.
        max_beta
            The largest beta value for which to perform the annealing process.
        preseed_betas
            When true, optimal
        n_restarts
            The number of times the optimization procedure should be restarted (for each value of beta) from different
            random initial conditions. (Taken from Embo.)
        rtol
            The maximum allowable difference between annealing iterations for which we would conclude that the process
            has converged. If left blank, the annealing process will proceed for max_iter iterations.
        max_iter
            The maximum number of iterations to run the annealing process. If max_iter is reached before convergence
            then the process halts. If left blank, the annealing process will proceed until the difference between
            iterations is less than thresh.
        n_threads
            The number of cpu threads to run concurrently. Only applied if
        ensure_monotonic
            If true, drops the beta values that violate monotonicity constraints under the assumption that the solutions
            for those values converged on local minima, which prevented the algorithm from achieving the global optimal
            solution.
        verbose
            If true then indicators for the start of each attempt and a status update indicating whether the attempt
            converged a printed to standard output.
        save_encoders
            If 'all' then the encoders, beta values and ib values from all attempts are saved on the model. If 'best'
            then only those values and encoders from the best run are saved. If None then no values are saved.

        Returns:
        --------
        complexity_vec
            IB-optimal complexity values, I(M;W) bits, for each beta.
        accuracy_vec
            IB-optimal accuracy values, I(U;W) bits, for each beta.
        entropy_vec
            Entropy of the IB-optimal q(w) distributions, for each beta.
        ib_vec
            Optimal IB function values for each beta.
        beta_vec
            Beta values kept after dropping those that got stuck in local minima.
        mi_um
            Mutual information of the observations, U, and the meanings, M.
        max_ent
            The maximum entropy of the meanings, max(H[p(m)]).
        """

        # Get beta values
        beta_vec = np.linspace(max_beta, min_beta, n_beta)

        # # Parallel computing of compression for desired beta values
        # with mp.Pool(processes=n_threads) as pool:
        #     results = [pool.apply_async(
        #         self.run_beta_specific_iterations,
        #         args=(b, n_restarts, max_iter, rtol, 0.0, True, verbose, save_encoders)) for b in beta_vec]
        #     results = [p.get() for p in results]
        # complexity_vec = [x[0] for x in results]
        # accuracy_vec = [x[1] for x in results]
        # entropy_vec = [x[2] for x in results]
        # ib_vec = [x[3] for x in results]
        # beta_vec = [x[4] for x in results]

        # Initialize
        self.randomly_initialize()

        # Alternatively, run in for loop on single thread
        # TODO: see if this fixes the issues saving encoders
        complexity_vec = []
        accuracy_vec = []
        entropy_vec = []
        ib_vec = []
        beta_vec2 = []
        for b in beta_vec:
            results = self.run_beta_specific_iterations_with_annealing(b, n_restarts, max_iter, rtol, 0.0,
                True, verbose, save_encoders)
            complexity_vec.append(results[0])
            accuracy_vec.append(results[1])
            entropy_vec.append(results[2])
            ib_vec.append(results[3])
            beta_vec2.append(results[4])

        # # TODO: remove this if we don't keep parallelization
        # # Values of beta may not be sorted appropriately due to out-of
        # # order execution if using many processes. So we have to sort
        # # the result lists (ix, iy, hm, beta) in ascending beta order.
        # complexity_vec = [x for _, x in sorted(zip(beta_vec, complexity_vec))]
        # accuracy_vec = [x for _, x in sorted(zip(beta_vec, accuracy_vec))]
        # entropy_vec = [x for _, x in sorted(zip(beta_vec, entropy_vec))]
        # ib_vec = [x for _, x in sorted(zip(beta_vec, ib_vec))]
        # beta_vec = sorted(beta_vec)

        if ensure_monotonic:
            ub, ids = compute_upper_bound(complexity_vec, accuracy_vec)
            complexity_vec = np.squeeze(ub[:, 0])
            accuracy_vec = np.squeeze(ub[:, 1])
            entropy_vec = np.array(entropy_vec)[ids]
            ib_vec = np.array(ib_vec)[ids]
            beta_vec2 = np.array(beta_vec2)[ids]

        # Return saturation point (mixy) and max horizontal axis (hx)
        mi_um = get_mutual_information(self.pu, self.pm, self.pu_m)
        max_ent = get_entropy(self.pm)

        # Save results
        self.complexity = complexity_vec
        self.accuracy = accuracy_vec
        self.beta_vals = beta_vec2
        self.obj_fun_val = ib_vec

        return complexity_vec, accuracy_vec, entropy_vec, ib_vec, beta_vec2, mi_um, max_ent

if __name__ == '__main__':

    import numpy as np
    import pandas as pd
    import pickle
    from matplotlib import pyplot as plt
    from empirical_tests.wcs_data_processing import wcs_data_pull
    from ib_from_scratch.wcs_meaning_distributions import generate_WCS_meanings
    # from ib_from_scratch.updated_ib_color_model import IBModel

    # Example using my IB test

    # Pull the data
    chip_df, term_df, lang_df, vocab_df, cielab_df = wcs_data_pull(
        wcs_data_path='/Users/lindsayskinner/Documents/school/CLMS/Thesis/data/WCS-Data-20110316',
        CIELAB_path='/Users/lindsayskinner/Documents/school/CLMS/Thesis/data/cnum_CIELAB_table.txt'
    )

    # Initialize color space values
    u_vals = cielab_df['chip_id'].values
    u_coords = cielab_df[['L', 'A', 'B']].values
    #words = np.array(['w1', 'w2', 'w3', 'w4', 'w5'])
    words = np.array(range(0, 330))

    # Generate meaning distributions, as defined in IB color paper
    meanings = generate_WCS_meanings(perceptual_variance=64)

    # Load the p(u) distribution used in the original project
    model_path = '/Users/lindsayskinner/Documents/school/CLMS/Thesis/IB_color_naming_model/IB_color_naming.pkl'
    with open(model_path, 'rb') as f:
        model_data = pkl.load(f)
    pu = model_data['pM']

    # compute the IB bound from the data (vanilla IB; Tishby et al 2001)
    ibm = IBModel(observations=u_vals, coordinates=u_coords, meanings=meanings, words=words, pu=pu)
    res = ibm.run_ib(n_beta=400, min_beta=1.01, max_beta=5.0, n_restarts=3, rtol=1e-5, max_iter=500,
                     ensure_monotonic=False, save_encoders='best')

    complexity, accuracy, entropy, ib, beta, mi, me = res

    # Specify version number
    vz = '20241207_8'

    # Save encoders
    encoder_path = f'/res_original/encoders/encoders_{vz}.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(ibm.saved_encoders, f)

    # Save results
    res_path = f'/res_original/ib_results/res_{vz}.csv'
    res_df = pd.DataFrame({'complexity': complexity,
                           'accuracy': accuracy,
                           'entropy': entropy,
                           'ib': ib,
                           'beta': beta})
    res_df.to_csv(res_path)
    print(f'Mutual Information: {mi}')
    print(f'Maximum Entropy: {me}')

    # plot the IB bound
    plt_path = f'/Users/lindsayskinner/Documents/school/CLMS/Thesis/figures/ibplot_{vz}.png'
    plt.plot(complexity, accuracy)
    #plt.show()
    # save figure
    plt.savefig(plt_path)

