""" My implementation of the IB color model """

import numpy as np
from typing import Tuple, Optional
from math import isinf
from scipy.stats import entropy

# Useful helper functions
def klDiv(p: np.array,
          q: np.array) -> float:
    """Function to calculate the kl divergence between two probability distributions.

    Arguments
    ----------
    p
        A probability distribution represented by a 1D array of pdf values.
    q
        A probability distribution represented by a 1D array of pdf values. The domain objects associated with these
        probabilities are assumed to be in the same order as those in p.

    Returns
    -------
    The value of the KL divergence, D_KL[ p || q ].
    """
    # tot = 0.0
    # for i in range(len(p)):
    #     p1 = p[i]
    #     p2 = q[i]
    #     val = p1*np.log2(p1/p2)
    #     tot += val
    #tot = np.sum(p * np.log2(p/q))
    tot = np.log2(p/q) @ p
    return tot

#Need to think about how I'm going to handle different beta values.... maybe fix it at the model level?

class IBModel():

    def __init__(self, observations, coordinates, meanings, words):

        self.u_vals: np.array = observations
        self.u_coords: np.array = coordinates
        self.meanings: np.array = meanings
        self.words: np.array = words
        self.beta: float = None
        self.beta_adjust: float = None

        # Initialize accuracy, complexity and ib objective function values
        self.accuracy: float = None
        self.complexity: float = None
        self.obj_fun_val: float = None

        # Assumes uniform distribution for linguistic need
        self.p_m: np.array = np.ones(meanings.shape[0]) / meanings.shape[0]

        #Assumed to be a 1D array of length |words|
        #q_beta(w)
        self.p_w = np.zeros(words.shape[0])

        #Assumed to be 2D with 'meaning' on axis 0 and 'word' on axis 1
        #q(w|m)
        self.fitted_encoder = np.zeros((meanings.shape[0], words.shape[0]))
        self.fitted_encoder_log = np.zeros((meanings.shape[0], words.shape[0]))

        #Assumed to be 2D with 'word' on axis 0 and 'meaning' on axis 1
        #q(m^|w)
        self.fitted_decoder = np.zeros((words.shape[0], meanings.shape[0]))

        #Assumed to be 2D with 'word' on axis 0 and 'u_vals' on axis 1
        #m^_w(u)
        self.fitted_mu_w = np.zeros((words.shape[0], observations.shape[0]))

    def randomly_initialize(self, rseed: Optional[int], uniform=False, old_method=False):
        """A function to randomly initialize the encoder and calculate the resulting decoder, p(w) and m_w(u)
        distributions in order to provide a random starting point for the IB annealing process.

        Arguments:
        ----------
        rseed
            Random seed, to be provided in the case that results need to be reproducible.
        uniform
            If true then the encoder is initialized with a uniform distribution, rather than random values.

        Returns:
        --------
            None. Initializes the fitted_encoder, fitted_decoder, p_w, and fitted_mu_w distributions for the IB
            annealing process.
        """

        # Set the random seed, if provided
        if rseed is not None:
            np.random.seed(rseed)

        # Intialize encoder
        if uniform:
            self.fitted_encoder = np.ones((self.meanings.shape[0], self.words.shape[0]))
        else:
            self.fitted_encoder = np.random.rand(self.meanings.shape[0], self.words.shape[0])
        # TODO: remove after testing
        if old_method:
            for i in range(self.fitted_encoder.shape[0]):
                self.fitted_encoder[i,:] = self.fitted_encoder[i,:] / np.sum(self.fitted_encoder[i, :])
        else:
            #self.fitted_encoder = (self.fitted_encoder.T / self.fitted_encoder.sum(axis=1)[:,None]).T
            self.fitted_encoder = self.fitted_encoder / self.fitted_encoder.sum(axis=1)[:, None]

        # Calculate decoder
        # TODO: remove after testing
        if old_method:
            for wi in range(self.words.shape[0]):
                for mi in range(self.meanings.shape[0]):
                    factor1 = self.fitted_encoder[mi, wi]
                    factor2 = self.p_m[mi]
                    divisor = np.sum(self.p_m * self.fitted_encoder[:,wi])
                    self.fitted_decoder[wi, mi] = (factor1 * factor2) / divisor
        else:
            # self.fitted_decoder = (
            #         (self.fitted_encoder.T @ self.p_m) / ((self.fitted_encoder.T @ self.p_m).sum(axis=0)).T
            # ).T
            # self.fitted_decoder = (
            #     (self.fitted_encoder * self.p_m[:, None]) / (self.p_m @ self.fitted_encoder)
            # )
            self.fitted_decoder = (
                ((self.fitted_encoder * self.p_m[:, None]) / (self.fitted_encoder * self.p_m[:, None]).sum(axis=0)).T
            )

        # Calculate p(w)
        # TODO: remove after testing
        if old_method:
            for wi in range(self.words.shape[0]):
                qvec = self.fitted_encoder[:,wi]
                prob = np.sum(qvec*self.p_m)
                self.p_w[wi] = prob
        else:
            # self.p_w = self.fitted_encoder.T * self.p_m
            self.p_w = self.p_m @ self.fitted_encoder


        # Calculate m_w(u)
        # TODO: remove after testing
        if old_method:
            for wi in range(self.words.shape[0]):
                for ui in range(self.u_vals.shape[0]):
                    u_tot = 0.0
                    for mi in range(self.meanings.shape[0]):
                        factor1 = self.meanings[mi,ui]
                        factor2 = self.fitted_decoder[wi,mi]
                        u_tot += factor1 * factor2
                    self.fitted_mu_w[wi,ui] = u_tot
        else:
            self.fitted_mu_w = self.fitted_decoder @ self.meanings

    def one_to_one_initialize(self, old_method=False):
        """A function to randomly initialize the encoder and calculate the resulting decoder, p(w) and m_w(u)
        distributions in order to provide a random starting point for the IB annealing process.

        Arguments:
        ----------
        old_method
            Indicates whether to use the old, slower method of calculating the probability distributions, or to use the
            newer method that takes advantage of matrix multiplication.

        Returns:
        --------
            None. Initializes the fitted_encoder, fitted_decoder, p_w, and fitted_mu_w distributions for the IB
            annealing process.
        """

        # Intialize encoder
        # if uniform:
        #     self.fitted_encoder = np.ones((self.meanings.shape[0], self.words.shape[0]))
        # else:
        #     self.fitted_encoder = np.random.rand(self.meanings.shape[0], self.words.shape[0])
        # # TODO: remove after testing
        # if old_method:
        #     for i in range(self.fitted_encoder.shape[0]):
        #         self.fitted_encoder[i,:] = self.fitted_encoder[i,:] / np.sum(self.fitted_encoder[i, :])
        # else:
        #     #self.fitted_encoder = (self.fitted_encoder.T / self.fitted_encoder.sum(axis=1)[:,None]).T
        #     self.fitted_encoder = self.fitted_encoder / self.fitted_encoder.sum(axis=1)[:, None]
        self.fitted_encoder = np.identity(self.meanings.shape[0])
        self.fitted_encoder = self.fitted_encoder + 0.001
        for i in range(self.fitted_encoder.shape[0]):
            self.fitted_encoder[i, :] = self.fitted_encoder[i, :] / np.sum(self.fitted_encoder[i, :])

        # Calculate decoder
        # TODO: remove after testing
        if old_method:
            for wi in range(self.words.shape[0]):
                for mi in range(self.meanings.shape[0]):
                    factor1 = self.fitted_encoder[mi, wi]
                    factor2 = self.p_m[mi]
                    divisor = np.sum(self.p_m * self.fitted_encoder[:,wi])
                    self.fitted_decoder[wi, mi] = (factor1 * factor2) / divisor
        else:
            self.fitted_decoder = (
                ((self.fitted_encoder * self.p_m[:, None]) / (self.fitted_encoder * self.p_m[:, None]).sum(axis=0)).T
            )

        # Calculate p(w)
        # TODO: remove after testing
        if old_method:
            for wi in range(self.words.shape[0]):
                qvec = self.fitted_encoder[:,wi]
                prob = np.sum(qvec*self.p_m)
                self.p_w[wi] = prob
        else:
            self.p_w = self.p_m @ self.fitted_encoder


        # Calculate m_w(u)
        # TODO: remove after testing
        if old_method:
            for wi in range(self.words.shape[0]):
                for ui in range(self.u_vals.shape[0]):
                    u_tot = 0.0
                    for mi in range(self.meanings.shape[0]):
                        factor1 = self.meanings[mi,ui]
                        factor2 = self.fitted_decoder[wi,mi]
                        u_tot += factor1 * factor2
                    self.fitted_mu_w[wi,ui] = u_tot
        else:
            self.fitted_mu_w = self.fitted_decoder @ self.meanings

    def update_encoder(self, beta: Optional[float]) -> float:
        """Function to be used in the fitting loop to iteratively update the encoder function, for a given beta value.

        Parameters
        ----------
        beta
            The given beta value to be used in the IB optimization

        Returns
        -------
        None
        """

        # # Set beta, if necessary
        # if beta is not None:
        #     self.beta = beta
        #     if self.beta > 10.0:
        #         self.beta_adjust = round(np.log10(self.beta), 1)/10
        #     else:
        #         self.beta_adjust = 0.0
        #
        # # Update encoder
        # for mi in range(self.meanings.shape[0]):
        #     for wi in range(self.words.shape[0]):
        #         #factor1 = self.p_w[wi]
        #         factor1 = np.log(self.p_w[wi])
        #         #kd = klDiv(self.meanings[mi, :], self.fitted_mu_w[wi, :])
        #         kd = entropy(pk=self.meanings[mi, :], qk=self.fitted_mu_w[wi, :], base=2.0)
        #         #factor2 = 2.0**(self.beta * kd)
        #         factor2 = np.log(2.0) * self.beta * kd
        #         #val = factor1 * factor2
        #         val = factor1 + factor2 - self.beta_adjust
        #         # if val > 0:
        #         #     self.fitted_encoder[mi, wi] = val
        #         # else:
        #         #     self.fitted_encoder[mi, wi] = 0.0
        #         self.fitted_encoder_log[mi, wi] = val
        #         #self.fitted_encoder[mi, wi] = val
        #     self.fitted_encoder[mi, :] = np.exp(self.fitted_encoder_log[mi, :])
        #     Z_m = np.sum(self.fitted_encoder[mi, :])
        #     self.fitted_encoder[mi, :] = self.fitted_encoder[mi, :] / Z_m

        # Set beta, if necessary
        if beta is not None:
            self.beta = beta
            if self.beta > 10.0:
                self.beta_adjust = round(np.log10(self.beta), 1)/10
            else:
                self.beta_adjust = 1.0

        # Update encoder
        for mi in range(self.meanings.shape[0]):
            for wi in range(self.words.shape[0]):
                factor1 = self.p_w[wi]
                #factor1 = np.log(self.p_w[wi])
                #kd = klDiv(self.meanings[mi, :], self.fitted_mu_w[wi, :])
                kd = entropy(pk=self.meanings[mi, :], qk=self.fitted_mu_w[wi, :], base=2.0)
                factor2 = 2.0**(self.beta * kd) / self.beta_adjust
                #factor2 = np.log(2.0) * self.beta * kd
                val = factor1 * factor2
                #val = factor1 + factor2 - self.beta_adjust
                # if val > 0:
                #     self.fitted_encoder[mi, wi] = val
                # else:
                #     self.fitted_encoder[mi, wi] = 0.0
                #self.fitted_encoder_log[mi, wi] = val
                self.fitted_encoder[mi, wi] = val
            #self.fitted_encoder[mi, :] = np.exp(self.fitted_encoder_log[mi, :])
            Z_m = np.sum(self.fitted_encoder[mi, :])
            self.fitted_encoder[mi, :] = self.fitted_encoder[mi, :] / Z_m

    def update_decoder(self, old_method = False) -> float:
        """Function to be used in the fitting loop to update the decoder function any time the encoder function is updated.

        Assumes the decoder is an optimal Bayesian decision maker, as outlined in the IB color paper.

        Returns
        -------
        None
        """

        # TODO: remove after testing
        if old_method:
            for wi in range(self.words.shape[0]):
                for mi in range(self.meanings.shape[0]):
                    factor1 = self.fitted_encoder[mi, wi]
                    factor2 = self.p_m[mi]
                    divisor = np.sum(self.p_m * self.fitted_encoder[:, wi])
                    self.fitted_decoder[wi, mi] = (factor1 * factor2) / divisor
        else:
            # self.fitted_decoder = (
            #         (self.fitted_encoder.T @ self.p_m) / ((self.fitted_encoder.T @ self.p_m).sum(axis=0))
            # ).T
            # self.fitted_decoder = (
            #     (self.fitted_encoder * self.p_m[:, None]) / (self.p_m @ self.fitted_encoder)
            # )
            self.fitted_decoder = (
                ((self.fitted_encoder * self.p_m[:, None]) / (self.fitted_encoder * self.p_m[:, None]).sum(axis=0)).T
            )


    def update_p_w(self, old_method=False) -> float:
        """Function to be used in the fitting loop to iteratively update the p(w) function, for a given beta value.

        Returns
        -------
        None
        """

        # TODO: remove after testing
        if old_method:
            for wi in range(self.words.shape[0]):
                qvec = self.fitted_encoder[:,wi]
                prob = np.sum(qvec*self.p_m)
                self.p_w[wi] = prob
        else:
            # self.p_w = (self.fitted_encoder.T @ self.p_m)
            self.p_w = self.p_m @ self.fitted_encoder

    def update_mu_w(self, old_method=False) -> None:
        """Function to be used in the fitting loop to iteratively update the m_w(u) function, for a given beta value.

        Returns
        -------
        None
        """

        # TODO: check this, then remove after testing
        if old_method:
            for wi in range(self.words.shape[0]):
                for ui in range(self.u_vals.shape[0]):
                    u_tot = 0.0
                    for mi in range(self.meanings.shape[0]):
                        factor1 = self.meanings[mi,ui]
                        factor2 = self.fitted_decoder[wi,mi]
                        u_tot += factor1 * factor2
                    self.fitted_mu_w[wi,ui] = u_tot
        else:
            self.fitted_mu_w = self.fitted_decoder @ self.meanings

    def run_single_annealing_iteration(self, beta: Optional[float], return_obj_fun_value: bool = False,
                                       old_method: bool = False, verbose: bool = False
                                       ) -> Tuple[float, float, float]:
        """Performs a single iteration of the annealing process, updating the encoder, decoder, p(w) and m_w(u)
        distributions.

        Arguments
        ---------
        beta
            The given beta value to be used in the IB optimization
        return_obj_fun_value
            If true then the objective function is evaluated and the value is returned.

        Returns
        -------
        obj_fun_value
            The value of the objective function I(M;W) - beta * I(W;U) evaluated using the current color naming system.
            Only returned if retun_obj_fun_value is set to True.
        """

        # Set beta if necessary
        if beta is not None:
            self.beta = beta

        # Perform single step of annealing process
        #self.update_p_w(old_method)
        #self.update_mu_w(old_method)
        self.update_p_w(old_method=True)
        self.update_mu_w(old_method=True)
        self.update_encoder(beta)
        #self.update_decoder(True)
        self.update_decoder(old_method=True)

        if verbose:
            print(f'p_w = {self.p_w[:3]}')
            print(f'mu_w = {self.fitted_mu_w[:3, :3]}')
            print(f'enc = {self.fitted_encoder[:3, :3]}')
            print(f'dec = {self.fitted_decoder[:3, :3]}')

        # If applicable, update accuracy, complexity and objective function values
        if return_obj_fun_value:
            self.accuracy = self.get_accuracy()
            self.complexity = self.get_complexity()
            obj_fun_value = self.complexity - (self.beta * self.accuracy)
            self.obj_fun_val = obj_fun_value

            return obj_fun_value

    #TODO: Add patience parameter
    def perform_beta_specific_annealing(
            self,
            beta: float,
            thresh: Optional[float],
            max_iter: Optional[int],
            verbose: bool = True
    ) -> dict:
        """Performs IB annealing until convergence (of what? see questions below) for a particular beta value.

        Arguments
        ---------
        beta
            The given beta value to be used in the IB optimization
        thresh
            The maximum allowable difference between annealing iterations for which we would conclude that the process
            has converged. If left blank, the annealing process will proceed for max_iter iterations.
        max_iter
            The maximum number of iterations to run the annealing process. If max_iter is reached before convergence
            then the process halts. If left blank, the annealing process will proceed until the difference between
            iterations is less than thresh.
        verbose
            If true then intermittent difference values are printed for each step during annealing. If false then
            nothing is printed to standard out during the annealing process.

        Returns
        -------
        results
            The fitted distributions for the encoder, decoder, p(w) and m_w(u) after the annealing process has
            converged.
        """

        # Ensure necessary stopping criteria have been provided
        assert (thresh is not None or max_iter is not None), 'At least one of thresh or max_iter must be provided.'

        n_iter = 0
        if thresh is not None and max_iter is not None:
            obj_diff = np.Inf
            old_val = 0
            while obj_diff > thresh and n_iter <= max_iter:
                obj_val = self.run_single_annealing_iteration(beta, return_obj_fun_value=True)
                if isinf(obj_diff):
                    obj_diff = obj_val
                else:
                    obj_diff = np.abs(old_val - obj_val)
                    old_val = obj_val
                if verbose:
                    print(f'{n_iter}. Objective: {obj_val}, Accuracy: {self.accuracy}, Complexity: {self.complexity}')
                n_iter += 1
        elif thresh is not None:
            obj_diff = np.Inf
            old_val = 0
            while obj_diff > thresh:
                obj_val = self.run_single_annealing_iteration(beta, return_obj_fun_value=True)
                if isinf(obj_diff):
                    obj_diff = obj_val
                else:
                    obj_diff = np.abs(old_val - obj_val)
                    old_val = obj_val
                if verbose:
                    print(f'{n_iter}. Objective: {obj_val}, Accuracy: {self.accuracy}, Complexity: {self.complexity}')
                n_iter += 1
        else:
            n_iter = 0
            while n_iter <= max_iter:
                obj_val = self.run_single_annealing_iteration(beta, return_obj_fun_value=True)
                if verbose:
                    print(f'{n_iter}. Objective: {obj_val}, Accuracy: {self.accuracy}, Complexity: {self.complexity}')
                n_iter += 1

        results = {
            'encoder': self.fitted_encoder,
            'decoder': self.fitted_decoder,
            'p_w': self.p_w,
            'm_w': self.fitted_mu_w
        }
        return results

    def perform_all_values_annealing(
            self,
            n_beta: int,
            min_beta: float,
            max_beta: float,
            thresh: Optional[float],
            max_iter: Optional[int],
    ) -> dict:
        """Performs IB annealing until convergence across all beta values.

        Starts with beta = 1.0 and ascends the remaining beta values in order, using the optimal results from the
        annealing process performed on the previous beta value as the starting distributions for the annealing process
        performed on the current beta value. This follows the methodology outlined in the Supplementary Information for
        Efficient compression in color naming and its evolution.

        Arguments:
        -----------
        n_beta
            The number of beta values over which to perform the annealing process. The beta values will be evenly spaced
            between the specified min and max beta value.
        min_beta
            The smalled beta value for which to perform the annealing process.
        max_beta
            The largest beta value for which to perform the annealing process.
        thresh
            The maximum allowable difference between annealing iterations for which we would conclude that the process
            has converged. If left blank, the annealing process will proceed for max_iter iterations.
        max_iter
            The maximum number of iterations to run the annealing process. If max_iter is reached before convergence
            then the process halts. If left blank, the annealing process will proceed until the difference between
            iterations is less than thresh.

        Returns:
        --------
        all_results
            Includes the optimal encoder, decoder, p(w) and m_w(u) for each beta value.
        """

        # Calculate the beta values
        assert n_beta > 0, 'n_beta must be a positive integer'
        if n_beta == 1:
            beta_vals = [(min_beta + max_beta)/2]
        else:
            beta_vals = []
            mesh = (max_beta - min_beta)/(n_beta - 1)
            cur_beta = min_beta
            while cur_beta <= max_beta:
                beta_vals.append(cur_beta)
                cur_beta += mesh

        # Initialize the distributions before annealing process can begin
        self.randomly_initialize(rseed=42)

        # Perform annealing for each beta value
        all_results = {}
        for beta in beta_vals:
            beta_results = self.perform_beta_specific_annealing(beta, thresh, max_iter, verbose=False)
            all_results[str(beta)] = beta_results

        return all_results

    def _q_word(self, wi: int):
        """
        Helper function to calculate the probability of a particular word occurring in the current encoding. To be used
        in the get_accuracy calculation.

        Arguments:
        ----------
        wi
            Index of the word for which the probability is being returned.

        Returns:
        --------
        The probability of the word corresponding to index wi occurring in the current naming system.
        """
        prob = np.sum(self.fitted_encoder[:, wi] * self.p_m)
        #prob = self.p_m @ self.fitted_encoder[:, wi]
        return prob

    def _m0(self, ui: int):
        """
        Helper function to calculate the probability of a particular observation using the prior over the observation
        space U before knowing the word w.

        Arguments:
        ----------
        ui
            Index of the observation for which the probability is being returned.

        Returns:
        --------
        The probability of the observation corresponding to index ui using the prior without conditioning on a
        particular word.
        """

        prob = np.sum(self.p_m * self.meanings[:, ui])
        return prob

    def get_complexity(self) -> float:
        """
        Calculates the complexity of the current color-naming system.

        Returns:
        --------
        comp_val
            A positive value indicating the level of complexity. Determined by the information rate, I(M|W).
        """

        # Calculate I(M;W)
        # TODO: remove after testing
        comp_val = 0
        for mi in range(self.meanings.shape[0]):
            pm = self.p_m[mi]
            kl_val = 0
            for wi in range(self.words.shape[0]):
                q_wm = self.fitted_encoder[mi, wi]
                q_w = self._q_word(wi)
                if q_w > 0:
                    if q_wm/q_w > 0:
                        kl_val += q_wm * np.log2(q_wm/q_w)
                else:
                    print(f'Encountered negative q_w value: {q_w}')
            comp_val += pm * kl_val
        #kl_vals = (self.fitted_encoder * np.log2(self.fitted_encoder / self._q_word[:, None])).sum(axis=1) #axis?
        #comp_val = self.p_m @ kl_vals

        return comp_val

    def get_accuracy(self) -> float:
        """
        Calculates the accuracy of the current color-naming system.

        Returns:
        --------
        acc_val
            A positive value indicating the accuracy. Determined by the information rate, I(U|W).
        """

        # Calculate I(U;W)
        acc_val = 0
        # #kl_vals = (self.fitted_mu_w * np.log2(self.fitted_mu_w / self._m0[:, None])).sum(axis=1)  # axis?
        # for wi in range(self.words.shape[0]):
        #     qw = self._q_word(wi)
        #     # TODO: remove after testing
        #     kl_val = 0
        #     for ui in range(self.u_vals.shape[0]):
        #         m_hat = self.fitted_mu_w[wi, ui]
        #         m0 = self._m0(ui)
        #         if m0 > 0:
        #             if m_hat/m0 > 0:
        #                 kl_val += m_hat * np.log2(m_hat/m0)
        #         else:
        #             print(f'Negative m0 value: {m0}')
        #     acc_val += qw * kl_val
            #acc_val += qw * kl_vals[wi]

        m0_vec = []
        for ui in range(self.u_vals.shape[0]):
            m0 = self._m0(ui)
            m0_vec.append(m0)

        for wi in range(self.words.shape[0]):
            qw = self._q_word(wi)
            kl_val = entropy(self.fitted_mu_w[wi, :], m0_vec)
            acc_val += qw * kl_val

        return acc_val


if __name__ == '__main__':

    # Imports
    import numpy as np
    from empirical_tests.wcs_data_processing import wcs_data_pull
    from ib_from_scratch.wcs_meaning_distributions import generate_WCS_meanings

    # Pull the data
    chip_df, term_df, lang_df, vocab_df, cielab_df = wcs_data_pull(
        wcs_data_path='/Users/lindsayskinner/Documents/school/CLMS/Thesis/data/WCS-Data-20110316',
        CIELAB_path='/Users/lindsayskinner/Documents/school/CLMS/Thesis/data/cnum_CIELAB_table.txt'
    )

    # Initialize color space values
    beta = 1.05
    u_vals = cielab_df['chip_id'].values
    u_coords = cielab_df[['L', 'A', 'B']].values
    #words = np.array(['w1', 'w2', 'w3', 'w4', 'w5'])
    words = np.array(range(0,330))

    # Generate meaning distributions, as defined in IB color paper
    meanings = generate_WCS_meanings(perceptual_variance=64)

    # Initialize IB Model
    ibm = IBModel(observations=u_vals, coordinates=u_coords, meanings=meanings, words=words)
    ibm.randomly_initialize(rseed=42, uniform=False)

    # Test single round of annealing
    #single_res = ibm.run_single_annealing_iteration(beta=beta, return_obj_fun_value=True, old_method=False, verbose=True)

    # Test full annealing round for a single beta value
    beta_res = ibm.perform_beta_specific_annealing(beta=beta, thresh=None, max_iter=100, verbose=True)

    # Examine color term distribution
    w_to_m_map = {}
    for mi in range(ibm.fitted_encoder.shape[0]):
        wi = np.argmax(ibm.fitted_encoder[mi,:])
        if wi in w_to_m_map.keys():
            w_to_m_map[wi].append(mi)
        else:
            w_to_m_map[wi] = [mi]

    for k in w_to_m_map.keys():
        print(f'{k}: {len(w_to_m_map[k])}')
