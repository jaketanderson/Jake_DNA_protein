"""
ConstrainedGaussianHMM: a subclass of hmmlearn's GaussianHMM that imposes the
shared-FRET-level constraint described in the GAF-DBD supplementary information
(Eqs. 5-9). Each hidden state's emission mean is constructed from 3 per-color
intensity-level vectors u^g, u^b, u^r (each of length 3 for high/mid/low),
selected via state-specific selection matrices M_i^g, M_i^b, M_i^r. The M-step
solves a single coupled 9x9 linear system for u^g, u^b, u^r rather than
updating each state's mean independently.

*** FOR TESTING PURPOSES
*** This script was NOT used to create HMMs included in this same commit.

Written by Claude
"""

import os
import glob
import numpy as np
from hmmlearn.hmm import GaussianHMM


def build_select_matrices(stateindex):
    """
    stateindex : (3, n_states) int array, rows = (green, blue, red), entries
                 are 1-based level indices (1=high, 2=mid, 3=low) per the
                 MATLAB convention. Levels actually used per color are
                 1..max(stateindex[c, :]) — typically 3.

    Returns Mg, Mb, Mr each of shape (n_features, n_levels, n_states),
    where n_features = 3 (green/blue/red).
    """
    d, n_states = stateindex.shape
    n_levels_g = stateindex[0].max()
    n_levels_b = stateindex[1].max()
    n_levels_r = stateindex[2].max()

    Mg = np.zeros((d, n_levels_g, n_states))
    Mb = np.zeros((d, n_levels_b, n_states))
    Mr = np.zeros((d, n_levels_r, n_states))
    for i in range(n_states):
        Mg[0, stateindex[0, i] - 1, i] = 1.0
        Mb[1, stateindex[1, i] - 1, i] = 1.0
        Mr[2, stateindex[2, i] - 1, i] = 1.0
    return Mg, Mb, Mr


def build_means_from_levels(stateindex, ug, ub, ur):
    """Reconstruct the (n_states, n_features) means matrix from level vectors."""
    n_states = stateindex.shape[1]
    means = np.zeros((n_states, 3))
    for i in range(n_states):
        means[i, 0] = ug[stateindex[0, i] - 1]
        means[i, 1] = ub[stateindex[1, i] - 1]
        means[i, 2] = ur[stateindex[2, i] - 1]
    return means


class ConstrainedGaussianHMM(GaussianHMM):
    """
    GaussianHMM with the shared-FRET-level constraint.

    Parameters
    ----------
    stateindex : (3, n_components) int array
        Per-color level index (1-based) for each state.
    All other params: forwarded to GaussianHMM.
    """

    def __init__(self, stateindex, **kwargs):
        kwargs.setdefault("n_components", stateindex.shape[1])
        super().__init__(**kwargs)
        self.stateindex = np.asarray(stateindex, dtype=int)
        self._Mg, self._Mb, self._Mr = build_select_matrices(self.stateindex)

    def _solve_constrained_means(self, stats):
        """
        Solve Eq. (9) from the supplement:

            [ A11 A12 A13 ] [ u^g ]   [ C1 ]
            [ A21 A22 A23 ] [ u^b ] = [ C2 ]
            [ A31 A32 A33 ] [ u^r ]   [ C3 ]

        where
            A_{αβ} = sum_i (M_i^α)^T  Σ_i  M_i^β  (sum_n γ_{ni})
            C_α   = sum_i (M_i^α)^T  Σ_i  (sum_n γ_{ni} x_n)

        Note: the MATLAB reference uses Σ_i directly in this solve, but the
        proper EM derivation uses the precision matrix Σ_i^{-1}. We use the
        precision form, which is what guarantees monotonic likelihood
        increase. (With near-equal diagonal Σ_i across states, the two forms
        give similar answers, which is why the MATLAB version converges in
        practice on its data.)

        stats['obs']  : (n_states, n_features) = sum_t γ_i(t) * x_t  per state
        stats['post'] : (n_states,)            = sum_t γ_i(t)        per state
        """
        Mg, Mb, Mr = self._Mg, self._Mb, self._Mr
        n_states = self.n_components
        gamaSumT = stats["post"]                  # (n_states,)
        gamaSumY = stats["obs"].T                 # (n_features, n_states)

        # self.covars_ is the property -- it always returns full (n_states, d, d)
        # regardless of covariance_type. Invert to get precision matrices.
        sigma = np.linalg.inv(self.covars_)

        kg = Mg.shape[1]
        kb = Mb.shape[1]
        kr = Mr.shape[1]

        A11 = np.zeros((kg, kg)); A12 = np.zeros((kg, kb)); A13 = np.zeros((kg, kr))
        A21 = np.zeros((kb, kg)); A22 = np.zeros((kb, kb)); A23 = np.zeros((kb, kr))
        A31 = np.zeros((kr, kg)); A32 = np.zeros((kr, kb)); A33 = np.zeros((kr, kr))
        C1 = np.zeros(kg); C2 = np.zeros(kb); C3 = np.zeros(kr)

        for i in range(n_states):
            S = sigma[i]                           # (d, d)
            w = gamaSumT[i]                        # scalar
            Mgi, Mbi, Mri = Mg[:, :, i], Mb[:, :, i], Mr[:, :, i]
            A11 += Mgi.T @ S @ Mgi * w
            A12 += Mgi.T @ S @ Mbi * w
            A13 += Mgi.T @ S @ Mri * w
            A21 += Mbi.T @ S @ Mgi * w
            A22 += Mbi.T @ S @ Mbi * w
            A23 += Mbi.T @ S @ Mri * w
            A31 += Mri.T @ S @ Mgi * w
            A32 += Mri.T @ S @ Mbi * w
            A33 += Mri.T @ S @ Mri * w
            yi = gamaSumY[:, i]                    # (d,)
            C1 += Mgi.T @ S @ yi
            C2 += Mbi.T @ S @ yi
            C3 += Mri.T @ S @ yi

        coeff = np.block([[A11, A12, A13],
                          [A21, A22, A23],
                          [A31, A32, A33]])
        rhs = np.concatenate([C1, C2, C3])
        # pinv handles rank-deficiency (state with ~0 occupancy)
        sol = np.linalg.pinv(coeff) @ rhs

        ug = sol[:kg]
        ub = sol[kg:kg + kb]
        ur = sol[kg + kb:kg + kb + kr]
        return build_means_from_levels(self.stateindex, ug, ub, ur), (ug, ub, ur)

    def _do_mstep(self, stats):
        # Update transmat_ / startprob_ only (skip GaussianHMM's mean+cov
        # update by jumping past it in the MRO -- goes to BaseHMM._do_mstep).
        super(GaussianHMM, self)._do_mstep(stats)

        # Constrained means using the *current* covariances (those from the
        # previous iteration's M-step), then update covariances against the
        # new constrained means. This ordering keeps mean/covar consistent
        # within a single iteration.
        if "m" in self.params:
            self.means_, (self._ug, self._ub, self._ur) = \
                self._solve_constrained_means(stats)

        if "c" in self.params:
            self._update_covars(stats)

    def _update_covars(self, stats):
        """Standard Gaussian-HMM covariance update against self.means_."""
        post = stats["post"][:, None]              # (n_states, 1)
        floor = 1e-5
        denom = np.maximum(post, floor)
        if self.covariance_type == "diag":
            cn = (stats["obs**2"]
                  - 2 * self.means_ * stats["obs"]
                  + self.means_ ** 2 * post)
            self.covars_ = np.maximum(cn / denom, floor)
        elif self.covariance_type == "full":
            new_covars = np.empty_like(self.covars_)
            for c in range(self.n_components):
                obs = stats["obs"][c]
                obs2 = stats["obs*obs.T"][c]
                mu = self.means_[c]
                cv = (obs2
                      - np.outer(obs, mu) - np.outer(mu, obs)
                      + np.outer(mu, mu) * post[c]) / denom[c]
                new_covars[c] = cv + floor * np.eye(cv.shape[0])
            self.covars_ = new_covars
        else:
            raise NotImplementedError(
                f"covariance_type={self.covariance_type!r} not supported "
                "by ConstrainedGaussianHMM in this example.")


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------
def load_trace(path):
    """Load a .dat file. Columns: time, green, red, blue.
    Reorder to (green, blue, red) to match the MATLAB convention used here.
    No per-trace normalization (data is already background-subtracted)."""
    data = np.loadtxt(path)
    time = data[:, 0] - data[0, 0]
    y = np.column_stack([data[:, 1], data[:, 3], data[:, 2]])  # green, blue, red
    return time, y


def main():
    # 6 states, with the same stateindex used in maincode.m
    stateindex = np.array([
        [3, 1, 2, 2, 2, 1],
        [3, 3, 1, 2, 3, 3],
        [3, 3, 3, 2, 1, 3],
    ])

    # Initial level guesses in raw count units (matches the recent MATLAB tweaks)
    ug0 = np.array([430.0, 230.0, 0.0])
    ub0 = np.array([120.0, 56.0, 5.0])
    ur0 = np.array([250.0, 110.0, 3.0])

    sigma_g = np.array([70.0, 60.0, 40.0])
    sigma_b = np.array([25.0, 30.0, 18.0])
    sigma_r = np.array([40.0, 55.0, 30.0])

    means0 = build_means_from_levels(stateindex, ug0, ub0, ur0)

    # Per-state diagonal covariances built from the per-level sigmas
    n_states = stateindex.shape[1]
    covars0 = np.zeros((n_states, 3))
    for i in range(n_states):
        covars0[i, 0] = sigma_g[stateindex[0, i] - 1] ** 2
        covars0[i, 1] = sigma_b[stateindex[1, i] - 1] ** 2
        covars0[i, 2] = sigma_r[stateindex[2, i] - 1] ** 2

    # Pick a few traces from one condition to fit jointly
    cond_dir = "GAFsmFRETdata/expData_3colorFRET/expCondition_461/group1"
    files = sorted(glob.glob(os.path.join(cond_dir, "*.dat")))[:60]
    Xs = []
    lengths = []
    for f in files:
        _, y = load_trace(f)
        Xs.append(y)
        lengths.append(y.shape[0])
    X = np.concatenate(Xs, axis=0)
    print(f"Fitting on {len(files)} traces, total {X.shape[0]} timepoints")

    model = ConstrainedGaussianHMM(
        stateindex=stateindex,
        covariance_type="diag",
        n_iter=1000,
        tol=1e-3,
        init_params="",      # don't let hmmlearn re-initialize
        params="stmc",       # update startprob, transmat, means, covars
        verbose=True,
    )
    model.startprob_ = np.full(n_states, 1.0 / n_states)
    model.transmat_ = 0.9 * np.eye(n_states) + 0.1 / n_states
    model.means_ = means0
    model.covars_ = covars0

    model.fit(X, lengths=lengths)

    print("\nLearned per-color level vectors:")
    print(f"  ug = {model._ug}")
    print(f"  ub = {model._ub}")
    print(f"  ur = {model._ur}")
    print("\nLearned means_ (n_states x 3):")
    print(np.array2string(model.means_, precision=2, suppress_small=True))
    print("\nLearned covars_ (n_states x 3, diagonal):")
    print(np.array2string(np.sqrt(model.covars_), precision=2, suppress_small=True))


if __name__ == "__main__":
    main()
