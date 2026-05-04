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
from pathlib import Path
import numpy as np
from hmmlearn.hmm import GaussianHMM

from analyze import analyze_model

# Per-channel, per-trace normalisation percentiles.
# The Nth-low percentile maps to 0; the Nth-high percentile maps to 1.
# Everything outside [floor, ceiling] is clipped to [0, 1].
# Adjust LOW upward to ignore a noisy dark baseline; adjust HIGH downward
# to be more robust to brief intensity spikes.
NORM_PERCENTILE_LOW  =  5
NORM_PERCENTILE_HIGH = 95


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
def load_trace(path,
               low_percentile:  float = NORM_PERCENTILE_LOW,
               high_percentile: float = NORM_PERCENTILE_HIGH):
    """Load a .dat file. Columns: time, green, red, blue.
    Reorder to (green, blue, red) to match the MATLAB convention used here.

    Each channel is normalised independently to [0, 1]:
      - the low_percentile value maps to 0  (robust dark-baseline anchor)
      - the high_percentile value maps to 1 (robust bright-state anchor)
    Values outside [floor, ceiling] are clipped to [0, 1].
    """
    data = np.loadtxt(path)
    time = data[:, 0] - data[0, 0]
    y = np.column_stack([data[:, 1], data[:, 3], data[:, 2]])  # green, blue, red

    # Per-channel two-sided normalisation.
    floors   = np.percentile(y, low_percentile,  axis=0)  # shape (3,)
    ceilings = np.percentile(y, high_percentile, axis=0)  # shape (3,)
    y = (y - floors) / (ceilings - floors)
    y = np.clip(y, 0.0, 1.0)

    return time, y


# Number of independent restarts.  Restart 0 always uses the hand-crafted
# starting point; restarts 1..N_RESTARTS-1 use randomly sampled starts.
N_RESTARTS = 25

# RNG seed for reproducible random starts.
MULTISTART_SEED = 42


def _build_start(stateindex, ug0, ub0, ur0, sigma_g, sigma_b, sigma_r):
    """Assemble means0 and covars0 from level vectors and per-level sigmas."""
    n_states = stateindex.shape[1]
    means0 = build_means_from_levels(stateindex, ug0, ub0, ur0)
    covars0 = np.zeros((n_states, 3))
    for i in range(n_states):
        covars0[i, 0] = sigma_g[stateindex[0, i] - 1] ** 2
        covars0[i, 1] = sigma_b[stateindex[1, i] - 1] ** 2
        covars0[i, 2] = sigma_r[stateindex[2, i] - 1] ** 2
    return means0, covars0


def _sample_start(rng, stateindex):
    """Sample a random starting point.

    Level vectors: draw 3 values uniformly in (0, 1) per channel and sort
    them descending so high > mid > low.  Sigmas are drawn uniformly in
    [0.05, 0.25] — a reasonable range for normalised data.
    """
    def ordered_levels():
        vals = rng.uniform(0.0, 1.0, size=3)
        vals.sort()
        return vals[::-1].copy()   # high, mid, low

    ug = ordered_levels()
    ub = ordered_levels()
    ur = ordered_levels()
    sigma_g = rng.uniform(0.05, 0.5, size=3)
    sigma_b = rng.uniform(0.05, 0.5, size=3)
    sigma_r = rng.uniform(0.05, 0.5, size=3)
    return _build_start(stateindex, ug, ub, ur, sigma_g, sigma_b, sigma_r)


def fit_once(X, lengths, stateindex, means0, covars0, restart_id=0):
    """Run EM from a single starting point.  Returns (model, train_log_likelihood)."""
    n_states = stateindex.shape[1]
    model = ConstrainedGaussianHMM(
        stateindex=stateindex,
        covariance_type="diag",
        n_iter=10000,
        tol=1e-4,
        init_params="",   # we set all params manually below
        params="stmc",    # update startprob, transmat, means, covars
        verbose=False,
    )
    model.startprob_ = np.full(n_states, 1.0 / n_states)
    model.transmat_  = 0.9 * np.eye(n_states) + 0.1 / n_states
    model.means_     = means0
    model.covars_    = covars0

    model.fit(X, lengths=lengths)
    ll = model.score(X, lengths=lengths)
    print(f"  restart {restart_id:2d}: converged={model.monitor_.converged}  "
          f"train log-lik/frame = {ll / X.shape[0]:.4f}")
    return model, ll


def multistart_fit(X, lengths, stateindex, default_means0, default_covars0,
                   n_restarts=N_RESTARTS, seed=MULTISTART_SEED):
    """Run EM from n_restarts starting points; return the best model."""
    rng = np.random.default_rng(seed)
    best_model, best_ll = None, -np.inf

    for r in range(n_restarts):
        if r == 0:
            means0, covars0 = default_means0, default_covars0
        else:
            means0, covars0 = _sample_start(rng, stateindex)

        try:
            model, ll = fit_once(X, lengths, stateindex, means0, covars0,
                                 restart_id=r)
        except Exception as exc:
            print(f"  restart {r:2d}: failed ({exc})")
            continue

        if ll > best_ll:
            best_ll, best_model = ll, model

    return best_model, best_ll


def main():
    # 6 states, with the same stateindex used in maincode.m
    stateindex = np.array([
        [3, 1, 2, 2, 2, 1],
        [3, 3, 1, 2, 3, 3],
        [3, 3, 3, 2, 1, 3],
    ])

    # Hand-crafted starting point (restart 0) on the normalised [0, 1] scale.
    # Original raw-count values were ~[430, 230, 0] / [120, 56, 5] / [250, 110, 3].
    ug0 = np.array([1.00, 0.53, 0.00])
    ub0 = np.array([1.00, 0.47, 0.04])
    ur0 = np.array([1.00, 0.44, 0.01])
    sigma_g = np.array([0.16, 0.14, 0.09])
    sigma_b = np.array([0.21, 0.25, 0.15])
    sigma_r = np.array([0.16, 0.22, 0.12])
    default_means0, default_covars0 = _build_start(
        stateindex, ug0, ub0, ur0, sigma_g, sigma_b, sigma_r)

    # Load traces
    here = Path(__file__).resolve().parent          # HMM/
    cond_dir = here.parent / "GAFsmFRETdata" / "expData_3colorFRET" / "expCondition_461" / "group1"
    files = sorted(glob.glob(str(cond_dir / "*.dat")))[:60]
    Xs, times, lengths = [], [], []
    for f in files:
        t, y = load_trace(f)
        Xs.append(y); times.append(t); lengths.append(y.shape[0])
    X = np.concatenate(Xs, axis=0)
    print(f"Fitting on {len(files)} traces, {X.shape[0]} total frames, "
          f"{N_RESTARTS} restarts\n")

    model, best_ll = multistart_fit(
        X, lengths, stateindex, default_means0, default_covars0)

    print(f"\nBest train log-lik/frame: {best_ll / X.shape[0]:.4f}")
    print("Learned per-color level vectors:")
    print(f"  ug = {model._ug}")
    print(f"  ub = {model._ub}")
    print(f"  ur = {model._ur}")
    print("\nLearned means_ (n_states x 3):")
    print(np.array2string(model.means_, precision=2, suppress_small=True))
    print("\nLearned covars_ (n_states x 3, diagonal):")
    print(np.array2string(np.sqrt(model.covars_), precision=2, suppress_small=True))

    # Visualise best model
    condition_name = cond_dir.name
    traces = [(Path(f).stem, times[k], Xs[k]) for k, f in enumerate(files)]
    out_dir = here / "analyze_output"
    out_dir.mkdir(exist_ok=True)
    analyze_model(model, traces, out_dir / f"{condition_name}_python.pdf",
                  condition_name=condition_name)


if __name__ == "__main__":
    main()
