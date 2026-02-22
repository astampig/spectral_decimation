import numpy as np

def stepf(n: int, I_1_2: float) -> float:
    return (6.0 / (I_1_2 * n)) ** (1.0 / 3.0)

def GOE_params(d, delta):
    M_GOE = np.pi*np.pi*delta/12
    variance_GOE = M_GOE * (1 - delta * M_GOE) / (d * delta)
    return M_GOE, variance_GOE


def decimation(
    in_gaps,
    f: float = 0.01,
    dmax: int = 50,
    rng=None,
):
    # in this implementation of the decimation algorithm, the fraction f is set to be GOE. If you want to test different fractions, set frac = f in 70
    if rng is None:
        rng = np.random.default_rng()

    in_gaps = in_gaps[in_gaps <= 6.]

    N = len(in_gaps)
    used = np.zeros(N, dtype=bool)
    iter = 0
    s_noise = np.inf

    while True:
        available_mask = ~used
        d = available_mask.sum()

        if d <= dmax:
            break

        gaps = in_gaps[available_mask]


        delta = stepf(d, 0.5)

        bin_gaps = (gaps / delta).astype(np.int32)

        bin_counts = np.bincount(bin_gaps)
        max_bin = len(bin_counts)

        M = min(bin_counts[0] / (delta * d), 1.)

        idx = np.arange(max_bin, dtype=np.float64)
        poisson_counts = np.exp(-idx * delta) * (1 - np.exp(-delta)) * d

        z_q = 1.96
        check_compat = np.abs(poisson_counts - bin_counts) <= z_q * np.sqrt(poisson_counts * (1.- poisson_counts/d))

        s_noise = 0

        bin_noise = int(s_noise / delta)

        if check_compat[0]:
            poisson_counts[bin_noise:][check_compat[bin_noise:]] = bin_counts[bin_noise:][check_compat[bin_noise:]]

        # --- Rejection sampling --- #
        u = rng.random(d)
        bc = bin_counts[bin_gaps]
        pc = poisson_counts[bin_gaps]

        check_RS = u * bc <= M * pc
        successes_local_idx = np.flatnonzero(check_RS)

        M_GOE, var_GOE = GOE_params(d, delta)
        frac = M_GOE # here the fraction f can be inserted
        d_E = int(frac * d)

        if successes_local_idx.size <= d_E:
            break

        chosen_local = rng.choice(successes_local_idx, size=d_E, replace=False)

        global_indices = np.flatnonzero(available_mask)[chosen_local]
        used[global_indices] = True
        iter += 1

    return in_gaps[~used]



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(12345)

    N = 1_000_000
    gaps = rng.exponential(scale=1.0, size=N)
    print(f"dmax = {int(0.05 * N)}")
    out = decimation(gaps, dmax=int(0.05 * N))
    print("Final size:", out.size)

    # Plot histogram of r values
    plt.figure(figsize=(10, 5))
    plt.hist(gaps, bins='fd', alpha=0.6, label='Original gaps', density=True,range=(0,4))
    plt.hist(out, bins='fd', alpha=0.6, label='Decimated gaps', density=True,range=(0,4))
    plt.xlabel(r'$s$')
    plt.ylabel('PDF')
    plt.title('Decimation of Poisson gaps')
    plt.legend()
    plt.savefig(f"decimated_r_ratio_exp.pdf")
    plt.close()
