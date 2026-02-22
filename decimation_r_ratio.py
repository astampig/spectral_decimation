import numpy as np
from scipy.interpolate import CubicSpline
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "p_r_CDF.npz")

data = np.load(file_path)
#print(data["integral_vals"])
r_GOE_CDF = CubicSpline(data["r_vals"], data["integral_vals"], extrapolate=False)

def stepf(n: int, I_1_2: float) -> float:
    return (6.0 / (I_1_2 * n)) ** (1.0 / 3.0)

def GOE_params(d, delta):
    #M_GOE = (1 - np.exp(-delta * delta * 0.25 * np.pi)) / delta
    M_GOE = r_GOE_CDF(delta)/delta
    variance_GOE = M_GOE * (1 - delta * M_GOE) / (d * delta)
    return M_GOE, variance_GOE

def poisson_r_ratio(r):
    r = np.asarray(r)
    return 2./((1.+r)**2)

def poisson_r_ratio_counts(n, delta):
    n = np.asarray(n)
    return delta*2./((1.+ n*delta)*(1.+(n+1)*delta))

def decimation_r_ratio(
    in_r,
    f: float = 0.01,
    dmax: int = 50,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    

    N = len(in_r)
    used = np.zeros(N, dtype=bool)
    iter = 0

    while True:
        available_mask = ~used
        d = available_mask.sum()

        if d <= dmax:
            break

        # Work with a view via mask (no copy yet)
        r = in_r[available_mask]

        #print(d, np.mean(gaps), s_noise)

        delta = stepf(d, 3.1)

        bin_r = (r / delta).astype(np.int32)

        # Histogram
        bin_counts = np.bincount(bin_r)
        max_bin = len(bin_counts)

        M = min(bin_counts[0] / (delta * d)/2, 1.)

        #print(iter, d, M)

        idx = np.arange(max_bin, dtype=np.float64)
        poisson_counts =  d * poisson_r_ratio_counts(idx, delta)

        z_q = 1.96
        #if iter == 0:
            #print(poisson_counts)
        check_compat = np.abs(poisson_counts - bin_counts) <= z_q * np.sqrt(poisson_counts * (1.- poisson_counts/d))

        # Noise threshold
        s_noise = 0#(np.log(d*delta)- 2*np.log(z_q) + np.log(1.))
        #s_noise = 0.

        bin_noise = int(s_noise / delta)

        if check_compat[0]:
            poisson_counts[bin_noise:][check_compat[bin_noise:]] = bin_counts[bin_noise:][check_compat[bin_noise:]]

        # --- Rejection sampling ---
        u = rng.random(d)
        bc = bin_counts[bin_r]
        pc = poisson_counts[bin_r]

        check_RS = u * bc <= M * pc
        successes_local_idx = np.flatnonzero(check_RS)

        M_GOE, var_GOE = GOE_params(d, delta)
        frac = M_GOE
        d_E = int(frac * d)
        d_E = max(1, d_E + int(z_q*rng.uniform(-1,1)*np.sqrt(d_E))) #this provides gaussian distribution of dout rather than delta peaks

        if successes_local_idx.size <= d_E:
            break

        chosen_local = rng.choice(successes_local_idx, size=d_E, replace=False)

        # Map back to original indices without rebuilding arrays
        global_indices = np.flatnonzero(available_mask)[chosen_local]
        used[global_indices] = True
        iter += 1

    return in_r[~used]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(12345)

    N = 1_000_000
    gaps = rng.exponential(scale=1.0, size=N)
    r = np.minimum(gaps[1:], gaps[:-1]) / np.maximum(gaps[1:], gaps[:-1])
    print(f"dmax = {int(0.05 * N)}")
    out = decimation_r_ratio(r, f=0.1, dmax=int(0.05 * N))
    print("Final size:", out.size)

    # Plot histogram of r values
    plt.figure(figsize=(10, 5))
    plt.hist(r, bins=100, alpha=0.6, label='Original r', density=True)
    plt.hist(out, bins=100, alpha=0.6, label='Decimated r', density=True)
    plt.xlabel('r value')
    plt.ylabel('PDF')
    plt.title('Distribution of r and decimated r')
    plt.legend()
    plt.savefig(f"decimated_r_ratio_exp.pdf")
    plt.close()
