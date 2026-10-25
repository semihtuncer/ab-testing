import numpy as np
from scipy.stats import beta

def compare_ads(impressions_a, clicks_a, impressions_b, clicks_b, alpha=1, beta_prior=1, samples=100_000, seed=42):
    rng = np.random.default_rng(seed)

    a1 = alpha + clicks_a
    b1 = beta_prior + impressions_a - clicks_a

    a2 = alpha + clicks_b
    b2 = beta_prior + impressions_b - clicks_b
    
    theta_a = rng.beta(a1, b1, samples)
    theta_b = rng.beta(a2, b2, samples)
    difference = theta_b - theta_a

    return {
        "prob_b_wins": float(np.mean(difference > 0)),
        "estimated_ctr_a": a1 / (a1 + b1),
        "estimated_ctr_b": a2 / (a2 + b2),
        "credible_interval_a": beta.ppf(
            [0.025, 0.975], a1, b1
        ),
        "credible_interval_b": beta.ppf(
            [0.025, 0.975], a2, b2
        ),
        "difference_interval": np.quantile(
            difference, [0.025, 0.975]
        ),
        "expected_loss_choose_a": float(
            np.maximum(difference, 0).mean()
        ),
        "expected_loss_choose_b": float(
            np.maximum(-difference, 0).mean()
        ),
        "samples_a": theta_a,
        "samples_b": theta_b
    }

if __name__ == "__main__":
    result = compare_ads(
        impressions_a=10000,
        clicks_a=340,
        impressions_b=10000,
        clicks_b=420
    )

    print("P(B > A):", result["prob_b_wins"])
    print("Posterior CTR A:", result["estimated_ctr_a"])
    print("Posterior CTR B:", result["estimated_ctr_b"])
    print("95% uplift interval (absolute):",
    result["difference_interval"])