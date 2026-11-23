"""Module A: evidence audit only; never reconstructs private attribution scores."""
import math

def binomial_tail(k, n, p):
    return sum(math.comb(n, i)*p**i*(1-p)**(n-i) for i in range(k,n+1))

def run():
    # These are figures stated in the supplied v6.9 source code, not fresh run output.
    prior = {"hits":8,"valid":10}
    # 34/40 is a reported pooled result; the individual v6.9 run CSV was not supplied.
    reported = {"hits":34,"valid":40,"chance":0.2}
    return {"module":"A: archival evidence audit", "status":"reported, not reproduced",
            "v68_reference":prior, "reported_pooled":reported,
            "binomial_tail_under_independent_random_guessing":binomial_tail(34,40,.2),
            "warning":"No original scenario predictions supplied; this does not independently verify 34/40. No null false-positive rate inferred from random-label matches."}
