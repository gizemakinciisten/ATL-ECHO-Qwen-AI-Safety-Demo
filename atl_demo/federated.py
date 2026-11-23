"""Module C: illustrative pairwise additive masking, NOT secure aggregation."""
import random

def run(seed=19, agents=5, width=8):
    rng=random.Random(seed)
    values=[[rng.uniform(-1,1) for _ in range(width)] for _ in range(agents)]
    masked=[v[:] for v in values]
    for i in range(agents):
        for j in range(i+1,agents):
            mask=[rng.uniform(-3,3) for _ in range(width)]
            for k in range(width):
                masked[i][k]+=mask[k]; masked[j][k]-=mask[k]
    plain=[sum(row[k] for row in values) for k in range(width)]
    aggregate=[sum(row[k] for row in masked) for k in range(width)]
    error=max(abs(a-b) for a,b in zip(plain,aggregate))
    assert error<1e-10
    return {"module":"C: additive masking illustration","status":"synthetic numerical check",
            "agents":agents,"dimensions":width,"maximum_aggregate_error":error,
            "warning":"Deterministic illustrative masks; NOT cryptographically secure, private, or production-ready. Individual values must not be considered protected."}
