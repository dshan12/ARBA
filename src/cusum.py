import numpy as np


class CUSUMModule:
    def __init__(self, lambda_0: float, lambda_1: float, threshold: float = 5.0):
        if lambda_0 <= 0 or lambda_1 <= 0:
            raise ValueError(f"lambda_0 and lambda_1 must be > 0, got {lambda_0}, {lambda_1}")
        if lambda_1 == lambda_0:
            raise ValueError("lambda_1 must differ from lambda_0 for CUSUM to detect shifts")
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.threshold = threshold
        self.k = (self.lambda_1 - self.lambda_0) / np.log(
            self.lambda_1 / self.lambda_0
        )
        self.S_t = 0.0

    def update(self, x_t: float) -> float:
        self.S_t = max(
            0.0,
            self.S_t
            + x_t * np.log(self.lambda_1 / self.lambda_0)
            - (self.lambda_1 - self.lambda_0),
        )
        return self.S_t

    def is_triggered(self) -> bool:
        return self.S_t >= self.threshold

    def reset(self, lambda_0: float | None = None, lambda_1: float | None = None):
        if lambda_0 is not None:
            self.lambda_0 = lambda_0
        if lambda_1 is not None:
            self.lambda_1 = lambda_1
        self.k = (self.lambda_1 - self.lambda_0) / np.log(
            self.lambda_1 / self.lambda_0
        )
        self.S_t = 0.0

    def calibrate_threshold(
        self,
        arl_target: float,
        n_sim: int = 10000,
        seed: int = 42,
    ) -> float:
        rng = np.random.default_rng(seed)
        lo, hi = 1.0, 50.0
        found_h = None

        while hi - lo > 0.5:
            mid = (lo + hi) / 2.0
            false_alarms = 0
            for _ in range(n_sim):
                self.S_t = 0.0
                for _ in range(200):
                    x = rng.poisson(self.lambda_0)
                    self.update(x)
                    if self.S_t >= mid:
                        false_alarms += 1
                        break
            far = false_alarms / n_sim
            if far <= 1.0 / arl_target:
                hi = mid
                found_h = mid
            else:
                lo = mid

        self.threshold = found_h if found_h is not None else hi
        return self.threshold
