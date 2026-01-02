from enum import Enum, auto
import numpy as np
from scipy.stats import truncnorm

class Category(Enum):
    A = auto()
    B = auto()
    C = auto()
    D = auto()

class CategoryHelper:
    def __init__(self, category : Category, dist : str):
        self.category = category
        self.dist = dist
        if dist not in ['avg','bimodal','low-weight','high-weight']:
            raise Exception('Distribution not allowed. Select avg, bimodal, low-weight, or high-weight')
    
    def get_preference(self):
        if self.dist == 'avg':
            return self.get_avg_preference()
        if self.dist == 'bimodal':
            return self.get_bimodal_preference()
        if self.dist == 'low-weight':
            return self.get_low_weight_preference()
        if self.dist == 'high-weight':
            return self.get_high_weight_preference()
        
    def get_avg_preference(self):
        return self.get_truncated_normal(mean=3.0, sd=0.5, low=0.0, upp=5.0).rvs(random_state=np.random.default_rng())

    def get_bimodal_preference(self):
        return self.sample_trunc_bimodal_normal(
            size = 1,
            mu1 = 1,
            sigma1 = 0.5,
            mu2 = 4,
            sigma2 = 0.5,
            weight=.5
        )

    def get_low_weight_preference(self):
        return self.sample_trunc_bimodal_normal(
            size = 1,
            mu1 = 1,
            sigma1 = 0.5,
            mu2 = 4,
            sigma2 = 0.5,
            weight=.25
        )
    
    def get_high_weight_preference(self):
        return self.sample_trunc_bimodal_normal(
            size = 1,
            mu1 = 1,
            sigma1 = 0.5,
            mu2 = 3.5,
            sigma2 = 0.5,
            weight=.75
        )
    
    # helper: draw from N(mean, sd^2) truncated to [low, upp]
    def get_truncated_normal(self, mean=0.0, sd=1.0, low=0.0, upp=1.0):
        a, b = (low - mean) / sd, (upp - mean) / sd  # convert to standard-normal coords [web:74][web:176]
        return truncnorm(a, b, loc=mean, scale=sd) 

    def sample_trunc_bimodal_normal(
        self,
        size: int,
        mu1: float, sigma1: float,
        mu2: float, sigma2: float,
        weight: float,
        low: float = 0.0,
        high: float = 5.0,
        max_iter: int = 1000,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """
        Draw 'size' samples from a mixture of two normals truncated to [low, high].

        Mixture:
        X ~ weight * N(mu1, sigma1^2) + (1-weight) * N(mu2, sigma2^2)
        then reject samples outside [low, high].
        """
        samples = []
        needed = size
        it = 0

        while needed > 0 and it < max_iter:
            # propose a batch
            n = max(needed, 1000)
            # choose component: True -> component 1, False -> component 2
            choose_1 = rng.random(n) < weight
            # draw from each normal
            x1 = rng.normal(mu1, sigma1, size=n)
            x2 = rng.normal(mu2, sigma2, size=n)
            x = np.where(choose_1, x1, x2)

            # keep only in [low, high]
            mask = (x >= low) & (x <= high)
            accepted = x[mask]

            samples.append(accepted[:needed])
            needed -= accepted.size
            it += 1

        if needed > 0:
            raise RuntimeError("Failed to draw enough truncated samples; adjust parameters.")

        return np.concatenate(samples)


class Reviewer:
    def __init__(self, categories : list, random = True,  tol : float = None, spread: float = None, categories_dict=None):
        if random:
            self.tol = self.get_truncated_normal(mean=3.0, sd=0.5, low=0.0, upp=5.0).rvs(random_state=np.random.default_rng())
            self.spread = np.random.default_rng().uniform(0.25, 1.0)
            self.init_categories(categories)
        else:
            self.tol = tol
            self.spread = spread
            self.categories = categories_dict

        if self.tol is None:
            raise Exception('Tol not supplied')
        if self.spread is None:
            raise Exception('Spread not supplied')
        if self.categories is None:
            raise Exception('Categories not supplied')

    # helper: draw from N(mean, sd^2) truncated to [low, upp]
    def get_truncated_normal(self, mean=0.0, sd=1.0, low=0.0, upp=1.0):
        a, b = (low - mean) / sd, (upp - mean) / sd  # convert to standard-normal coords [web:74][web:176]
        return truncnorm(a, b, loc=mean, scale=sd) 

    def init_categories(self, categories):
        self.categories = dict()
        for cat in categories:
            self.categories[cat.category] = cat.get_preference()
                               

    def review(self, work):
        category_preference = self.categories[work.category]

        if category_preference > self.tol:
            return self.clear_review(work)
        else:
            return self.bias_review(work)

    def _truncnorm_0_5(self, loc: float) -> float:
        """Draw one sample from N(loc, spread^2) truncated to [0, 5]."""
        a, b = (0.0 - loc) / self.spread, (5.0 - loc) / self.spread 
        return truncnorm.rvs(a, b, loc=loc, scale=self.spread)

    def clear_review(self, work):
        """
        'Clear' review case: mean equals work.merit.
        """
        rating = self._truncnorm_0_5(loc=work.merit)
        work.reviews.append(rating)
        return True

    def bias_review(self, work):
        """
        'Bias' case: mean equals the reviewer's category bias
        self.categories[work.category].
        """
        mu = self.categories[work.category]
        rating = self._truncnorm_0_5(loc=mu)
        work.reviews.append(rating)
        return True
    
class Work:
    def __init__(self, category: Category, merit:float):
        self.category = category
        self.merit = merit
        self.reviews = list()