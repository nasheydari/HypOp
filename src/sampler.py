from src.adanSample.utils.sampling_utils import Gaussian_sampler
import numpy as np
from src.utils import samples_to_Xs
class Sampler:
    def __init__(self, params, n, f) -> None:
        # sampler initialization
        self.boundaries = np.asarray([[0, 1] for _ in range(n*f)])
        self.sampler = Gaussian_sampler(
            self.boundaries, params['minimum_good_samples'], params['random_portion'], 
            params['local_portion'], params['cross_portion'], 'top_and_random'
        )
        self.num_samples = params['num_samples']
        self.adam = params['Adam']
        self.n = n
        self.f = f
    def get_Xs(self, time):
        if self.adam:
            if time == 0:
                self.samples = self.sampler.sample_uniform(self.num_samples)
                Xs = [samples_to_Xs(self.samples[i], self.n, self.f) for i in range(self.num_samples)]
                self.origins = ['U']*len(self.samples)
            else:
                self.samples, self.origins = self.sampler.sample(self.num_samples, verbose=False)
                Xs = [samples_to_Xs(self.samples[i], self.n, self.f) for i in range(self.num_samples)]
        else:
            self.samples = self.sampler.sample_uniform(self.num_samples)
            Xs = [samples_to_Xs(self.samples[i], self.n, self.f) for i in range(self.num_samples)]

        return Xs
    def update(self, scores):
        if self.adam:
            print(self.samples, scores, self.origins)
            self.sampler.update(samples=self.samples, scores=[-x for x in scores], origins=self.origins, alpha_max=1.0)
            self.sampler.configure_alpha(1.0, verbose=False)
        
