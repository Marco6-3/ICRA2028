import unittest
import numpy as np
from temporal_probe import causal_features


class CausalFeatureTests(unittest.TestCase):
    def test_future_invariance(self):
        rng=np.random.default_rng(2); s=rng.normal(size=(50,4,4))
        data={'state4':s}; changed={'state4':s.copy()}; changed['state4'][21:]+=1000
        for mode in ('mean','delta_sum','mean_delta'):
            for w in (4,8,16):
                a=causal_features(data,f'{mode}_w{w}'); b=causal_features(changed,f'{mode}_w{w}')
                np.testing.assert_array_equal(a[:21],b[:21])

    def test_telescoping_identity(self):
        rng=np.random.default_rng(3); s=rng.normal(size=(50,4,4)); diff=np.diff(s,axis=0)
        for w in (4,8,16):
            result=causal_features({'state4':s},f'delta_sum_w{w}')
            for i in range(w,50):
                np.testing.assert_allclose(result[i],diff[i-w:i].sum(axis=0).reshape(-1),atol=1e-12)


if __name__=='__main__': unittest.main()
