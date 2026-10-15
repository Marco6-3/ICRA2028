import unittest
from unittest.mock import patch
import numpy as np
import aligned_policy

class Stub:
    def __init__(self,*args): pass
    def step(self,mcf,velocity,angle):
        assert not np.any(velocity) and angle==0
        return float(mcf[0,0]),0.,0.,0.

class PolicyTest(unittest.TestCase):
    def build(self,trigger='none',close=.001):
        with patch.object(aligned_policy,'Controller',Stub):
            p=aligned_policy.AlignedController('mcf',1/120,None,close,trigger)
        mcf=np.zeros((4,8)); mcf[0,0]=.001
        for t in np.arange(54)/120:
            delta,_,_,_=p.step(mcf,t)
            self.assertEqual(delta,0.)
        return p,mcf

    def test_baseline_subtraction_and_shared_absolute_limits(self):
        p,x=self.build()
        self.assertAlmostEqual(p.step(x,.45)[0],0.)
        x[0,0]=.002
        self.assertAlmostEqual(p.step(x,.46)[0],.001)
        x[0,0]=.1
        self.assertAlmostEqual(p.step(x,.47)[0]+p.base_close,.003)
        x[0,0]=0.
        self.assertAlmostEqual(p.step(x,.48)[0]+p.base_close,0.)

    def test_pulse_is_one_shot_and_causal(self):
        p,x=self.build('pulse'); p.step(x,.45)
        self.assertFalse(p.fired)
        x[2,1]=2e-6
        delta,_,fired,_=p.step(x,.50)
        self.assertTrue(fired); self.assertAlmostEqual(delta,.002)
        self.assertAlmostEqual(p.step(x,.525)[0],0.)
        self.assertAlmostEqual(p.step(x,.70)[0],0.)
        self.assertEqual(p.triggered_at,.50)

    def test_latch_requires_contiguous_confirmation_and_decays(self):
        p,x=self.build('latch'); p.step(x,.45)
        x[2,1]=2e-6
        p.step(x,.50); p.step(x,.50+1/120)
        self.assertFalse(p.fired)
        x[2,1]=0.; p.step(x,.50+2/120)
        x[2,1]=2e-6
        for t in (.55,.55+1/120,.55+2/120): p.step(x,t)
        self.assertTrue(p.fired)
        self.assertAlmostEqual(p.step(x,p.triggered_at+.1)[0],.002)
        self.assertAlmostEqual(p.step(x,p.triggered_at+.40)[0],.002*np.exp(-1))

if __name__=='__main__': unittest.main()
