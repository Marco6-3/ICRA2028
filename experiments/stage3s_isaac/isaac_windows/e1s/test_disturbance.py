import unittest
import numpy as np
from disturbance import Trial, pilot_matrix


class LoadingTests(unittest.TestCase):
    def test_eccentric_moment_and_release(self):
        tr = Trial('test', amplitude_n=2, point_m=(0,0,.03))
        f,t,p = tr.wrench(.3, [0,0,0,1,0,0,0], [1,2,3])
        np.testing.assert_allclose(f,[2,0,0])
        np.testing.assert_allclose(t,[0,.06,0])
        np.testing.assert_allclose(p,[1,2,3.03])
        np.testing.assert_array_equal(tr.wrench(.5,[0,0,0,1,0,0,0],[0,0,0])[0],0)

    def test_rotated_body_and_reference(self):
        q=[np.sqrt(.5),0,0,np.sqrt(.5)]
        pose=[0,0,0,*q]
        tr=Trial('test', amplitude_n=2, frame='object')
        np.testing.assert_allclose(tr.wrench(.3,pose,[0,0,0])[0],[0,2,0],atol=1e-12)
        tr.frame='reference'
        r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_allclose(tr.wrench(.3,pose,[0,0,0],r)[0],[0,2,0])
        with self.assertRaises(ValueError):
            tr.wrench(.3,pose,[0,0,0],np.zeros((3,3)))

    def test_reverse_and_ramp(self):
        tr=Trial('test', waveform='reverse')
        self.assertEqual(tr.gain(.3),1)
        self.assertEqual(tr.gain(.4),-1)
        self.assertEqual(tr.gain(.6),0)
        tr.waveform='ramp'
        self.assertAlmostEqual(tr.gain(.375),.5)

    def test_coverage(self):
        trials=pilot_matrix()
        self.assertEqual(len(trials),135)
        self.assertEqual(len({t.trial_id for t in trials}),135)
        self.assertEqual({t.frame for t in trials},{'world'})


if __name__=='__main__':
    unittest.main()
