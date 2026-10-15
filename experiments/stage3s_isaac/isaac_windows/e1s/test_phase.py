import unittest
import numpy as np
from analyze_phase import area_proxy
from phase_diagnostics import first_hit

class PhaseMathTests(unittest.TestCase):
    def test_analytic_rectangles(self):
        grid=np.ones((7,5),bool)
        self.assertAlmostEqual(area_proxy(grid),(.25-.10)*(20-0))
        self.assertAlmostEqual(area_proxy(grid,10),(.25-.10)*(20-10))
        grid[3:]=False
        self.assertAlmostEqual(area_proxy(grid),(.15-.10)*20)

    def test_hole_is_not_interpolated_safe(self):
        grid=np.ones((7,5),bool); grid[2,2]=False
        self.assertAlmostEqual(area_proxy(grid),3.-4*(.025*5))

    def test_confirmation_is_causal_and_missing_is_null(self):
        times=np.arange(6)*.01
        self.assertEqual(first_hit([0,0,1,1,1,0],times,.01,.5,3),.04)
        self.assertIsNone(first_hit([0]*6,times,.01,.5,3))

if __name__=='__main__': unittest.main()
