import unittest
from atl_demo.attribution import binomial_tail,run as audit
from atl_demo.federated import run as masking
class Tests(unittest.TestCase):
    def test_binomial(self): self.assertAlmostEqual(binomial_tail(1,1,.2),.2)
    def test_archival_label(self): self.assertEqual(audit()["status"],"reported, not reproduced")
    def test_masking(self): self.assertLess(masking()["maximum_aggregate_error"],1e-10)
if __name__=="__main__": unittest.main()
