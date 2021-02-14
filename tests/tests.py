import unittest
import numpy as np

from Protein_Landscape.landscape_class import Protein_Landscape

# Need to find a better way to do the test below

class TestGenLandscape(unittest.TestCase):
    def gen_landscape(self):
        landscape = Protein_Landscape(csv_path="../Data/Small_NK.csv",gen_graph=True)

landscape = Protein_Landscape(csv_path="../Data/Small_NK.csv",gen_graph=True)


class TestIndexing(unittest.TestCase):
    def test_string_idx(self):
        landscape["AAC"]
    def test_int_idx(self):
        landscape[1]
    def test_tuple_idx(self):
        landscape[(0,1,1)]
    def test_len(self):
        len(landscape)

class TestDistanceGeneration(unittest.TestCase):
    def test_get_distance_normal(self):
        landscape.get_distance(2)
    def test_gen_d_data(self):
        landscape.gen_d_data(seq="ACL")
    def test_get_distance_custom_d_data(self):
        out = landscape.get_distance(dist=0,d_data=landscape.gen_d_data(seq="ACL"))
        assert out[0][0] == 'ACL'

if __name__ == "__main__":
    unittest.main()
