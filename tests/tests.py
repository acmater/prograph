import unittest
import numpy as np
import torch

from Protein_Landscape.landscape_class import Protein_Landscape

# Need to find a better way to do the test below

class TestGenLandscape(unittest.TestCase):
    def gen_landscape(self):
        landscape = Protein_Landscape(csv_path="../Data/Small_NK.csv",gen_graph=True)

landscape = Protein_Landscape(csv_path="../Data/Small_NK.csv",gen_graph=True)

class TestQuery(unittest.TestCase):
    def test_string_idx(self):
        assert landscape["AAC"][0] == "AAC", "String indexing has failed"
    def test_int_idx(self):
        assert landscape[26][0] == "ADH", "Integer indexing has failed"
    def test_tuple_idx(self):
        assert landscape[(0,1,1)][0] == "ACC", "Tuple indexing has failed"
    def test_len(self):
        assert len(landscape) == 1000, "__getitem__ method is failing"
    def test_list(self):
        assert landscape[[1,2,4]][2][0] == "AAF", "list indexing has failed"
    def test_array(self):
        assert landscape[np.array([63,87])][1][0] == "AKI", "numpy array indexing has failed"

class TestIndexing(unittest.TestCase):
    def test_positions(self):
        assert len(landscape.indexing(positions=[1,2])) == 99, "Positional Indexing is not working"
    def test_distances(self):
        assert len(landscape.indexing(distances=3)) == 729, "Distance based indexing is not working"
    def test_pos_dist(self):
        assert len(landscape.indexing(positions=[1,2],distances=2)) == 81 and len(landscape.indexing(distances=2)) == 243, "Positional and distance indexing are not working when combined"
    def test_percentage(self):
        assert len(landscape.indexing(percentage=0.7)) == 700, "Percentage indexing is not working"
    def test_pos_dist_perc(self):
        assert len(landscape.indexing(positions=[1,2],distances=2,percentage=0.3)) == 24, "All forms of indexing fail when combined"

class TestDistanceGeneration(unittest.TestCase):
    def test_get_distance_normal(self):
        landscape.get_distance(2)
    def test_gen_d_data(self):
        landscape.gen_d_data(seq="ACL")
    def test_get_distance_custom_d_data(self):
        out = landscape.data[landscape.get_distance(dist=0,d_data=landscape.gen_d_data(seq="ACL"))]
        assert out[0][0] == 'ACL'
    def test_calc_neighnours(self):
        assert np.all(landscape.calc_neighbours(seq="ACL",explicit_neighbours=False)[1] == landscape.calc_neighbours(seq="ACL",explicit_neighbours=True)[1]), "Calc neighbours has an error"

class TestPyTorchDataLoaders(unittest.TestCase):
    def test_pytorch_dataloader_generation(self):
        train, test = landscape.pytorch_dataloaders()
        assert len(next(iter(test))[0]) == 200, "Failed to generate dataloaders with default arguments"

    def test_pytorch_dataloader_generation_indexing(self):
        idxs = landscape.evolved_trajectory_data(num_steps=9)
        train, test = landscape.pytorch_dataloaders(idxs=idxs)
        assert len(next(iter(test))[0]) == 2, "Failed to generate dataloaders when particular indices are provided"

    def test_pytorch_dataloader_distances(self):
        train_dl, test_dl = landscape.pytorch_dataloaders(idxs=landscape.indexing(distances=1))
        assert len(next(iter(test_dl))[0]) == 6, "Failed to generate dataloaders for a single distance"

    def test_pytorch_dataloader_multiple_distances(self):
        train_dl, test_dl = landscape.pytorch_dataloaders(idxs=landscape.indexing(distances=[1,2]))
        assert len(next(iter(test_dl))[0]) == 54, "Failed to generate dataloaders for multiple distances"

    def test_pytorch_dataloader_positions(self):
        train_dl, test_dl = landscape.pytorch_dataloaders(idxs=landscape.indexing(positions=[1,2]))
        assert len(next(iter(test_dl))[0]) == 20, "Failed to generate dataloaders for particular positions"

    def test_pytorch_dataloader_unsupervised(self):
        train_dl, test_dl = landscape.pytorch_dataloaders(unsupervised=True)
        assert torch.all(0 == next(iter(test_dl))[1]) and len(next(iter(test_dl))[1]) == 200, "Failed to generate correct unsupervised dataloaders"

class TestDataSamplingMethods(unittest.TestCase):
    def test_deep_sequence_generation(self):
        idxs = landscape.deep_sequence_data(max_distance=2)
        assert len(idxs) == 270, "Deep Sequence Sampling now has an error"
        assert 111 not in idxs, "An amino acid with a distance from 3 is failing"
    def test_deep_sequence_generation_initial_seq(self):
        idxs = landscape.deep_sequence_data(initial_seq="CAL",max_distance=1)
        assert len(idxs) == 27, "Wrong sequence length for deep sampling"
        assert 9 in idxs, "Not correctly calculating distance of 1 from 'CAL'"


class TestIndexingOperations(unittest.TestCase):
    def test_distance_indexing(self):
        with self.assertRaises(AssertionError):
            landscape.indexing(distances=[1,2,4])
        assert len(landscape.indexing(distances=[1,3])) == 756
    def test_distance_reference_indexing(self):
        assert landscape[landscape.indexing(reference_seq="LDC",positions=[1])][3][0] == "LFC", "Reference indexing not working correctly."

if __name__ == "__main__":
    unittest.main()
