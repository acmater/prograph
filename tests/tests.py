import unittest
import numpy as np
import torch

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
        out = landscape.data[landscape.get_distance(dist=0,d_data=landscape.gen_d_data(seq="ACL"))]
        assert out[0][0] == 'ACL'

class TestPyTorchDataLoaders(unittest.TestCase):
    def test_pytorch_dataloader_generation(self):
        train, test = landscape.pytorch_dataloaders()
        assert len(next(iter(test))[0]) == 200, "Failed to generate dataloaders with default arguments"

    def test_pytorch_dataloader_generation_indexing(self):
        idxs = landscape.evolved_trajectory_data(num_steps=9)
        train, test = landscape.pytorch_dataloaders(idxs=idxs)
        assert len(next(iter(test))[0]) == 2, "Failed to generate dataloaders when particular indices are provided"

    def test_pytorch_dataloader_distances(self):
        train_dl, test_dl = landscape.pytorch_dataloaders(distance=1)
        assert len(next(iter(test_dl))[0]) == 6, "Failed to generate dataloaders for a single distance"

    def test_pytorch_dataloader_multiple_distances(self):
        train_dl, test_dl = landscape.pytorch_dataloaders(distance=[1,2])
        assert len(next(iter(test_dl))[0]) == 54, "Failed to generate dataloaders for multiple distances"

    def test_pytorch_dataloader_positions(self):
        train_dl, test_dl = landscape.pytorch_dataloaders(positions=[1,2])
        assert len(next(iter(test_dl))[0]) == 20, "Failed to generate dataloaders for particular positions"

    def test_pytorch_dataloader_unsupervised(self):
        train_dl, test_dl = landscape.pytorch_dataloaders(unsupervised=True)
        assert torch.all(0 == next(iter(test_dl))[1]) and len(next(iter(test_dl))[1]) == 200, "Failed to generate correct unsupervised dataloaders"


if __name__ == "__main__":
    unittest.main()
