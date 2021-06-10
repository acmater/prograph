#!/usr/bin/env python3.9

import unittest
import numpy as np
import torch
import os
from prograph.utils import save
from prograph.distance import *

from prograph import Prograph

# Need to find a better way to do the test below

class TestGenpgraph(unittest.TestCase):
    def gen_pgraph(self):
        pgraph = Prograph(file="data/synthetic_data.csv")

pgraph = Prograph(file="data/synthetic_data.csv")

class TestQuery(unittest.TestCase):
    def test_string_idx(self):
        assert pgraph["AAC"]["Sequence"] == "AAC", "String indexing has failed"
    def test_int_idx(self):
        assert pgraph("Sequence")[26] == "ADH", "Integer indexing has failed"
    def test_tuple_idx(self):
        assert pgraph[(1,2,2)]["Sequence"] == "ACC", "Tuple indexing has failed"
    def test_len(self):
        assert len(pgraph) == 1000, "__getitem__ method is failing"
    def test_list(self):
        assert pgraph[[1,2,4]]["Sequence"][2] == "AAD", "list indexing has failed"
    def test_array(self):
        assert pgraph[np.array([63,87])]["Sequence"][87] == "AKI", "numpy array indexing has failed"

class TestIndexing(unittest.TestCase):
    def test_positions(self):
        assert len(pgraph.indexing(positions=[1,2])) == 99, "Positional Indexing is not working"
    def test_distances(self):
        assert len(pgraph.indexing(distances=3)) == 729, "Distance based indexing is not working"
    def test_pos_dist(self):
        assert len(pgraph.indexing(positions=[1,2],distances=2)) == 81 and len(pgraph.indexing(distances=2)) == 243, "Positional and distance indexing are not working when combined"
    def test_percentage(self):
        assert len(pgraph.indexing(percentage=0.7)) == 700, "Percentage indexing is not working"
    def test_pos_dist_perc(self):
        assert len(pgraph.indexing(positions=[1,2],distances=2,percentage=0.3)) == 24, "All forms of indexing fail when combined"
    def test_pos_dist_complement(self):
        assert pgraph.indexing(positions=[1,2],distances=2,complement=True)[1][12] == 30, "The Complement Method is failing"

class TestDistanceGeneration(unittest.TestCase):
    """def test_get_distance_normal(self):
        pgraph.get_distance(2)
    def test_gen_d_data(self):
        pgraph.gen_d_data(seq="ACL")
    def test_get_distance_custom_d_data(self):
        out = pgraph[pgraph.get_distance(dist=0,d_data=pgraph.gen_d_data(seq="ACL"))]
        assert out[19]["Sequence"] == 'ACL'"""
    def test_calc_neighnours(self):
        assert np.all(pgraph.calc_neighbours(seq="ACL") == pgraph["ACL"]["Neighbours"]), "Calc neighbours has an error"

class TestPyTorchDataLoaders(unittest.TestCase):
    def test_pytorch_dataloader_generation(self):
        train, test = pgraph.pytorch_dataloaders()
        assert len(next(iter(test))[0]) == 200, "Failed to generate dataloaders with default arguments"

    def test_pytorch_dataloader_generation_indexing(self):
        idxs = np.arange(10)
        train, test = pgraph.pytorch_dataloaders(idxs=idxs)
        assert len(next(iter(test))[0]) == 2, "Failed to generate dataloaders when particular indices are provided"

    def test_pytorch_dataloader_distances(self):
        train_dl, test_dl = pgraph.pytorch_dataloaders(idxs=pgraph.indexing(distances=1))
        assert len(next(iter(test_dl))[0]) == 6, "Failed to generate dataloaders for a single distance"

    def test_pytorch_dataloader_multiple_distances(self):
        train_dl, test_dl = pgraph.pytorch_dataloaders(idxs=pgraph.indexing(distances=[1,2]))
        assert len(next(iter(test_dl))[0]) == 54, "Failed to generate dataloaders for multiple distances"

    def test_pytorch_dataloader_positions(self):
        train_dl, test_dl = pgraph.pytorch_dataloaders(idxs=pgraph.indexing(positions=[1,2]))
        assert len(next(iter(test_dl))[0]) == 20, "Failed to generate dataloaders for particular positions"

    def test_pytorch_dataloader_unsupervised(self):
        train_dl, test_dl = pgraph.pytorch_dataloaders(unsupervised=True)
        assert torch.all(0 == next(iter(test_dl))[1]) and len(next(iter(test_dl))[1]) == 200, "Failed to generate correct unsupervised dataloaders"

class TestLabelIter(unittest.TestCase):
    def test_label_iter(self):
        with self.assertRaises(KeyError):
            pgraph.label_iter("PLK")

class TestIndexingOperations(unittest.TestCase):
    def test_distance_indexing(self):
        with self.assertRaises(AssertionError):
            pgraph.indexing(distances=[1,2,4])
        assert len(pgraph.indexing(distances=[1,3])) == 756
    def test_distance_reference_indexing(self):
        assert pgraph[pgraph.indexing(reference_seq="LDC",positions=[1])]["Sequence"][901] == "LAC", "Reference indexing not working correctly."

class TestNetworkx(unittest.TestCase):
    def test_networkx_generation(self):
        pgraph.graph_to_networkx(labels=["Fitness","Tokenized"],update_self=True)
        assert "Fitness" in pgraph.networkx_graph.nodes["AAA"].keys(), "Networkx graph generation is not working."

class TestLoadPrograph(unittest.TestCase):
    def test_load_pgraph(self):
        pgraph = Prograph(file="data/synthetic_data_pgraph.pkl")
        assert pgraph[0]["Fitness"] == 0.660972597708149, "Loaded graph is not functioning correctly."
    def test_load_wrong_type(self):
        with self.assertRaises(AssertionError) and self.assertRaises(FileNotFoundError):
            pgraph = Prograph(file=2)
    def test_load_empty(self):
        with self.assertRaises(FileNotFoundError):
            pgraph = Prograph(file=None)

class TestSavePrograph(unittest.TestCase):
    def test_save_pgraph(self):
        pgraph = Prograph(file="data/synthetic_data.csv")
        assert save(pgraph,name="test",directory="./"), "Protein graph could not be saved correctly"
        new_pgraph = Prograph(file="test.pkl")
        assert new_pgraph[0]["Sequence"] == "AAA", "Graph loaded following saving is not functioning correctly."
        os.remove("test.pkl")

class TestTokenization(unittest.TestCase):
    def test_basic_tokenization(self):
        assert np.all(pgraph.tokenize("ACA") == np.array([1,2,1])), "Basic tokenization is not working correctly."
    def test_batch_tokenization(self):
        assert np.all(pgraph.tokenize(["ACA","ACC"]) == np.array([[1,2,1],[1,2,2]])), "Batch tokenization is not working."
    def test_variable_length_tokenization(self):
        tokens = pgraph.tokenize(["ACCCACAAA","ACAA"])
        assert np.all(tokens == np.array([[1, 2, 2, 2, 1, 2, 1, 1, 1],[1, 2, 1, 1, 0, 0, 0, 0, 0]])), "Variable length tokenization is not working correctly."
    def test_empty_tokenization(self):
        assert len(pgraph.tokenize([])) == 0, "Tokenizing an empty object is not returning an empty object."

class TestMatrixGeneration(unittest.TestCase):
    def test_sparse_generation(self):
        assert np.all(pgraph.sparse().todense()[:3,:3] == np.array([[0,1,1],[1,0,1],[1,1,0]])), "Sparse matrix generation is not working correctly."

class TestDistanceCalculators(unittest.TestCase):
    # There needs to be a better way to check equality for tensors. See if I can use the assertAlmostEqual unittest method of pytorch tensors.
    def test_2d2d_hamming(self):
        X = torch.Tensor([[1,2,3],[4,5,6]])
        Y = torch.Tensor([[1,2,3],[7,8,9]])
        assert torch.all(hamming(X,Y) == torch.Tensor([[0,3],[3,3]])), "Hamming distance calculator not working for two 2D tensors."
    def test_2d1d_hamming(self):
        X = torch.Tensor([[1,2,3],[4,5,6]])
        Y = torch.Tensor([1,2,3])
        assert torch.all(hamming(X,Y) == torch.Tensor([[0,3]])), "Hamming distance calculator not working for one 1D tensor and one 2D tensor."
    def test_1d1d_hamming(self):
        X = torch.Tensor([4,5,6])
        Y = torch.Tensor([1,2,3])
        assert torch.all(hamming(X,Y) == torch.Tensor([[3]])), "Hamming distance calculator not working for one 1D tensor and one 1D tensor."
    def test_1dempty_hamming(self):
        X = torch.Tensor([4,5,6])
        Y = torch.Tensor()
        with self.assertRaises(ValueError):
            hamming(X,Y), "Empty vectors are not throwing value errors."
    def test_2d2d_minkowski(self):
        X = torch.Tensor([[1,2,3],[4,5,6]])
        Y = torch.Tensor([[1,2,3],[7,8,9]])
        assert torch.all(minkowski(X,Y) == torch.Tensor([[0.0000,5.19615242],[10.3923048454,  5.19615242]])), "minkowski distance calculator not working for two 2D tensors."
    def test_2d1d_minkowski(self):
        X = torch.Tensor([[1,2,3],[4,5,6]])
        Y = torch.Tensor([1,2,3])
        assert torch.all(minkowski(X,Y) == torch.Tensor([[0.0000,5.19615242]])), "minkowski distance calculator not working for one 1D tensor and one 2D tensor."
    def test_1d1d_minkowski(self):
        X = torch.Tensor([4,5,6])
        Y = torch.Tensor([1,2,3])
        assert torch.all(minkowski(X,Y) == torch.Tensor([[5.19615242]])), "minkowski distance calculator not working for one 1D tensor and one 1D tensor."
    def test_1dempty_minkowski(self):
        X = torch.Tensor([4,5,6])
        Y = torch.Tensor()
        with self.assertRaises(ValueError):
            minkowski(X,Y), "Empty vectors are not throwing value errors."

if __name__ == "__main__":
    unittest.main()
