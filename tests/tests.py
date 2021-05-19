import unittest
import numpy as np
import torch
import os
from prograph.utils import save, load

from prograph import Prograph

# Need to find a better way to do the test below

class TestGenpgraph(unittest.TestCase):
    def gen_pgraph(self):
        pgraph = Prograph(csv_path="data/synthetic_data.csv",gen_graph=True)

pgraph = Prograph(csv_path="data/synthetic_data.csv",gen_graph=True)

class TestQuery(unittest.TestCase):
    def test_string_idx(self):
        assert pgraph["AAC"]["seq"] == "AAC", "String indexing has failed"
    def test_int_idx(self):
        assert pgraph[26]["seq"] == "ADH", "Integer indexing has failed"
    def test_tuple_idx(self):
        assert pgraph[(0,1,1)]["seq"] == "ACC", "Tuple indexing has failed"
    def test_len(self):
        assert len(pgraph) == 1000, "__getitem__ method is failing"
    def test_list(self):
        assert pgraph[[1,2,4]][2]["seq"] == "AAD", "list indexing has failed"
    def test_array(self):
        assert pgraph[np.array([63,87])][87]["seq"] == "AKI", "numpy array indexing has failed"

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
    def test_get_distance_normal(self):
        pgraph.get_distance(2)
    def test_gen_d_data(self):
        pgraph.gen_d_data(seq="ACL")
    def test_get_distance_custom_d_data(self):
        out = pgraph[pgraph.get_distance(dist=0,d_data=pgraph.gen_d_data(seq="ACL"))]
        assert out[19]["seq"] == 'ACL'
    def test_calc_neighnours(self):
        assert np.all(pgraph.calc_neighbours(seq="ACL") == pgraph["ACL"]["neighbours"]), "Calc neighbours has an error"

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
        assert pgraph[pgraph.indexing(reference_seq="LDC",positions=[1])][901]["seq"] == "LAC", "Reference indexing not working correctly."

class TestNetworkx(unittest.TestCase):
    def test_networkx_generation(self):
        pgraph.graph_to_networkx(labels=["Fitness","tokenized"],update_self=True)
        assert "Fitness" in pgraph.networkx_graph.nodes["AAA"].keys(), "Networkx graph generation is not working."

class TestLoadPrograph(unittest.TestCase):
    def test_load_pgraph(self):
        pgraph = Prograph(saved_file="data/synthetic_data.pkl")
        assert pgraph[0]["Fitness"] == 0.660972597708149, "Loaded graph is not functioning correctly."
    def test_load_wrong_type(self):
        with self.assertRaises(AssertionError) and self.assertRaises(FileNotFoundError):
            pgraph = Prograph(saved_file=2)
    def test_load_empty(self):
        pgraph = Prograph(saved_file=None)
        assert pgraph.__dict__ == {}, "The initialized graph is not empty."

class TestSavePrograph(unittest.TestCase):
    def test_save_pgraph(self):
        pgraph = Prograph(saved_file="data/synthetic_data.pkl")
        assert save(pgraph,name="test",directory="./"), "Protein graph could not be saved correctly"
        new_pgraph = load("test.pkl")
        assert new_pgraph[0]["seq"] == "AAA", "Graph loaded following saving is not functioning correctly."
        os.remove("test.pkl")

if __name__ == "__main__":
    unittest.main()
