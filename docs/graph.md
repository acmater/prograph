Data in the Prograph class is constructed in an iterative manner. The fundamental graph structure relies only on
the index and sequence for each protein.

```python
{idx : Protein(sequence) for idx,sequence in enumerate(sequences)}
```

This is techically not a graph structure. This is best described mathematically as a set of protein sequences. The
graph structure is build over time, at first by using the update_graph method to label the proteins in the set
with properties such as attributes as fitness and tokenized structure.

```python
self.update_graph(fitnesses,"fitness")
```

Finally, if `gen_graph` is set to true, then the connectivity is calculated in accordance with the Hamming distance.
This creates a graph structure by calculating the connectivity of the graph with the `build_graph` function.

```python
self.update_graph(self.build_graph(sequences, fitnesses),"neighbours")
```

Once again, the `update_graph` function is used to append a neighbours attribute to each node. This makes random walks
exceedingly easy.
