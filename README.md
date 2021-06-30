# Prograph
GPU accelerated construction and manipulation of protein graphs.

The codebase revolves around the prograph object which functions as a polymorphic graph object, enabling it to integrate fluidly
with a wide variety of machine learning codebases.

## Basic Usage
The prograph object can be instantiated simply by calling it with a csv file and identifying which columns correspond to
the sequences and any associated labels of interest (i.e fitness).

```python
import prograph as pg
pgraph = pg.Prograph(csv_path="data/synthetic_data.csv")
```

Following this, the object can make different forms of data available simple by calling it with different arguments. If a protein label
is passed, such as "seq" or "fitness" then it will return a numpy array of these values for the graph. Alternatively, different ML frameworks can be provided, in which case the object will return its internal data in a manner amenable to that architecture.

```python
sequences = pgraph("seq")
fitnesses = pgraph("fitness")

# Scikit-learn
X_train, Y_train, X_val, Y_val, X_test, Y_test = pgraph("sklearn")

# PyTorch
train_dataloader, val_dataloader, test_dataloader = pgraph("pytorch")
```

## Graph Features

The strength of prograph lies in its internal graph structure which is calculated using GPU accelerated pairwise distances. This structure enables efficient indexing operations to be performed, allowing for sequences a certain distance from a particular (wild type or seed) sequence to be returned or all sequences that are mutated at a particular position. Additionally it can return the distance from any sequence to the underlying dataset.

All of these operations can be arbitrarily composed with the function calls provided above.

```python
train_dl, val_dl, test_dl = pgraph("pytorch",positions=[1,2,4])
# Will isolate the data in which positions 1, 2, and 4 were modified and return dataloaders associated with it.
X_train, Y_train, _, _, X_test, Y_test = pgraph("sklearn",distance=3,positions=[2,9,11],split=[0.8,0.2])
# Returns only train and test for the dataset with all sequences 3 mutations from wild type only mutated at positions 2, 9, and 11.
```
## Under the hood
The graph is stored internally as a pandas dataframe. This frame is what is exported (to a csv) when saved and enables a storage efficient
and flexible internal model that can be reconstructed at will.

The codebase was designed to adhere to the functional paradigm as much as possible, with the class having internally stored data and a wide variety of functions which manipulate this data.

The codebase was designed to be highly extensible, with some distance functions provided along with a cleaners to enable the generation of new distance functions with ease. The distance functions rely on broadcasting to rapidly perform pairwise comparisons, and thus in order to be efficiently vectorized and computed, any new distance functions must conform to this syntax.

The codebase is also integratable with networkx and scipy, enabling graph analytics to be readily performed.

### Todo
Major
- Need to figure out how to build an extensible tokenization system so that it can integrate with methods like TAPE.
- Make cKDTree the CPU bound alternative and also check its scaling.

Minor
- Add support for the generation of tensorflow data loaders (https://www.tensorflow.org/tutorials/load_data/csv)
- Add feature to write out its own graphml object
- Add cosine similarity to distance metrics.
- Add tests for pytorch dataloaders and sklearn data gatherers to ensure that they behave correctly when passed weird arguments.
