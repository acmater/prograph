import os
import pickle
import prograph

def load(name,pgraph=None):
    """
    Functions that instantiates the landscape from a saved file if one is provided

    Parameters
    ----------
    name: str
        Provides the name of the file. MUST contain the extension.

    pgraph : prograph.Prograph, default=None
        An optional protein graph that can be passed and will have its data overwritten.
    """
    print(f"Loading Graph {name}")
    assert type(name) == str, "Wrong type for name, must be string."
    assert os.path.exists(name), "Path does not exist."
    try:
        file = open(name,"rb")
        dataPickle = file.read()
        file.close()
    except Exception as e:
        print("Error occurred during loading of file:", e)

    if pgraph is None:
        pgraph = prograph.Prograph()
        print("Reading data into empty graph")

    pgraph.__dict__ = pickle.loads(dataPickle)
    return pgraph
