import os
import pickle
import prograph

def save(pgraph,name=None,ext=".pkl",directory=None):
    """
    Save function that stores the entire landscape so that it can be reused without
    having to recompute distances and tokenizations

    Parameters
    ----------
    pgraph : prograph.Prograph
        The protein graph that will be saved.

    name : str, default=None
        Name that the class will be saved under. If none is provided it defaults
        to the same name as the csv file provided if it exists or the template name "pgraph"

    ext : str, default=".pkl"
        Extension that the file will be saved with.

    directory : str, default=None
        The directory that the object will be saved to. If None, defaults to the location of the csv file if
        it exists or the local directory.
    """
    if directory is None:
        if hasattr(pgraph,'csv_path'):
            directory, file = pgraph.csv_path.rsplit("/",1)
            directory += "/"
        else:
            directory, file = "./", "pgraph"
    if not name:
        name = file.rsplit(".",1)[0]
    print(f"Saving Graph to {name + ext}")
    pgraph.graph[[x for x in pgraph.graph if x != "tokenized"]].to_csv("data/test.csv")
    try:
        file = open(directory+name+ext,"wb")
        file.write(pickle.dumps(pgraph.__dict__))
        file.close()
    except Exception as e:
        print("Error occurred during saving:", e)
    return True
