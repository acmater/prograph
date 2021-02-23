import numpy as np
from molecule import Molecule

class Protein(Molecule):
    """
    Python class which handles instances of individual proteins. Protein is
    initialized with a variety of properties and utilises dictionary type syntaxing
    to make it compliant with the remainder of the code.

    Parameters
    ----------
    seq : str

        The only required argument. Is used to calculate length and to determine if
        two proteins are equivalent.
    """
    def __init__(self, rep,**kwargs):
        self.rep = rep
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def rep(self):
        return self.rep

    def __repr__(self):
        return super().__repr__()

    def __getitem__(self,keys):
        return super().__getitem__(keys)

    def __len__(self):
        return super().__len__()

    def __eq__(self,other):
        return super().__eq__(other)

    """def __eq__(self,prot):
        for attr in vars(self).keys():
            try:
                comp = self[attr] == prot[attr]
                if hasattr(comp, '__iter__'):
                    if all(comp) is not True:
                        return False
                else:
                    if comp is not True:
                        return False
            except:
                return False
        return True
        """
if __name__ == "__main__":
    a = Protein("AAC",fitness=0.3)
    print(len(a))
    print(a.rep)
    b = Protein("ACA")
    c = Protein("AAC")
    print(a==b)
    print(a==c)
    print(a["fitness"])
    print(a.__repr__())
