from abc import ABC, abstractmethod
import numpy as np

class Molecule(ABC):
    """
    Python abstract class for a molecule. Protein is
    initialized with a variety of properties and utilises dictionary type syntaxing
    to make it compliant with the remainder of the code.

    Parameters
    ----------
    rep :

        The only required argument. Is used to calculate length and to determine if
        two proteins are equivalent.
    """
    def __init__(self,rep,**kwargs):
        self.rep = rep
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def __repr__(self):
        parsed = []
        for key, value in vars(self).items():
            if isinstance(value,np.ndarray):
                parsed.append(f"{key}=np.array({list(value)})")
            elif isinstance(value, str):
                parsed.append(f"{key}='{value}'")
            else:
                parsed.append(f"{key}={value}")
        return f"Protein(" + ",".join(parsed) + ")"

    @abstractmethod
    def __getitem__(self,keys):
        if isinstance(keys, list):
            # customized behaviour to allow user to specify tuple of keys
            return tuple([self.__dict__[x] for x in keys])
        return self.__dict__[keys]

    @abstractmethod
    def __len__(self):
        return len(self.rep)

    @abstractmethod
    def __eq__(self,other):
        return True if self.rep == other.rep else False

if __name__ == "__main__":
    pass
    #test = Molecule("ABC")
