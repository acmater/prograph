import numpy as np

class Protein():
    """
    Python class which handles instances of individual proteins. Protein is
    initialized with a variety of properties and utilises dictionary type syntaxing
    to make it compliant with the remainder of the code.
    """
    def __init__(self, seq,**kwargs):
        self.seq = seq
        for key, value in kwargs.items():
            setattr(self, key, value)

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

    def __getitem__(self,keys):
        if isinstance(keys, list):
            # customized behaviour to allow user to specify tuple of keys
            return tuple([self.__dict__[x] for x in keys])
        return self.__dict__[keys]

    def __len__(self):
        return len(self.seq)

if __name__ == "__main__":
    a = Protein("AAC")
    b = Protein("ACA")
