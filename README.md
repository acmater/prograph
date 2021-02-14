# Protein Landscape

Package that interacts with structured protein datasets and provides a large
range of functionalities to integrate it with machine learning workflows

### Todo

1. Work out how to handle the seed sequence. Should it be removed or not?
2. Add functionality to export protein landscape graph to cytoscape
3. Need to make all data acquisition schemes utilize the same syntax which allows you to manually specify a data array

### Fundamental Restructuring

The code has two fundamental subcomponents:
1.   The ability to slice the data
2.   The ability to return the data in useful objects to you.

The goal should be to standardize these two with similar call signatures across the two of them.
