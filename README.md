Python Cost Distance Analysis
===
`pycda` is a Python package that performs cost-distance analysis on rioxarray rasters.
Support for both omnidirectional cost-rasters and direction specific costs.

Features
===
- Least-Cost Path: trace the least-cost path between two points on a cost raster
![least-cost path image](https://github.com/avkoehl/pycda/blob/main/images/lcp.png?raw=true)

- Cumulative Cost: Generate a cumulative cost raster given a set of one or more origin points
![cumulative cost image](https://github.com/avkoehl/pycda/blob/main/images/accumulated.png?raw=true)

- Least-Cost Allocation: Determine the source point that is connected with the least-cost path to each cell in a raster


Usage
===
Initialize the `CostDistance` class with either a roughness/cost raster or a directed graph. 
Run `trace_path` or `cost_accumulation`.

For more examples see the notebooks in the `/examples` directory
