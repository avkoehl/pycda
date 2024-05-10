Python Cost Distance Analysis

Overview
===
`pycda` is a Python package that performs cost-distance analysis on rioxarray rasters.
Support for both omnidirectional cost-rasters and direction specific costs.

Features
===
- Least-Cost Path: trace the least-cost path between two points on a cost raster
- Cumulative Cost: Generate a cumulative cost raster given a set of one or more origin points
- Least-Cost Allocation: Determine the source point that is connected with the least-cost path to each cell in a raster

![least-cost path image](https://github.com/avkoehl/pycda/blob/main/images/lcp.png?raw=true)
![cumulative cost image](https://github.com/avkoehl/pycda/blob/main/images/accumulated.png?raw=true)

Usage
===
See notebooks in the `/examples` directory
