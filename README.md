## The fixed angle QAOA

The fixed angle conjecture for the quantum approximate optimization algorithm onregular MaxCut graphs

### The universal angles

The universal angles for MaxCut on regular graphs QAOA with guaranteed performance are located in [angles_regular_graphs.json](angles_regular_graphs.json)

#### File structure

```
connectivity (str):
    p (str):
        gamma: list
        beta: list
        AR: float # The approximation ratio guaranteed by the angles
```

#### Usage

```python
def get_angles(p: int, conn: int) -> tuple:
    try:
        gamma_beta = data[str(conn)][str(p)]['angles']
        gamma, beta = gamma_beta[:p], gamma_beta[p:]
        return gamma, beta
    except KeyError:
        print('Angles not found!')
```

### Directory structure

The general approach is to have the data separate from the presentation, à-la MVC pattern.

* `data/*.(nc|json|npy)` - data files
* `data/generators/` - scripts that generate the data 
* `plots/*.ipynb` - scripts that generate figures
* `plots/pdf/` - pdf output from figure generators


### Nice tools used

You'll need these to understand and run the code

* [QTensor](https://github.com/danlkv/qtensor) - tensor network simulator with focus on MaxCut
* [Cartesian Explorer](https://github.com/danlkv/cartesian-explorer/) - a handy tool to map multi-dimensional data
* [xarray](http://xarray.pydata.org/en/stable/) - used to store datasets with their coordinates in [netcdf format](http://xarray.pydata.org/en/stable/getting-started-guide/quick-overview.html?highlight=netcdf#read-write-netcdf-files)
* [Gurobi](https://www.gurobi.com/) and [gurobipy](https://pypi.org/project/gurobipy/) - classical optimization of combinatorial problems.


#### Gurobi installation
See instructions on the official site, the general steps are the following:

1. Register at the website [gurobi.com](https://www.gurobi.com)
2. Unpack an archive 
3. Run licensing code (included in data/generators/*)
