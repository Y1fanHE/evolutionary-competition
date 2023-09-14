# Evolutionary Competition

> compete the "evolution" and evolve the "competition"

## Generating Benchmark by GP

Generate benchmark functions to differ a pair of evolutionary algorithms based on genetic programming algorithm

![flowchart](flow.svg)

### Requirements

- Python 3.9
- deap
- scipy
- matplotlib

### Usage

See `example.py`

![eg](img/eg1.svg)
![eg](img/eg2.svg)
![eg](img/eg3.svg)
![eg](img/eg4.svg)

### Evolutionary Algorithms

- random search
- differential evolution
- particle swarm optimization
- cuckoo search

### Distance Metrics

- wasserstein distance

## Todo

- [ ] write README file
- [ ] check high dimension cases
- [ ] add multiprocessing
- [ ] implement map-elite gp
- [ ] add more eas
- [ ] add new metrics
