<div align="center">
  Computational Intelligence Problems
  <br />
  <a href="#about"><strong>Explore the docs »</strong></a>
  <br />
 </div>

<div align="center">
<br />

[![Project license](https://img.shields.io/github/license/DarthReca/Computational-Intelligence.svg)](LICENSE)
[![code with love by DarthReca](https://img.shields.io/badge/%3C%2F%3E%20with%20%E2%99%A5%20by-DarthReca-ff1414.svg?style=flat-square)](https://github.com/DarthReca)

</div>

<details open="open">
<summary>Table of Contents</summary>

- [About](#about)
- [Getting Started](#getting-started)
- - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Authors & contributors](#authors--contributors)
- [License](#license)
- [Acknowledgements](#acknowledgements)

</details>

## About

The project is a set of simple program written in Python to solve some games. It was built for the course Computational Intelligence attended at Polytechnic University of Turin.

## Getting Started

### Prerequisites

It is necessary to have a version of Python >= 3.0 and the library Numpy.

## Usage

### Sudoku

To solve some random sudokus run `sudoku/main.py` 

### Connect four

To play connect four with the agent run `connect-four/main.py`. If you want to change the type of agent set the `solver` to a different type. 

## Authors & contributors

The original setup of this repository is by [Daniele Rege Cambrin](https://github.com/DarthReca).

## License

This project is licensed under the **MIT license**.

See [LICENSE](LICENSE) for more information.

## Acknowledgements

### Sudoku

The sudoku is solved using Algorithm X proposed in _Donald E. Knuth. (2000). Dancing links._

### Connect four

Connect four agent can use [Minimax](https://en.wikipedia.org/wiki/Minimax) or [Monte Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search).
