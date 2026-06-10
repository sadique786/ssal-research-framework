# Contributing Guide

Thank you for contributing to the SSAL Research Framework.

## Branch Strategy

main

* Stable releases only.

develop

* Active integration branch.

feature/*

* New functionality.

experiment/*

* Research experiments.

---

## Coding Standards

* Python 3.11+
* PEP8 compliant
* Type hints encouraged
* Docstrings required for public functions

---

## Commit Convention

feat: add SimCLR module

fix: resolve entropy query bug

docs: update README

refactor: improve trainer abstraction

test: add density peak tests

---

## Pull Request Requirements

* Code runs successfully
* Tests pass
* Documentation updated
* Experiment configuration included

---

## Research Reproducibility

Every experiment must include:

* Random seed
* Dataset
* Hyperparameters
* Query budget
* Query strategy
* Model checkpoint

No result should be reported without reproducible configuration files.

