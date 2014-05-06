fyp
===

Sheffield Mechanical Engineering MEng Final Year Project:
"Finite Element Optimisation of Bolted Joint Locations using Differential Evolution"
by Blake Hemingway.

Thanks for showing an interest in the code behind this project.


Summary:

Structural design optimisation using Finite Element Analysis (FEA) is a laborious and time-consuming process, but is ubiquitous in industry. Therefore, this study examined the feasibility of using a class of evolutionary algorithm known as Differential Evolution (DE) to automate this process in the general case, with specific emphasis on the optimisation of bolted joint locations.

Firstly, the performance of the most widely-cited DE algorithms in literature was reviewed. Based on independent testing, the JADE algorithm was selected and modified for use as an FEA optimiser. Subsequently, a simple and accessible optimisation API was developed for the Abaqus FEA package, and was used to successfully solve a real-world bolted joint optimisation problem.

The study concluded that the use of DE in FEA is viable for small linear FEA models, and is useful for yielding optimum and perhaps non-obvious design solutions. However, real-world optimisation problems requiring FEA are typically large-scale and nonlinear, and the large amount of CPU time required may limit the industrial usefulness of this approach at present. Nevertheless, given the current rate of computer hardware development, it is hypothesised that that this technique may become commercially viable within the next 10-20 years.


Software and conventions used:

All algorithms were developed in the Python programming language for reasons of open access, code readability and experience. The Abaqus 6.11 FEA package was used as the finite element solver as it features a built-in Python Application Programming Interface (API).

Although it is widely accepted in the Python community to use the lower_case_with_underscores variable naming convention, the mixedCase convention was used instead for consistency with the Abaqus API.


Files in this repository:

- 'population.py' provides a general population class, suitable for use in any population-based evolutionary algorithm.
- 'deBase.py' contains object-oriented versions of the traditional, non-adaptive DE algorithms, as originally proposed by Storn and Price.
- 'mixins.py' provides mixins to extend/edit the DE classes in some way, e.g. logging output.
- 'sade.py', 'jade.py' and 'jde.py' contain adaptive variants of DE from literature, implemented as subclasses of one or more of the deBase.py classes.
- 'hybrids.py' contains cannibalised algorithms derived from SaDE, JADE and jDE.
- 'deAbaqus.py' contains AbaqusJADE and the AbaqusProblem class.
- 'testFunctions' contains the benchmark cost functions detailed in Appendix 2 of the main report.
- 'study.py' is the file used to perform the benchmarking analysis undertaken in Section 2 of the main report.
- 'beamProblem.py' and 'bloodhoundProblem.py' are the trial and objective FEA optimisation files.
- 'bloodhound.cae' is the Abaqus CAE file for the Bloodhound jet fuel tank baffle.
- 'test.sh' is the script file used to run the Bloodhound job on the University of Sheffield's 'Iceberg' supercomputer.
