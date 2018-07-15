### Differential Evolution

Modify from blake01/[fyp](https://github.com/blake01/fyp). Thanks a lot.

I want to improve differential evolution algorithm by reinforcement learning but it failed. Sad...

### What is it

Differential Evolution(DE) algorithm is a powerful evolutionary optimizer for the continuous parameter spaces which utilizes NP D-dimensional parameter vectors, so-called individuals. The canonial DE algorithm as followed:

- Initialization

  $x_{i,0}^j=x^j_{min}+rand(0,1)*(x^j_{max}-x^j_{min})\ j=1,2,..,D$

- Mutation operation

  After initialization, DE employs the mutation operation to produce a mutant vector. There are many mutation strategies like "DE/best/1":
  $$
  V_{i,G}=X_{best,G}+F*(X_{r_1^j,G}-X_{r^i_2,G})
  $$

- Crossover operation

  After the mutation phase, crossover operation is applied to each pair of the target vector $X_{i,G}$ and its corresponding mutant vector $V_{i,G}$ to generate a trial vector $U_{i,G}$:
  $$
  u_{i,G}^j=\left\{\begin{matrix}
  v_{i,G}^j\quad if (rand_j[0,1]\leq CR) or (j=j_{rand})
  \\ 
  x_{i,G}^j\quad otherwise
  \end{matrix}\right.
  j=1,2,..,D
  $$

- Selection operation

  After crossover operation, keep individuals with lower fitness value.
  $$
  X_{i,G+1}=\left\{\begin{matrix}
  U_{i,G},\quad if f(U_{i,G})\leq f(X_{i,G})
  \\ 
  X_{i,G},\quad otherwise
  \end{matrix}\right.
  $$


You can know more about DE by reading [this](https://www.sciencedirect.com/science/article/pii/S2210650216000146) or related paper.

### File Structure

- 'population.py' provides a general population class, suitable for use in any population-based evolutionary algorithm. 'tests.py' are unit tests for this file.
- 'deBase.py' contains object-oriented versions of the traditional, non-adaptive DE algorithms, as originally proposed by Storn and Price.
- 'mixins.py' provides mixins to extend/edit the DE classes in some way, e.g. logging output.
- 'sade.py', 'jade.py' and 'jde.py' contain adaptive variants of DE from literature, implemented as subclasses of one or more of the deBase.py classes.
- 'hybrids.py' contains cannibalised algorithms derived from SaDE, JADE and jDE.
- 'deAbaqus.py' contains AbaqusJADE and the AbaqusProblem class.
- 'testFunctions' contains the benchmark cost functions detailed in Appendix 2 of the main report.
- 'study.py' is the file used to perform the benchmarking analysis undertaken in Section 2 of the main report.
- 'beamProblem.py' and 'bloodhoundProblem.py' are the trial and objective FEA optimisation files.
- 'abaqusFiles/bloodhound.cae' is the Abaqus CAE file for the Bloodhound jet fuel tank baffle.
- 'DE_DDQN.py' is invented by me. It uses DDQN(one of reinforcement algorithms) to improve DE.
- 'DE_RL.py' is also invented by me. It uses Q-learning(one of reinforcement algorithms) to improve DE. However, all of them are not performed well.

### Outputs

You can see some metrics in the following files to check your algorithm performence.

study.out: run_name, bestVector

xlsx file：functionEvaluations, bestVector.cost, cr, f

### Connect

[cnmengnan@gmail.com](mailto:cnmengnan@gmail.com)

blog: [WinterColor blog](http://www.cnblogs.com/mengnan/)

Good luck