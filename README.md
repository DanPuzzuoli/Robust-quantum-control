# Robust quantum control

(This file last edited 03-12-2018)

The goal of this project is to implement algorithms for finding robust control sequences for quantum systems in Python. The computational methods are based on work to be published shortly with coauthors Holger Haas and David Cory. The project is in its very early stages; much of what I would consider the "core" functionality of this project still needs to be added, and as such the structure of the project and its interfaces are liable to change. Nevertheless, the code is functional, and successfully finds control sequences implementing gates that satisfy certain robustness criteria. 

## Documentation and examples

Documentation is still being built; almost all files and functions are documented, however there is no comprehensive project description/outline. This is primarily because the code will likely change heavily over the next few months. 

Current project-level documentation and examples can be found in: 
* docs/Project_description.pdf contains a rough project description, as well as notes on future functionality
* docs/Examples.pdf contains a description and plotted outputs of example control finding problems that are implemented as functions in control_problems.py.

## Next features

As already mentioned, docs/Project_description.pdf contains notes on future functionality to be implemented, but the main next things to implement are:

* Hessian computation
	Currently the code computes the functions and their Jacobians, and uses SciPy's implementation of the BFGS algorithm to perform the search. The BFGS algorithm approximates the Hessian based on previous Jacobian calls. Directly computing the Hessian will be more expensive (though not necessarily that much more expensive), but can potentially lead to searches converging with far fewer evaluations. (I am basing these statements on the work of Goodwin and Kuprov. I need to think about the possible benefits a bit more, but at the very least their work seems to suggest that there will be large benefits to using a Newton-type method with exact Hessian computation. As measured by Goodwin and Kuprov, the best algorithm for pulse finding in their scenarios is dubbed Newton (RFO), which is a Newton method using a particular way to deal with possible numerical issues with the Hessian. This is not implemented in SciPy, though the second best one in their paper, dubbed Newton (TRM), is, so it's a good first algorithm to use after Hessian computation is completed.)
* Investigate efficiency of specialized exponentiation protocols for upper triangular block matrices
	The vast majority of computational cost in this project comes from expoentiation of block matrices that have upper triangular block structure. Furthermore, it is common for a large number of the blocks to be either 0, or to be redundant (e.g. in most cases the diagonal blocks will all be the same). Hence, in principle, the number of matrix multiplications required to exponentiate these matrices will be far fewer than an arbitrary block matrix of this size (even when accounting for the upper triangular block structure). Hence, its worth exploring options for implementing specialized exponentiation procedures.
* Handling redundancies in computation
	There are many potential redundancies that can be eliminated in these computations. Many boil down to the previous point about exponentiation, but other simple ones also exist; e.g. the base system computation is a special case of the computation for the system that gives deriatives, or that computes decoupling terms. This is vague, but a solution that I think solves this issue is to have a data structure representing a hierarchy of "control systems", where control systems that are redundant will potentially call those lower in hierarchy to extract information when needed. The benefit of this approach is that it won't require the functions that propagate the systems to "know" anything about how the systems are related. They will simply call the relevant control_system objects as needed, and the redundancy handling will be set up as required by each control problem. (I put this bullet after the previous one, even though it is a more general problem, as I think the previous point is more "fundamental" to the computation, and its conclusion will influence how other redundancies are dealt with.)


## Contributors/Acknowledgements

Daniel Puzzuoli, Ian Hincks

This project has also benefitted from discussion with Holger Haas.