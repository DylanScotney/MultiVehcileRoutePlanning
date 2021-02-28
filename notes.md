# mVRP Notes

### Vehicle Routing Problem (VRP)
* If n is the number of stops then (n-1)! is the total number of possible
routes
* This is obviously impractical to solve for large number of stops
* Rather than considering all possible tours, heuristic algorithms for
solving the VRP are capable of substantially reducing the number of
tours to be taken into consideration.

### Multiple Vehicle Routing Problem
A generalisation of the VRP is the multiple vehicle routing problem
(mVRP) which consists of determining a set of routes for m vehicles.
The mVRP can in general be defined as follows:
Given a set of nodes, let there be m vehicles located at a single depot
node. The remaining nodes that are to be visited are called intermediate
noes. Then, the mVRP consits of finding tours for all m vehicles which 
all start and end at the depot, such that each intermediate node is
visited exactly once and the total cost of visiting all nodes is 
minimised. 

### Problem Background and Problem Formulation 

For a VRP he objective function is to minimise the sum of all distances
of all the selected nodes of the tour such that
