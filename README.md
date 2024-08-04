#Probabilistic Roadmap (PRM) with Path Shortcutting

Exercises: "Solving a 2D motion planning problem by PRM" and "Post-processing a 2D path" from http://www.osrobotics.org/osr/

##Functions
1. Plan a collision-free path by RPM in an simulated environment with triangular obstacles
2. Post process the path by iterations of path shortcutting
3. Visualize RPM path planning.

##Command Line Usage
Run default setting without path shortcutting
'''
python PRMpathplanning.py
'''
Enable path shortcutting with 20 iterations.
'''
python PRMpathplanning.py --shortcut --iter 20
'''
Enable nodes and edges plotting
'''
python PRMpathplanning.py --plot_nodes --plot_edges
'''

##Reference
1. https://github.com/devanys/PRM-PathPlanning
2. https://github.com/Jayden9912/PRM
3. ChatGPT
