# Probabilistic Roadmap (PRM) with Path Shortcutting
This repo is a solution to exercises "Solving a 2D motion planning problem by PRM" and "Post-processing a 2D path" from http://www.osrobotics.org/osr/


## Functions
1. Plan a collision-free path by RPM in an simulated environment with triangular obstacles
2. Post process the path by iterations of path shortcutting
3. Visualize RPM path planning.


## Command Line Usage
Run default setting without path shortcutting
```
python PRMpathplanning.py
```

![Image](https://github.com/bk1021/PRM_pathplanning/blob/main/pictures/output1.png)

Enable path shortcutting with 20 iterations.
```
python PRMpathplanning.py --shortcut --iter 20
```

![Image](https://github.com/bk1021/PRM_pathplanning/blob/main/pictures/output2.png)

Enable nodes and edges plotting
```
python PRMpathplanning.py --plot_nodes --plot_edges
```

![Image](https://github.com/bk1021/PRM_pathplanning/blob/main/pictures/output3.png)


## Reference
1. https://github.com/devanys/PRM-PathPlanning
2. https://github.com/Jayden9912/PRM
3. ChatGPT
