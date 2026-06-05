# silica-water-NNP

This repository was forked from the public arcann_training repository. It was modified for training a silica and water NNP. The steps followed for setting up the required directories and files for training the silica and water NNP are listed below.

steps:
1. create a directory for training
2. create a data and user_files subdirectories in the training directory
3. create a directory starting with init in the data folder
4. move initial data set to the data folder 
example folder set up:
  data/
    init_silica_water/
      type.raw
      box.raw       optional
      coord.raw     optional
      energy.raw    optional
      force.raw     optional
      set.000/
        box.npy
        coord.npy
        energy.npy
        force.npy
5. create lammps input file from the example user_files template
6. create propterties.txt with same ordering as type.raw but use 1 based indexing
7. copy machine.json and modify project_name
8. copy training_3.0.json and modify type_map to have the same ordering as type.raw and properties.txt then make guesses for sel
9. Rename training_3.0.json to dptrain_3.0.json 
10. change subpartition: "ib" to subpartition: "ib,intel" for cpu nodes in machine.json
11. copy slurm scripts from Elle's repository
12. create cp2k file using cp2k template and old cp2k file
13. create .lmp file from frame 10000
14. remove references to plumed files and python exploration script in slurm scripts
15. in lammps slurm scripts uncommented the gpu line