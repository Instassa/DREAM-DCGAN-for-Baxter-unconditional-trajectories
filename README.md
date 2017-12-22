# DREAM-DCGAN-for-Baxter-unconditional-trajectories
22.12.2017 DREAM report: A module that would take Baxter trajectories produced by quality diversity (QD) search and generate new trajectories based on these.


## Short Summary:
The `format_data_unconditional.py` function takes [archive_3600_trj.dat](https://dream.isir.upmc.fr/databases/ball_throwing_baxter/20170703-01/) and [motion folder](https://dream.isir.upmc.fr/databases/ball_throwing_baxter/20170706-01/) as an input and outputs a data folder in a format digestable by DCGAN function.   
The `main.py` function takes the name of the data folder, training stage (i.e. train/test), plus (optional) specific information, e.g. input/output height/width, a preferred number of training epochs and so on. This function produces the batches of samples (64 in each) that can be then used to control Baxter in Gazebo (or later at some point a real Baxter robot)


## Prerequisites:
* Python 3.3+
* Tensorflow 0.12.1 (GPU configuration)
* SciPy
* pillow
* numpy
* math
* os


## Usage:
After downloading this repository, please add the [archive_3600_trj.dat](https://dream.isir.upmc.fr/databases/ball_throwing_baxter/20170703-01/) and [unzipped motion folder](https://dream.isir.upmc.fr/databases/ball_throwing_baxter/20170706-01/) in `DREAM-DCGAN-for-Baxter-unconditional-trajectories/data` folder.
Then run the following comands:

    cd ./DREAM-DCGAN-for-Baxter-unconditional-trajectories
    python format_data_unconditional.py
    python main.py --dataset formated_unconditional_trajectories --train

These commands are formating data (so it is the dataset of the first 1000 trajectories in the original dataset) and then training DCGANs to generate new trajectories.

Once the model is sufficiently trained, run:

    python main.py --dataset formated_unconditional_trajectories --test
    
This will produce 10 batches (64 samples in each) of the newly generated trajectories for Baxter.


## Related algorithms:
This is based on the following DCGAN code: https://github.com/carpedm20/DCGAN-tensorflow


