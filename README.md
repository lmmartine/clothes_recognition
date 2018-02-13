
## Installation

**Matlab with ROS support(> 2015)**

### Matlab with Custom Messages in ROS

roboticsSupportPackages
rosgenmsg(directory)

Follow indications in matlab:

1. Edit javaclasspath.txt, add the following file locations as new lines, and save the file:
 
.../matlab_gen/jar/uchile_msgs-2.0.0.jar
.../matlab_gen/jar/uchile_srvs-2.0.0.jar
 
2. Add the custom message folder to the MATLAB path by executing:
 
addpath('.../matlab_gen/msggen')
savepath
 
3. Restart MATLAB


## Run the demo

download dataset from: ...

compress and move it to home directory i.e. ~/clothes_sequence_dataset

run demo.m




