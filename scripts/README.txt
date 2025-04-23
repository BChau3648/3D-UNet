todo:
check transformations on data

use 2 gpu with maybe 10gb of memory

look at predictions from first run and calculate dice score and hausdorff distance

Check to see if using right loss and eval score (apparently can't use boht dice loss and dice eval?; but does eval even matter?)
USE GENERALIZED DICE LOSS

weighting background much less

preprocess kidney ct data; check to see if i can use jupyter lab from pygeneral. run interactive node on pygenral env --> jupyter lab?

Want to see if good results are mainly coming from background predictions? What do the tumors look like?

Include hausdorff distance as metric

filter slice builder

change patch shape, stride shape, halo?