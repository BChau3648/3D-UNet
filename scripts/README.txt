todo:
parameter tuning --> patch shape, stride, filter slice
- try half stride

MAKE PREDICTION AND EVALUATION SCRIPT (DICE SCORE AND HAUSDORFF DISTANCE)

MAYBE TRACK GPU USAGE DURING TRAINING (AND DATA LOADING)
- try to see if zero padding input images will aversely affect training!
- check to see if training from patches is even widely used! (it should be)
- try to see why training with 2 gpus with 10gb is slower than 1 gpu with 16gb   <----- doing this rn and testing to see if that's the case with iso-run_1

look at predictions from first run and calculate dice score and hausdorff distance

Check to see if using right loss and eval score (apparently can't use boht dice loss and dice eval?; but does eval even matter?)
USE GENERALIZED DICE LOSS

weighting background much less

Want to see if good results are mainly coming from background predictions? What do the tumors look like?

Include hausdorff distance as metric
