current:
relook at dimensions for images and labels (did i forget to incorporate depth so images/labels should have 4 dimensions?)

compare compute dice per channel and generalized dice and figure out why they are not matching (look at how denominator is calculated differently and why)

todo:
Check to see if using right loss and eval score (apparently can't use boht dice loss and dice eval?; but does eval even matter?)

Want to see if good results are mainly coming from background predictions? What do the tumors look like?

Include hausdorff distance as metric

