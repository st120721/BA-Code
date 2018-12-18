from hyperopt import hp
import numpy as  np

class Sequential_Feature_Selector():
    type_sequential_forward_selector = ["Sequential Forward Selection (SFS)", "Sequential Backward Selection (SBS)",
                                        "Sequential Forward Floating Selection(SFFS)",
                                        "Sequential Backward Floating Selection(SBFS)"]
    list_parameter = ["estimator", "k_features", "forward", "floating", "scoring"]
    sfs_param_space = {"k_features": hp.choice("k_features", np.arange(1, 3, 1, dtype=int).tolist()),
                       "forward": hp.choice("forward", [True, False]),
                       "floating": hp.choice("floating", [True, False]),
                       }
    @staticmethod
    def sfs_definition(forward,floating):
        if forward == True and floating == False:
            sfs = "SFS"
        elif forward == False and floating == False:
            sfs = "SBS"
        elif forward == True and floating == True:
            sfs = "FSFS"
        elif forward == False and floating == True:
            sfs = "FBFS"
        return sfs
