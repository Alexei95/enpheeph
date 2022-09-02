import captum
import captum.attr

class CaptumSensitivityAnalysis(object):
    SENSITIVITY_CLASSES = {
        # it computes sensitivity of output neurons, so it works for all layer types
        "LayerConductance": captum.attr.LayerConductance
    }
    def __init__(self, sensitivity_class):
        try:
            self.sensitivity_class = self.SENSITIVITY_CLASSES[sensitivity_class]
        except KeyError:
            raise ValueError(f"unknown sensitivity classes, choices are {','.join(self.SENSITIVITY_CLASSES.keys())}")

    def run_analysis(self, model, test_input, layers):
        for l in layers:
            attr_instance = self.sensitivity_class(forward_func=model, layer=l.module)
            # test_input should contain the keys "inputs" and "target"
            attr = attr_instance.attribute(**test_input)
            l.set_sensitivity_analysis_result(attr.mean(dim=0))
