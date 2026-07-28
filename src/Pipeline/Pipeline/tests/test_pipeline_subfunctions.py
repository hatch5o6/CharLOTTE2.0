import pytest
import os


from Pipeline.Pipeline import pipeline
from utilities.utilities import set_vars_in_path

EXP_HOME = set_vars_in_path("&{EXP_HOME}")
CONFIG = "src/configs/test/test.xx_yy-->zz.pipeline.yaml"

def test__method_comparator():
    assert pipeline._method_comparator("charlotte", "web") == -1
    assert pipeline._method_comparator("web", "fuzz") == -1
    assert pipeline._method_comparator("fuzz", "charlotte") == 1
    with pytest.raises(AssertionError):
        pipeline._method_comparator("web", "web")


def test__get_all_scen_OC_afterok():
    class FakeJob:
        def __init__(self, job_id):
            self.id = job_id

    results = {
        "method_a": {
            "scenario_1": {"jobs": {"infer": FakeJob(101)}},
            "scenario_2": {"jobs": {"infer": FakeJob(102)}},
        },
        "method_b": {
            "scenario_1": {"jobs": {"infer": FakeJob(201)}},
        },
    }

    result = pipeline._get_all_scen_OC_afterok(results)

    assert result == "101:102:201"
    
    # single
    results = {
        "method_a": {
            "scenario_1": {"jobs": {"infer": FakeJob(42)}},
        }
    }
    assert pipeline._get_all_scen_OC_afterok(results) == "42"

    # empty
    assert pipeline._get_all_scen_OC_afterok({}) == ""
