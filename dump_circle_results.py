import pickle
from copy import deepcopy
from test.pt2_to_circle_test.test_pt2_to_circle import (
    convert_nnmodule_to_circle,
    convert_nnmodule_to_pt2,
    convert_pt2_to_circle,
    infer_circle,
    infer_nnmodule,
    validate_result,
    verify_circle,
)
# Florence2 클래스는 프로젝트 내에서 import 한다고 가정
from test.modules.model.Florence2.model import Florence2
import argparse

florence = Florence2()
circle_model_path = 'test/pt2_to_circle_test/artifacts/model/Florence2/model/Florence2.circle' 
forward_args, forward_kwargs = florence.get_example_inputs()
circle_result = infer_circle(
    circle_model_path,
    forward_args=deepcopy(forward_args),
    forward_kwargs=deepcopy(forward_kwargs),
)

# cli에서 인자로 파일명 받아오기
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--filename",
    required=True,
    help="To dump all models in a module, provide a module name.\
            Or you can dump a single class in the module.\
            (e.g. test.modules.op.add, test.modules.op.mean.SimpleMeanKeepDim)",
)
# pickle로 저장
with open(parser.parse_args().filename, "wb") as f:
    pickle.dump(circle_result, f)
