# 비교용 Florence2 inference pickle 스크립트 사용법 예시
import pickle
import torch

# Florence2 클래스는 프로젝트 내에서 import 한다고 가정
from test.modules.model.Florence2.model import Florence2

def dump_inference_results(pickle_path):
    florence = Florence2()
    inputs, _ = florence.get_example_inputs()

    # 중간 값 추출: 예시로 입력, 모델 output 등
    result_dict = {
        'input_ids': inputs[0].cpu().numpy(),
        'pixel_values': inputs[1].cpu().numpy(),
        'attention_mask': inputs[2].cpu().numpy(),
        'decoder_input_ids': inputs[3].cpu().numpy(),
    }

    with torch.no_grad():
        # 예시: model의 중간 값 출력은 직접 분리하거나, Hook 등 사용
        output = florence.forward(*inputs)
        for key, value in output.items():
            # tensor만 저장
            if isinstance(value, torch.Tensor):
                result_dict["output_" + key] = value.cpu().numpy()

    # Pickle dump
    with open(pickle_path, "wb") as f:
        pickle.dump(result_dict, f)
    print(f"Dumped inference results to {pickle_path}")

if __name__ == "__main__":
    # 사용 예시: python script.py /tmp/torch260_florence2.pkl
    import sys
    dump_inference_results(sys.argv[1])
