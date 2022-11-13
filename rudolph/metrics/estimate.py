import os
import json
from pathlib import Path
from typing import Dict, Optional, Any, List

# Vqa
from rudolph.metrics.vqa.calc_meteor import calculate_meteor as calc_meteor_vqa

# Captioning
from rudolph.metrics.captioning.ruclip_score import calc_ruclip_metric
from rudolph.metrics.captioning.calc_meteor import calculate_meteor as calc_meteor_cap


def _get_data(file_path: str) -> Optional[Dict[str, Any]]:
    file_path = Path(file_path)
    if file_path.exists():
        try:
            with open(str(file_path), 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Data loaded: {len(data)}")
            return data
        except Exception as e:
            print(f"Error during data json loading:\n{e}")
            return None
    else:
        raise FileNotFoundError(f"Data file do not exists by path: {file_path} !")


def _separate_joint_json(joined_json: Dict[str, List[dict]]):
    pred_json = {}
    gt_json = {}
    for sample_idx, results in joined_json.items():
        results = results[0]
        pred_json[sample_idx] = [{'type': results['type'],
                                  'content': results['content']}]
        gt_json[sample_idx] = [{'type': results['type'],
                                'content': results['content'],
                                'left_text': results['left_text'],
                                'right_text': results['right_text'],
                                'task_label': results['task_label']
                                }]
    return pred_json, gt_json


class Estimator:

    @staticmethod
    def estimate_captioning(pred_filename: str, gt_filename: str = None,
                            single_file: bool = True,
                            images_folder: str = None):
        """
        Calculate METEOR and RuCLIP scores:
        return: (METEOR + RuCLIP) / 2
        """
        # {sample_idx: [{'type': 'text', 'content': 'generated text',
        # 'left_text': 'left text', 'right_text': 'right text', 'task_label': 'task_label as str'}]}
        pred_data = _get_data(pred_filename)
        if single_file:
            pred_data, gt_data = _separate_joint_json(pred_data)
        elif gt_filename is not None:
            gt_data = _get_data(pred_filename)
        else:
            raise AttributeError("Ground truth data file were not provided!")

        # METEOR
        meteor_score = calc_meteor_cap(gt_data, pred_data)
        print(f"METEOR score for Captioning: {round(meteor_score, 3)}")

        # RuCLIP
        # if images_folder is not None:
        #     clip_score = calc_ruclip_metric(gt_filename, pred_filename, image_folder=images_folder)
        #     print(f"ruCLIP score for Captioning: {round(clip_score, 3)}")
        #     return (meteor_score + clip_score) / 2
        return meteor_score

    def estimate_vqa(self, pred_filename: str, gt_filename: str = None,
                     single_file: bool = True):
        """
        Calculate METEOR and RuCLIP score for VQA task.
        """
        pred_data = _get_data(pred_filename)
        if single_file:
            pred_data, gt_data = _separate_joint_json(pred_data)
        elif gt_filename is not None:
            gt_data = _get_data(pred_filename)
        else:
            raise AttributeError("Ground truth data file were not provided!")

        # METEOR
        meteor_score = calc_meteor_vqa(gt_data, pred_data)
        print(f"METEOR score for Captioning: {round(meteor_score, 3)}")
        return meteor_score


if __name__ == '__main__':
    root_dir = Path().resolve().parent.parent.parent
    results_path = root_dir / "outputs" / "results_2tasks"
    cap_pred_filename = results_path / "no_sp_loss_2tasks_1epoch_preds_test_data_task-captioning.json"
    estimator = Estimator()
    estimator.estimate_captioning(str(cap_pred_filename))
