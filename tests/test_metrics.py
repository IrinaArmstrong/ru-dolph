import os
import json
import unittest
from pathlib import Path
from typing import Dict, Optional, Any

# Vqa
from rudolph.metrics.vqa import calc_meteor as calc_meteor_vqa

# Captioning
from rudolph.metrics.captioning.ruclip_score import calc_ruclip_metric
from rudolph.metrics.captioning import calc_meteor as calc_meteor_cap


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


class TestMetricsCalculation(unittest.TestCase):
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self.root_dir = Path().resolve().parent
        self.results_path = self.root_dir.parent / "outputs" / "results_2tasks" / "modified_rudolph_1eposh_2tasks"

    def estimate_vqa(self, pred_filename: str, gt_filename: str) -> float:
        vqa_filename = self.results_path / "gt_test_vqa_data_p0.json"

        pred_data = _get_data(pred_filename)
        gt_data = _get_data(gt_filename)
        # METEOR
        meteor_score = calc_meteor_vqa(gt_data, pred_data)
        print(f"METEOR score for VQA: {round(meteor_score, 3)}")
        return meteor_score

    def estimate_captioning(self, pred_filename: str, gt_filename: str,
                            images_folder: str = None) -> float:
        pred_data = _get_data(pred_filename)
        gt_data = _get_data(gt_filename)
        # METEOR
        meteor_score = calc_meteor_cap(gt_data, pred_data)
        print(f"METEOR score for Captioning: {round(meteor_score, 3)}")

        # RuCLIP
        if images_folder is not None:
            clip_score = calc_ruclip_metric(gt_filename, pred_filename, image_folder=images_folder)
            print(f"ruCLIP score for Captioning: {round(clip_score, 3)}")
            return (meteor_score + clip_score) / 2
        return meteor_score


if __name__ == '__main__':



    # captioning files
    cap_gt_filename = results_path / "gt_test_cap_data_p0.json"
    cap_images_folder = Path("E:/DATA/ImageCaptioning/captioning_images")


    unittest.main()
