import json
import os
import numpy as np
from typing import List, Tuple
from PIL import Image, ImageDraw
import re
import sys


class MiouCalculator:
    def __init__(self, json_path: str, img_size: Tuple[int, int]):
        self.json_path = json_path
        self.img_size = img_size  # 图像尺寸 (width, height)

    def polygon_to_mask(self, polygon: List[Tuple[float, float]]) -> np.ndarray:
        """
        将多边形转换为二值掩码。
        Args:
            polygon (List[Tuple[float, float]]): 多边形顶点坐标。
        Returns:
            np.ndarray: 二值掩码，shape 为 (height, width)。
        """
        if len(polygon) < 3:  # 至少需要 3 个点形成闭合多边形
            print(f"Warning: Polygon has insufficient points: {polygon}")
            return np.zeros(self.img_size, dtype=np.uint8)

        # 直接使用原始像素坐标
        mask = Image.new('L', self.img_size, 0)  # 创建黑色图像
        draw = ImageDraw.Draw(mask)
        try:
            draw.polygon(polygon, outline=1, fill=1)  # 绘制多边形
        except Exception as e:
            print(f"Error drawing polygon: {polygon}, error: {e}")
            return np.zeros(self.img_size, dtype=np.uint8)

        return np.array(mask)

    def compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        计算两个二值掩码的 IoU（交并比）。
        Args:
            mask1 (np.ndarray): 掩码1。
            mask2 (np.ndarray): 掩码2。
        Returns:
            float: IoU 值。
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def extract_polygon_from_rbox(self, rbox_str: str) -> List[List[Tuple[float, float]]]:
        """
        从 <rbox> 中提取多边形点。
        Args:
            rbox_str (str): 包含 <rbox> 的字符串。
        Returns:
            List[List[Tuple[float, float]]]: 多边形点的列表。
        """
        pattern = r"<rbox>\(\{(.*?)\}\)</rbox>"
        matches = re.findall(pattern, rbox_str)
        polygons = []
        for match in matches:
            points = re.findall(r"<(-?\d+\.?\d*),(-?\d+\.?\d*)>", match)
            polygon = [(float(x), float(y)) for x, y in points]
            polygons.append(polygon)
        return polygons

    def calculate_miou_for_sentence(self, gt_str: str) -> float:
        """
        计算单个句子的 Miou。
        Args:
            gt_str (str): ground truth 中的多边形字符串。
        Returns:
            float: 该句子的 Miou。
        """
        gt_polygons = self.extract_polygon_from_rbox(gt_str)
        ious = []

        for gt in gt_polygons:
            gt_mask = self.polygon_to_mask(gt)
            pred_mask = self.polygon_to_mask(gt)  # 使用 ground truth 得到的 mask 作为预测
            iou = self.compute_iou(gt_mask, pred_mask)
            ious.append(iou)

        return np.mean(ious) if ious else 0.0

    def calculate_miou(self) -> Tuple[List[float], float]:
        """
        计算 JSON 文件中所有句子的 Miou。
        Returns:
            Tuple[List[float], float]: 每个句子的 Miou 和总体 Miou。
        """
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return [], 0.0

        print(f"Loaded {len(data)} entries from JSON.")
        sentence_mious = []

        for entry in data:
            if isinstance(entry, list):  # 如果数据是嵌套在列表中的
                entry = entry[0]

            gt_str = entry.get('ground_truth', '')

            if "<rbox>" in gt_str:
                sentence_miou = self.calculate_miou_for_sentence(gt_str)
                sentence_mious.append(sentence_miou)

        overall_miou = np.mean(sentence_mious) if sentence_mious else 0.0
        return sentence_mious, overall_miou

    def visualize(self, gt_str: str, output_dir: str, file_name: str):
        """
        可视化 ground truth 多边形并保存为图像。
        Args:
            gt_str (str): 包含 <rbox> 的 ground truth 字符串。
            output_dir (str): 保存文件夹路径。
            file_name (str): 保存文件名。
        """
        gt_polygons = self.extract_polygon_from_rbox(gt_str)

        # 创建空白图像
        img = Image.new("RGB", self.img_size, "white")
        draw = ImageDraw.Draw(img)

        for polygon in gt_polygons:
            # 直接绘制多边形到图像上
            if len(polygon) >= 3:  # 至少需要 3 个点
                draw.polygon(polygon, outline="red", fill="blue")
            else:
                print(f"Skipping invalid polygon: {polygon}")

        os.makedirs(output_dir, exist_ok=True)  # 确保输出文件夹存在
        output_path = os.path.join(output_dir, file_name)
        img.save(output_path)
        # print(f"Saved visualization to {output_path}")

    @staticmethod
    def run(json_path: str, img_size: Tuple[int, int], visualize: bool = False, visualize_dir: str = "./visualizations"):
        """
        运行 Miou 计算并打印结果。
        Args:
            json_path (str): JSON 文件路径。
            img_size (Tuple[int, int]): 图像尺寸。
            visualize (bool): 是否可视化结果。
            visualize_dir (str): 可视化结果保存文件夹路径。
        """
        miou_calculator = MiouCalculator(json_path, img_size)
        sentence_mious, overall_miou = miou_calculator.calculate_miou()

        print("Per-sentence Miou:")
        for i, miou in enumerate(sentence_mious):
            print(f"Sentence {i+1}: {miou:.4f}")
        print(f"Overall Miou: {overall_miou:.4f}")

        if visualize:
            with open(json_path, 'r') as f:
                data = json.load(f)
            for i, entry in enumerate(data):
                if isinstance(entry, list):  # 如果数据是嵌套在列表中的
                    entry = entry[0]
                gt_str = entry.get('ground_truth', '')
                if "<rbox>" in gt_str:
                    file_name = f"visualization_{i+1}.png"
                    miou_calculator.visualize(gt_str, visualize_dir, file_name)

def redirect_output_to_file(log_file: str):
    """
    将控制台输出重定向到文件。
    Args:
        log_file (str): 日志文件路径。
    """
    sys.stdout = open(log_file, "w")
    sys.stderr = sys.stdout

def main():
    # JSON 文件路径和图像尺寸
    json_path = "/data1/zilun/grsm/SkySenseGPT/QA_generate/datasets/all_qa_data-test.json"  # 替换为您的 JSON 文件路径
    img_size = (256, 256)  # 图像尺寸 (宽, 高)
    visualize_dir = "./output_visualizations"  # 可视化输出文件夹

    # 运行 Miou 计算
    MiouCalculator.run(json_path, img_size, visualize=True, visualize_dir=visualize_dir)
    # 输出结果将打印到控制台，并保存到日志文件中
    log_file = "./miou_calculator_log.txt"  # 日志文件路径

    # 重定向控制台输出到日志文件
    redirect_output_to_file(log_file)

if __name__ == "__main__":
    main()
