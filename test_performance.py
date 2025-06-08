import time
import psutil
import json
import os
import argparse
import csv
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
import pytesseract
from PIL import Image
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import nltk
nltk.download('punkt', quiet=True)

class PerformanceTester:
    def __init__(self, sample_limit: int = None):
        self.results_dir = "results"
        self.sample_limit = sample_limit
        os.makedirs(self.results_dir, exist_ok=True)
    
    def calculate_metrics(self, original_text: str, restored_text: str) -> Dict:
        """Calculate all performance metrics between original and restored text."""
        if not original_text or not restored_text:
            return {
                "item_accuracy": 0.0,
                "char_accuracy": 0.0,
                "average_normalized_levenshtein": 0.0,
                "average_bleu_score": 0.0,
                "average_rouge_rouge-1": 0.0,
                "average_rouge_rouge-2": 0.0,
                "average_rouge_rouge-l": 0.0,
                "average_length_long": 0.0,
                "average_length_medium": 0.0,
                "average_length_short": 0.0,
                "average_size_large": 0.0,
                "average_type_textType1": 0.0,
                "average_region_bottom": 0.0
            }
        # Tokenize texts
        original_tokens = nltk.word_tokenize(original_text)
        restored_tokens = nltk.word_tokenize(restored_text)
        # Calculate item accuracy (exact match)
        item_accuracy = 1.0 if original_text == restored_text else 0.0
        # Calculate character accuracy
        char_accuracy = sum(1 for a, b in zip(original_text, restored_text) if a == b) / max(len(original_text), len(restored_text))
        # Calculate normalized Levenshtein distance
        from Levenshtein import distance
        max_len = max(len(original_text), len(restored_text))
        if max_len == 0:
            normalized_levenshtein = 0.0
        else:
            normalized_levenshtein = 1 - (distance(original_text, restored_text) / max_len)
        # Calculate BLEU score
        try:
            bleu_score = sentence_bleu([original_tokens], restored_tokens)
        except:
            bleu_score = 0.0
        # Calculate ROUGE scores
        try:
            rouge = Rouge()
            scores = rouge.get_scores(restored_text, original_text)
            rouge_1 = scores[0]['rouge-1']['f']
            rouge_2 = scores[0]['rouge-2']['f']
            rouge_l = scores[0]['rouge-l']['f']
        except:
            rouge_1 = rouge_2 = rouge_l = 0.0
        # Length metrics (example: classify based on token count)
        token_count = len(original_tokens)
        if token_count > 20:
            length_long = 1.0
            length_medium = 0.0
            length_short = 0.0
        elif token_count > 10:
            length_long = 0.0
            length_medium = 1.0
            length_short = 0.0
        else:
            length_long = 0.0
            length_medium = 0.0
            length_short = 1.0
        # Size and type metrics (example: dummy values)
        size_large = 1.0 if len(original_text) > 100 else 0.0
        type_textType1 = 1.0  # Dummy value
        region_bottom = 1.0    # Dummy value
        return {
            "item_accuracy": item_accuracy,
            "char_accuracy": char_accuracy,
            "average_normalized_levenshtein": normalized_levenshtein,
            "average_bleu_score": bleu_score,
            "average_rouge_rouge-1": rouge_1,
            "average_rouge_rouge-2": rouge_2,
            "average_rouge_rouge-l": rouge_l,
            "average_length_long": length_long,
            "average_length_medium": length_medium,
            "average_length_short": length_short,
            "average_size_large": size_large,
            "average_type_textType1": type_textType1,
            "average_region_bottom": region_bottom
        }
    
    def extract_text_from_label(self, label_data: Dict) -> str:
        """Extract text from label data"""
        if not label_data or 'annotations' not in label_data:
            return ""
        
        # Sort annotations by vertical position (top to bottom)
        annotations = sorted(label_data['annotations'], 
                           key=lambda x: x['annotation.bbox'][1])
        
        # Extract text from annotations
        text_parts = []
        current_line = []
        current_y = None
        
        for ann in annotations:
            text = ann['annotation.text']
            bbox = ann['annotation.bbox']
            y_pos = bbox[1]
            
            # If this is a new line
            if current_y is None or abs(y_pos - current_y) > 10:  # 10 pixel threshold
                if current_line:
                    text_parts.append(' '.join(current_line))
                current_line = [text]
                current_y = y_pos
            else:
                current_line.append(text)
        
        # Add the last line
        if current_line:
            text_parts.append(' '.join(current_line))
        
        return '\n'.join(text_parts)
    
    def find_corresponding_label(self, image_path: str) -> str:
        """Find the corresponding label file for an image"""
        parts = image_path.split('/')
        if 'images' in parts:
            idx = parts.index('images')
            parts[idx] = 'labels'
            label_path = '/'.join(parts)
            label_path = os.path.splitext(label_path)[0] + '.json'
            return label_path
        return None
    
    def load_label_data(self, label_path: str) -> Dict:
        """Load and parse label data"""
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found: {label_path}")
            return None
        
        with open(label_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def measure_performance(self, func, *args, **kwargs) -> Dict:
        """Measure performance metrics for a given function"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        return {
            "processing_time": end_time - start_time,
            "memory_usage": memory_used,
            "result": result
        }
    
    def get_test_images(self, base_dir: str = None, specific_image: str = None) -> List[str]:
        """Get list of test images with optional limit"""
        if specific_image:
            if os.path.exists(specific_image):
                return [specific_image]
            else:
                print(f"Error: Specified image file not found: {specific_image}")
                return []
        
        image_paths = []
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
        
        if self.sample_limit:
            image_paths = image_paths[:self.sample_limit]
        
        return image_paths
    
    def test_mlm_method(self, image_path: str, label_data: Dict = None) -> Dict:
        """Test MLM-based restoration using Tesseract"""
        try:
            # Open and preprocess image
            image = Image.open(image_path)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image, lang='kor+eng')
            
            original_text = self.extract_text_from_label(label_data) if label_data else ""
            metrics = self.calculate_metrics(original_text, text)
            return {
                "method": "MLM",
                "status": "tested",
                "extracted_text": text,
                "original_text": original_text,
                "metrics": metrics
            }
        except Exception as e:
            print(f"Error in MLM method: {str(e)}")
            return {
                "method": "MLM",
                "status": "failed",
                "extracted_text": f"Error: {str(e)}",
                "original_text": self.extract_text_from_label(label_data) if label_data else "",
                "metrics": self.calculate_metrics("", "")
            }
    
    def test_ocr_denoising(self, image_path: str, label_data: Dict = None) -> Dict:
        """Test OCR denoising method using Tesseract with preprocessing"""
        try:
            # Open image
            image = Image.open(image_path)
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Extract text using Tesseract with custom config
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, config=custom_config, lang='kor+eng')
            
            original_text = self.extract_text_from_label(label_data) if label_data else ""
            metrics = self.calculate_metrics(original_text, text)
            return {
                "method": "OCR Denoising",
                "status": "tested",
                "extracted_text": text,
                "original_text": original_text,
                "metrics": metrics
            }
        except Exception as e:
            print(f"Error in OCR denoising method: {str(e)}")
            return {
                "method": "OCR Denoising",
                "status": "failed",
                "extracted_text": f"Error: {str(e)}",
                "original_text": self.extract_text_from_label(label_data) if label_data else "",
                "metrics": self.calculate_metrics("", "")
            }
    
    def test_vision_based(self, image_path: str, label_data: Dict = None) -> Dict:
        """Test vision-based restoration using Tesseract with advanced preprocessing"""
        try:
            # Open image
            image = Image.open(image_path)
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Extract text using Tesseract with advanced config
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            text = pytesseract.image_to_string(image, config=custom_config, lang='kor+eng')
            
            original_text = self.extract_text_from_label(label_data) if label_data else ""
            metrics = self.calculate_metrics(original_text, text)
            return {
                "method": "Vision-based",
                "status": "tested",
                "extracted_text": text,
                "original_text": original_text,
                "metrics": metrics
            }
        except Exception as e:
            print(f"Error in vision-based method: {str(e)}")
            return {
                "method": "Vision-based",
                "status": "failed",
                "extracted_text": f"Error: {str(e)}",
                "original_text": self.extract_text_from_label(label_data) if label_data else "",
                "metrics": self.calculate_metrics("", "")
            }
    
    def run_comparison_test(self, image_path: str) -> Dict:
        """Run comparison test for all methods"""
        label_path = self.find_corresponding_label(image_path)
        label_data = self.load_label_data(label_path) if label_path else None
        
        results = {
            "image_path": image_path,
            "label_path": label_path,
            "methods": {}
        }
        
        # Test MLM
        mlm_results = self.measure_performance(self.test_mlm_method, image_path, label_data)
        results["methods"]["mlm"] = mlm_results
        
        # Test OCR Denoising
        ocr_results = self.measure_performance(self.test_ocr_denoising, image_path, label_data)
        results["methods"]["ocr_denoising"] = ocr_results
        
        # Test Vision-based
        vision_results = self.measure_performance(self.test_vision_based, image_path, label_data)
        results["methods"]["vision_based"] = vision_results
        
        return results
    
    def save_results(self, results: Dict, image_name: str):
        """Save test results to both JSON and TXT files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"performance_test_{image_name}_{timestamp}"
        
        # Save JSON
        json_filename = f"{self.results_dir}/{base_filename}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save TXT files for each method
        txt_files = []
        for method, result in results["methods"].items():
            method_filename = f"{self.results_dir}/{method}_{image_name}_{timestamp}.txt"
            with open(method_filename, 'w', encoding='utf-8') as f:
                extracted_text = result['result'].get('extracted_text', 'No text extracted')
                f.write(extracted_text)
            txt_files.append(method_filename)
        
        return json_filename, txt_files
    
    def generate_performance_report(self, all_results: List[Dict]) -> Dict:
        """Generate performance report from all test results"""
        report = {
            "mlm": {
                "avg_time": 0,
                "avg_memory": 0,
                "success_rate": 0,
                "text_matches": 0,
                "metrics": {}
            },
            "ocr_denoising": {
                "avg_time": 0,
                "avg_memory": 0,
                "success_rate": 0,
                "text_matches": 0,
                "metrics": {}
            },
            "vision_based": {
                "avg_time": 0,
                "avg_memory": 0,
                "success_rate": 0,
                "text_matches": 0,
                "metrics": {}
            }
        }
        
        # Calculate averages
        for result in all_results:
            for method in report.keys():
                if method in result["methods"]:
                    method_result = result["methods"][method]
                    report[method]["avg_time"] += method_result["processing_time"]
                    report[method]["avg_memory"] += method_result["memory_usage"]
                    if method_result["result"].get("status") == "tested":
                        report[method]["success_rate"] += 1
                    
                    # Compare extracted text with original
                    extracted = method_result["result"].get("extracted_text", "")
                    original = method_result["result"].get("original_text", "")
                    if extracted and original and extracted == original:
                        report[method]["text_matches"] += 1
                    metrics = method_result["result"].get("metrics", {})
                    for metric_name, value in metrics.items():
                        if metric_name not in report[method]["metrics"]:
                            report[method]["metrics"][metric_name] = 0
                        report[method]["metrics"][metric_name] += value
        
        # Calculate final averages
        num_samples = len(all_results)
        for method in report.keys():
            report[method]["avg_time"] /= num_samples
            report[method]["avg_memory"] /= num_samples
            report[method]["success_rate"] = (report[method]["success_rate"] / num_samples) * 100
            report[method]["text_matches"] = (report[method]["text_matches"] / num_samples) * 100
            for metric_name in report[method]["metrics"]:
                report[method]["metrics"][metric_name] /= num_samples
        
        return report
    
    def save_performance_report(self, report: Dict) -> Tuple[str, str, str]:
        """Save performance report to JSON, TXT, and CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"performance_report_{timestamp}"
        
        # Save JSON
        json_filename = f"{self.results_dir}/{base_filename}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save TXT
        txt_filename = f"{self.results_dir}/{base_filename}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("Performance Report Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for method, metrics in report.items():
                f.write(f"Method: {method}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average Processing Time: {metrics['avg_time']:.2f} seconds\n")
                f.write(f"Average Memory Usage: {metrics['avg_memory']:.2f} MB\n")
                f.write(f"Success Rate: {metrics['success_rate']:.2f}%\n")
                f.write(f"Text Match Rate: {metrics['text_matches']:.2f}%\n")
                for metric_name, value in metrics['metrics'].items():
                    f.write(f"{metric_name}: {value:.4f}\n")
                f.write("\n")
        
        # Save CSV
        csv_filename = f"{self.results_dir}/{base_filename}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['model_name', 'text_detection', 'text_recognition', 'preprocessing_steps',
                     'item_accuracy', 'char_accuracy', 'inference_time', 'average_type_textType1', 'average_region_bottom', 'average_length_short', 'average_size_large', 'average_normalized_levenshtein', 'average_bleu_score', 'average_rouge_rouge-1', 'average_rouge_rouge-2', 'average_rouge_rouge-l', 'average_length_long', 'average_length_medium']
            writer.writerow(header)
            
            # Write data
            for method, metrics in report.items():
                row = [
                    method,
                    method,  # text_detection
                    method,  # text_recognition
                    'no_preprocessing',  # preprocessing_steps
                    metrics['metrics'].get('item_accuracy', 0),  # item_accuracy
                    metrics['metrics'].get('char_accuracy', 0),  # char_accuracy
                    metrics['avg_time'],  # inference_time
                    metrics['metrics'].get('average_type_textType1', 0),  # average_type_textType1
                    metrics['metrics'].get('average_region_bottom', 0),  # average_region_bottom
                    metrics['metrics'].get('average_length_short', 0),  # average_length_short
                    metrics['metrics'].get('average_size_large', 0),  # average_size_large
                    metrics['metrics'].get('average_normalized_levenshtein', 0),  # average_normalized_levenshtein
                    metrics['metrics'].get('average_bleu_score', 0),  # average_bleu_score
                    metrics['metrics'].get('average_rouge_rouge-1', 0),  # average_rouge_rouge-1
                    metrics['metrics'].get('average_rouge_rouge-2', 0),  # average_rouge_rouge-2
                    metrics['metrics'].get('average_rouge_rouge-l', 0),  # average_rouge_rouge-l
                    metrics['metrics'].get('average_length_long', 0),  # average_length_long
                    metrics['metrics'].get('average_length_medium', 0)  # average_length_medium
                ]
                writer.writerow(row)
        
        return json_filename, txt_filename, csv_filename

def main():
    parser = argparse.ArgumentParser(description='Run OCR restoration performance tests')
    parser.add_argument('--limit', type=int, help='Limit the number of test samples')
    parser.add_argument('--data-dir', type=str, default='data/test/images',
                      help='Directory containing test images')
    parser.add_argument('--image', type=str,
                      help='Specific image file to test')
    args = parser.parse_args()
    
    tester = PerformanceTester(sample_limit=args.limit)
    
    # Get test images
    test_images = tester.get_test_images(args.data_dir, args.image)
    if not test_images:
        print("No test images found. Exiting.")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Run tests
    all_results = []
    for image_path in tqdm(test_images, desc="Running tests"):
        results = tester.run_comparison_test(image_path)
        all_results.append(results)
        
        # Save individual results
        image_name = os.path.basename(image_path)
        json_file, txt_files = tester.save_results(results, image_name)
        print(f"Test results saved to:\n  JSON: {json_file}")
        for txt_file in txt_files:
            print(f"  TXT: {txt_file}")
    
    # Generate and save performance report
    report = tester.generate_performance_report(all_results)
    json_file, txt_file, csv_file = tester.save_performance_report(report)
    print(f"Performance report saved to:\n  JSON: {json_file}\n  TXT: {txt_file}\n  CSV: {csv_file}")

if __name__ == "__main__":
    main() 