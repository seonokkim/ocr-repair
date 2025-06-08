import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import (
    BertForMaskedLM, BertTokenizer,
    RobertaForMaskedLM, RobertaTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    TrOCRProcessor, VisionEncoderDecoderModel,
    DonutProcessor
)

class MethodTester:
    def __init__(self):
        # Initialize models
        self.bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        self.roberta_model = RobertaForMaskedLM.from_pretrained("roberta-base")
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        
        self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        self.donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        self.donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        
        # Set models to eval mode
        for model in [self.bert_model, self.roberta_model, self.t5_model, 
                     self.trocr_model, self.donut_model]:
            model.eval()

    def test_mlm(self, text):
        """Test MLM methods (BERT and RoBERTa)"""
        # BERT
        bert_inputs = self.bert_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            bert_outputs = self.bert_model(**bert_inputs)
            bert_predictions = bert_outputs.logits.argmax(dim=-1)
        bert_restored = self.bert_tokenizer.decode(bert_predictions[0], skip_special_tokens=True)
        
        # RoBERTa
        roberta_inputs = self.roberta_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            roberta_outputs = self.roberta_model(**roberta_inputs)
            roberta_predictions = roberta_outputs.logits.argmax(dim=-1)
        roberta_restored = self.roberta_tokenizer.decode(roberta_predictions[0], skip_special_tokens=True)
        
        return {
            "bert": bert_restored,
            "roberta": roberta_restored
        }

    def test_ocr_correction(self, text):
        """Test OCR correction methods (T5)"""
        input_text = f"fix OCR errors: {text}"
        inputs = self.t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.t5_model.generate(
                inputs["input_ids"],
                max_length=512,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        
        restored_text = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return restored_text

    def test_vision(self, image_path):
        """Test vision-based methods (TrOCR and Donut)"""
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # TrOCR
        trocr_pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            trocr_outputs = self.trocr_model.generate(trocr_pixel_values)
            trocr_text = self.trocr_processor.batch_decode(trocr_outputs, skip_special_tokens=True)[0]
        
        # Donut
        task_prompt = "<s_docvqa><s_question>What is the text in this document?</s_question><s_answer>"
        donut_pixel_values = self.donut_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            donut_outputs = self.donut_model.generate(
                donut_pixel_values,
                max_length=512,
                early_stopping=True,
                pad_token_id=self.donut_processor.tokenizer.pad_token_id,
                eos_token_id=self.donut_processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=4,
                bad_words_ids=[[self.donut_processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True
            )
        donut_sequence = self.donut_processor.batch_decode(donut_outputs.sequences)[0]
        donut_sequence = donut_sequence.replace(self.donut_processor.tokenizer.eos_token, "").replace(self.donut_processor.tokenizer.pad_token, "")
        donut_text = donut_sequence.split("<s_answer>")[-1].split("</s_answer>")[0]
        
        return {
            "trocr": trocr_text,
            "donut": donut_text
        }

def main():
    # Initialize tester
    tester = MethodTester()
    
    # Paths
    image_path = "/root/ocr-repair/data/test/images/5350224/1996/5350224-1996-0001-0037.jpg"
    label_path = "/root/ocr-repair/data/test/labels/5350224/1996/5350224-1996-0001-0037.json"
    output_dir = "/root/ocr-repair/data/test/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth
    with open(label_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    # Test MLM
    mlm_results = tester.test_mlm(ground_truth['text'])
    
    # Test OCR correction
    ocr_results = tester.test_ocr_correction(ground_truth['text'])
    
    # Test vision-based methods
    vision_results = tester.test_vision(image_path)
    
    # Save results in different formats
    
    # 1. JSON format
    json_results = {
        "image_path": image_path,
        "ground_truth": ground_truth,
        "mlm_results": mlm_results,
        "ocr_results": ocr_results,
        "vision_results": vision_results
    }
    with open(os.path.join(output_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    # 2. Text format
    with open(os.path.join(output_dir, "results.txt"), 'w', encoding='utf-8') as f:
        f.write("=== Ground Truth ===\n")
        f.write(ground_truth['text'])
        f.write("\n\n=== MLM Results ===\n")
        f.write(f"BERT: {mlm_results['bert']}\n")
        f.write(f"RoBERTa: {mlm_results['roberta']}\n")
        f.write("\n=== OCR Results ===\n")
        f.write(ocr_results)
        f.write("\n\n=== Vision Results ===\n")
        f.write(f"TrOCR: {vision_results['trocr']}\n")
        f.write(f"Donut: {vision_results['donut']}\n")
    
    # 3. Markdown format
    with open(os.path.join(output_dir, "results.md"), 'w', encoding='utf-8') as f:
        f.write("# Text Restoration Results\n\n")
        f.write("## Ground Truth\n\n")
        f.write(ground_truth['text'])
        f.write("\n\n## MLM Results\n\n")
        f.write(f"### BERT\n{mlm_results['bert']}\n\n")
        f.write(f"### RoBERTa\n{mlm_results['roberta']}\n\n")
        f.write("## OCR Results\n\n")
        f.write(ocr_results)
        f.write("\n\n## Vision Results\n\n")
        f.write(f"### TrOCR\n{vision_results['trocr']}\n\n")
        f.write(f"### Donut\n{vision_results['donut']}\n")

if __name__ == "__main__":
    main() 