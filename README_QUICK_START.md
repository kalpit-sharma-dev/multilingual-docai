# 🚀 PS-05 Quick Start Guide

## 🎯 **What This Project Does**

This is a **document understanding system** that processes images/documents and:
1. **Detects layout** (text, tables, images, etc.)
2. **Extracts text** with language identification
3. **Understands content** (describes tables, charts, images)

## 🏃‍♂️ **Quick Start - Run Everything in 3 Steps**

### **Step 1: Check System Status**
```bash
python scripts/run_stages.py --status
```

### **Step 2: Test Advanced Models (Optional)**
```bash
python scripts/run_stages.py --test-advanced
```

### **Step 3: Run Your Desired Stage**

#### **Option A: Run Stage 1 Only (Layout Detection)**
```bash
python scripts/run_stages.py --stage 1 --input data/test/your_image.png
```

#### **Option B: Run Stage 2 Only (Text Extraction)**
```bash
python scripts/run_stages.py --stage 2 --input data/test/your_image.png
```

#### **Option C: Run Stage 3 Only (Content Understanding)**
```bash
python scripts/run_stages.py --stage 3 --input data/test/your_image.png
```

#### **Option D: Run ALL Stages (Complete Pipeline)**
```bash
python scripts/run_stages.py --stage all --input data/test/your_image.png
```

## 📁 **What Each Stage Does**

### **Stage 1: Layout Detection** 🎯
- **Input**: Image/document
- **Output**: JSON with bounding boxes for text, tables, images, etc.
- **Uses**: Your existing YOLOv8 model
- **Result**: `results/stage1_results/layout_results.json`

### **Stage 2: Text Extraction** 📝
- **Input**: Image/document
- **Output**: JSON with extracted text + language identification
- **Uses**: Your existing OCR + language detection
- **Result**: `results/stage2_results/text_results.json`

### **Stage 3: Content Understanding** 🧠
- **Input**: Image/document
- **Output**: JSON with natural language descriptions
- **Uses**: Advanced models (if available) or basic processing
- **Result**: `results/stage3_results/content_results.json`

## 🔧 **Setup Requirements**

### **Basic Setup (Stages 1 & 2)**
```bash
# Install basic dependencies
pip install -r requirements.txt
```

### **Advanced Setup (Stage 3)**
```bash
# Install advanced models
pip install transformers datasets fasttext
```

## 📊 **Example Usage**

### **1. Place a test image**
```bash
# Put your test image in data/test/
mkdir -p data/test
# Copy your image to data/test/sample.png
```

### **2. Run complete pipeline**
```bash
python scripts/run_stages.py --stage all --input data/test/sample.png
```

### **3. Check results**
```bash
# Results will be in:
ls results/
# ├── stage1_results/
# ├── stage2_results/
# └── stage3_results/
```

## 🆘 **Troubleshooting**

### **"Input file not found"**
```bash
# Check if your image exists
ls data/test/
# Use --input to specify correct path
python scripts/run_stages.py --stage 1 --input /path/to/your/image.png
```

### **"Advanced models not available"**
```bash
# Install dependencies
pip install transformers datasets fasttext
# Test again
python scripts/run_stages.py --test-advanced
```

### **"Stage failed"**
```bash
# Check system status
python scripts/run_stages.py --status
# Check logs for specific errors
```

## 📋 **File Structure (Simplified)**

```
multilingual-docai/
├── scripts/
│   └── run_stages.py          # 🎯 MAIN SCRIPT - Use this!
├── core/                      # Core functionality
├── backend/                   # Backend services
├── configs/                   # Configuration files
├── data/                      # Your input data
└── results/                   # Output results
```

## 🎉 **Success Indicators**

### **Stage 1 Success**
- ✅ "Stage 1 completed successfully!"
- 📄 Results in `results/stage1_results/layout_results.json`

### **Stage 2 Success**
- ✅ "Stage 2 completed successfully!"
- 📄 Results in `results/stage2_results/text_results.json`

### **Stage 3 Success**
- ✅ "Stage 3 completed successfully!"
- 📄 Results in `results/stage3_results/content_results.json`

### **All Stages Success**
- 🎉 "ALL STAGES COMPLETED SUCCESSFULLY!"
- 📁 All results in `results/` directory

## 🚀 **Next Steps After Running**

1. **Check results** in the output JSON files
2. **Train on DocLayNet** (if you want to improve performance)
3. **Customize** for your specific use case
4. **Submit** to PS-05 competition!

## 💡 **Pro Tips**

- **Start with Stage 1** to test basic functionality
- **Use --status** to see what's available
- **Use --test-advanced** to check advanced models
- **Check logs** if something fails
- **Use --help** to see all options

---

## 🆘 **Still Confused?**

**Just run this one command to see everything:**
```bash
python scripts/run_stages.py --help
```

**This will show you all available options and examples!**

---

## 📓 Evaluation Day (A100 GPU, Docker)

See the detailed runbook:
```
docs/EVALUATION_DAY_RUNBOOK.md
```