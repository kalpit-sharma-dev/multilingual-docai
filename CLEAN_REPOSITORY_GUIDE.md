# 🧹 Clean Repository Guide - PS-05

## 🎯 **What I've Done to Clean Up the Confusion**

I've **simplified and organized** your repository to remove all confusion. Here's what's changed:

### **✅ KEPT (Essential Files)**
- **`scripts/run_stages.py`** - 🎯 **MAIN SCRIPT** - Use this for everything!
- **`scripts/cleanup.py`** - Clean up temporary files
- **`README_QUICK_START.md`** - Simple, clear instructions
- **`core/models/advanced_models.py`** - Advanced models (optional)
- **`core/evaluation/enhanced_evaluator.py`** - Enhanced evaluation (optional)
- **`scripts/ingest_datasets.py`** - Dataset download (optional)

### **🗑️ REMOVED (Confusing Files)**
- ~~`scripts/quick_start_advanced.py`~~ - Too complex
- ~~`scripts/comprehensive_evaluation.py`~~ - Too complex
- ~~`docs/ADVANCED_FEATURES_IMPLEMENTATION.md`~~ - Too detailed

## 🚀 **NOW YOU HAVE ONLY ONE SCRIPT TO RUN EVERYTHING!**

### **The One Script You Need:**
```bash
python scripts/run_stages.py
```

## 📋 **Complete Usage Guide**

### **1. Clean Up Repository (First Time Only)**
```bash
python scripts/cleanup.py
```

### **2. Check System Status**
```bash
python scripts/run_stages.py --status
```

### **3. Run Your Desired Stage**

#### **Stage 1: Layout Detection**
```bash
python scripts/run_stages.py --stage 1 --input data/test/your_image.png
```

#### **Stage 2: Text Extraction**
```bash
python scripts/run_stages.py --stage 2 --input data/test/your_image.png
```

#### **Stage 3: Content Understanding**
```bash
python scripts/run_stages.py --stage 3 --input data/test/your_image.png
```

#### **All Stages: Complete Pipeline**
```bash
python scripts/run_stages.py --stage all --input data/test/your_image.png
```

## 🎯 **What Each Stage Does (Simplified)**

### **Stage 1: Layout Detection** 🎯
- **Input**: Image/document
- **Output**: JSON with bounding boxes
- **Uses**: Your existing YOLOv8 model
- **Result**: `results/stage1_results/layout_results.json`

### **Stage 2: Text Extraction** 📝
- **Input**: Image/document  
- **Output**: JSON with text + language
- **Uses**: Your existing OCR + language detection
- **Result**: `results/stage2_results/text_results.json`

### **Stage 3: Content Understanding** 🧠
- **Input**: Image/document
- **Output**: JSON with descriptions
- **Uses**: Advanced models (if available) or basic
- **Result**: `results/stage3_results/content_results.json`

## 🔧 **Setup (Simplified)**

### **Basic Setup (Stages 1 & 2)**
```bash
pip install -r requirements.txt
```

### **Advanced Setup (Stage 3)**
```bash
pip install transformers datasets fasttext
```

## 📁 **Clean Directory Structure**

```
multilingual-docai/
├── scripts/
│   ├── run_stages.py          # 🎯 MAIN SCRIPT - Use this!
│   └── cleanup.py             # Clean up repository
├── core/                      # Core functionality
├── backend/                   # Backend services  
├── configs/                   # Configuration files
├── data/
│   └── test/                  # Put your test images here
├── results/                   # Output results
├── models/                    # Model storage
├── README_QUICK_START.md      # 📖 Simple instructions
└── CLEAN_REPOSITORY_GUIDE.md # 📖 This file
```

## 🏃‍♂️ **Quick Start (3 Steps)**

### **Step 1: Clean up**
```bash
python scripts/cleanup.py
```

### **Step 2: Check status**
```bash
python scripts/run_stages.py --status
```

### **Step 3: Run a stage**
```bash
# Put your image in data/test/
python scripts/run_stages.py --stage 1 --input data/test/your_image.png
```

## 🆘 **Help Commands**

### **See all options**
```bash
python scripts/run_stages.py --help
```

### **Check system status**
```bash
python scripts/run_stages.py --status
```

### **Test advanced models**
```bash
python scripts/run_stages.py --test-advanced
```

## 🎉 **Success Indicators**

- **Stage 1**: ✅ "Stage 1 completed successfully!"
- **Stage 2**: ✅ "Stage 2 completed successfully!"  
- **Stage 3**: ✅ "Stage 3 completed successfully!"
- **All Stages**: 🎉 "ALL STAGES COMPLETED SUCCESSFULLY!"

## 💡 **Pro Tips**

1. **Start with Stage 1** to test basic functionality
2. **Use --status** to see what's available
3. **Use --help** to see all options
4. **Check logs** if something fails
5. **Put test images in `data/test/`**

## 🚀 **Next Steps**

1. **Run the cleanup script** to organize everything
2. **Check system status** to see what's working
3. **Test with Stage 1** to verify basic functionality
4. **Run complete pipeline** with `--stage all`
5. **Check results** in the `results/` directory

---

## 🎯 **Summary**

**You now have ONE simple script that does everything:**
- ✅ **`scripts/run_stages.py`** - Run any stage or all stages
- ✅ **`scripts/cleanup.py`** - Clean up repository
- ✅ **`README_QUICK_START.md`** - Simple instructions

**No more confusion! Just run `python scripts/run_stages.py --help` to see everything!** 🚀
