# 🧭 PS-05 Navigation Index

## 🎯 **Quick Navigation Guide**

This index helps you quickly find what you need in the PS-05 project.

## 🚀 **Main Entry Points**

| **What You Want** | **Main Command** | **Direct Script** | **Documentation** |
|-------------------|------------------|-------------------|-------------------|
| **Run Pipeline** | `python ps05.py pipeline` | `scripts/core/run_stages.py` | `README_QUICK_START.md` |
| **Clean Data** | `python ps05.py clean` | `scripts/cleaning/eda_with_cleaning.py` | `CLEANING_SERVICES_GUIDE.md` |
| **Train Models** | `python ps05.py train` | `scripts/training/` | `README_QUICK_START.md` |
| **Use API** | `python ps05.py backend` | `backend/app/main.py` | `API_USAGE_GUIDE.md` |
| **Utilities** | `python ps05.py utils` | `scripts/utilities/` | `CLEAN_REPOSITORY_GUIDE.md` |

## 📁 **Directory Quick Reference**

### **🏗️ CORE SYSTEM**
```
core/                    # ML models, evaluation, preprocessing
backend/                 # FastAPI server with all services
configs/                 # Configuration files
```

### **🔧 SCRIPTS (Organized by Function)**
```
scripts/
├── core/                # Main pipeline execution
├── training/            # Training and dataset preparation
├── cleaning/            # Data cleaning and EDA
└── utilities/           # Maintenance and utilities
```

### **📊 DATA & RESULTS**
```
data/                    # Your datasets
models/                  # Trained models
results/                 # Processing outputs
```

## 🔍 **Finding Specific Functionality**

### **1. Want to Run the Pipeline?**
- **Start here**: `python ps05.py pipeline --stage all`
- **Script location**: `scripts/core/run_stages.py`
- **Documentation**: `README_QUICK_START.md`

### **2. Want to Clean Your Data?**
- **Start here**: `python ps05.py clean --mode cleaning_with_eda`
- **Script location**: `scripts/cleaning/eda_with_cleaning.py`
- **Documentation**: `CLEANING_SERVICES_GUIDE.md`

### **3. Want to Train Models?**
- **Start here**: `python ps05.py train --prepare-data --train-model`
- **Script location**: `scripts/training/`
- **Documentation**: `README_QUICK_START.md`

### **4. Want to Use the API?**
- **Start here**: `python ps05.py backend --start`
- **Code location**: `backend/app/main.py`
- **Documentation**: `API_USAGE_GUIDE.md`

### **5. Want to Understand the Project?**
- **Start here**: `README_MASTER.md`
- **Structure**: `PROJECT_STRUCTURE.md`
- **Details**: `DIRECTORY_STRUCTURE.md`

## 📚 **Documentation Map**

### **📖 Getting Started**
- **`README_MASTER.md`** - Complete project overview
- **`README_QUICK_START.md`** - Quick start guide
- **`PROJECT_STRUCTURE.md`** - High-level organization

### **🔧 How-To Guides**
- **`CLEANING_SERVICES_GUIDE.md`** - Cleaning services usage
- **`EDA_CLEANING_INTEGRATION_GUIDE.md`** - EDA integration
- **`API_USAGE_GUIDE.md`** - API usage and examples

### **📁 Reference**
- **`DIRECTORY_STRUCTURE.md`** - Detailed file layout
- **`CLEAN_REPOSITORY_GUIDE.md`** - Repository management

## 🎯 **Common Use Cases**

### **Use Case 1: "I want to process my documents"**
```bash
# Quick start
python ps05.py pipeline --stage all --input data/my_docs --output results/processed

# What happens: 3-stage pipeline runs on your documents
# Results saved to: results/processed/
```

### **Use Case 2: "I want to clean my training data"**
```bash
# Quick start
python ps05.py clean --input data/raw --output data/cleaned --mode cleaning_with_eda

# What happens: EDA analysis + 21 cleaning tasks
# Results saved to: data/cleaned/
```

### **Use Case 3: "I want to train a custom model"**
```bash
# Quick start
python ps05.py train --prepare-data --input data/train --output data/prepared
python ps05.py train --train-model

# What happens: Data preparation + YOLO training
# Model saved to: models/
```

### **Use Case 4: "I want to deploy this as an API"**
```bash
# Quick start
python ps05.py backend --start --port 8000

# What happens: FastAPI server starts
# API available at: http://localhost:8000
```

## 🔧 **Configuration Files**

| **File** | **Purpose** | **Location** |
|----------|-------------|--------------|
| `ps05_config.yaml` | Main configuration | `configs/` |
| `competition_config.yaml` | Competition settings | `configs/` |
| `dataset_registry.yaml` | Dataset registry | `configs/` |

## 🚨 **Troubleshooting Quick Reference**

### **Problem: "Import Error"**
```bash
# Solution: Make sure you're in project root
cd /path/to/multilingual-docai
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### **Problem: "Script not found"**
```bash
# Solution: Use the main entry point
python ps05.py --help  # See all options
```

### **Problem: "Backend won't start"**
```bash
# Solution: Check dependencies and port
pip install -r requirements.txt
python ps05.py backend --start --port 8001  # Try different port
```

### **Problem: "Memory issues"**
```bash
# Solution: Use smaller batches or EDA only mode
python ps05.py clean --mode eda_only  # Less memory usage
```

## 💡 **Pro Tips**

1. **Always start with `python ps05.py --help`** - Shows all available options
2. **Use the main entry point** - `ps05.py` handles all the complexity
3. **Check documentation first** - Most questions are answered in the guides
4. **Start simple** - Use `pipeline` command first, then add complexity
5. **Organize your data** - Put datasets in `data/` directory

## 🎯 **Quick Commands Reference**

```bash
# See all options
python ps05.py --help

# Run pipeline
python ps05.py pipeline --stage all

# Clean data
python ps05.py clean --input data/raw --output data/cleaned

# Train models
python ps05.py train --prepare-data --train-model

# Start backend
python ps05.py backend --start

# Utilities
python ps05.py utils --cleanup
```

---

## 🎉 **You're All Set!**

- **🎯 Start with**: `python ps05.py --help`
- **📚 Read**: `README_MASTER.md` for complete overview
- **🔧 Use**: `ps05.py` as your main entry point
- **📁 Navigate**: Use this index to find what you need

**💡 The project is now organized and easy to understand!**
