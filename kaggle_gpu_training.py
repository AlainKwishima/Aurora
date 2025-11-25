"""
Rwanda Aurora - Kaggle GPU Training Script
==========================================

Upload this script to Kaggle and run with GPU enabled.

Setup:
1. Create new Kaggle notebook
2. Settings → Accelerator → GPU T4 x2
3. Copy this code into a cell
4. Run!

Expected Training Time: ~10-30 minutes on GPU
"""

# ===== STEP 1: Install Dependencies =====
print("📦 Installing dependencies...")
import subprocess
import sys

packages = [
    'wandb',
    'einops',
    'timm',
    'xarray',
    'netcdf4',
    'seaborn',
    'matplotlib',
    'scikit-learn'
]

for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print("✓ Dependencies installed\n")


# ===== STEP 2: Clone Repository =====
print("📥 Cloning Aurora repository...")
import os

if not os.path.exists('Aurora'):
    subprocess.check_call(['git', 'clone', 'https://github.com/AlainKwishima/Aurora.git'])
    os.chdir('Aurora')
else:
    os.chdir('Aurora')
    subprocess.check_call(['git', 'pull'])

print("✓ Repository ready\n")


# ===== STEP 3: Install Aurora Package =====
print("📦 Installing Aurora package...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'])
print("✓ Aurora installed\n")


# ===== STEP 4: Check GPU =====
print("🔍 Checking GPU availability...")
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("✓ GPU ready!\n")
else:
    print("⚠️  WARNING: No GPU detected!")
    print("Please enable GPU: Settings → Accelerator → GPU T4 x2\n")


# ===== STEP 5: Configure for GPU Training =====
print("⚙️  Configuring for GPU training...")

from notebooks.rwanda_aurora_training import KaggleConfig, RwandaAuroraLite, train_model, create_dataloaders

# Override config for GPU
class GPUKaggleConfig(KaggleConfig):
    """Optimized config for Kaggle GPU"""
    
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    @staticmethod
    def training_config():
        base = KaggleConfig.training_config()
        base.update({
            'batch_size': 8,  # Larger batch with GPU
            'num_epochs': 100,
            'learning_rate': 3e-5,
            'accumulation_steps': 2,  # Effective batch: 16
            'early_stopping_patience': 20,
        })
        return base

print("✓ Configuration ready\n")


# ===== STEP 6: Optional W&B Tracking =====
print("📊 Setting up Weights & Biases (optional)...")
try:
    import wandb
    
    # Uncomment and add your W&B API key to enable tracking
    # wandb.login(key='YOUR_API_KEY_HERE')
    # wandb.init(project="rwanda-aurora-kaggle", config=GPUKaggleConfig.training_config())
    # use_wandb = True
    
    use_wandb = False
    print("⚠️  W&B not configured (optional)")
except:
    use_wandb = False
    print("⚠️  W&B not available (optional)")

print()


# ===== STEP 7: Check for Data =====
print("📂 Checking for data...")

data_paths = [
    '/kaggle/input/rwanda-era5/rwanda_era5.nc',
    '/kaggle/input/rwanda-era5-train/rwanda_era5_train.nc',
    'data/rwanda_era5.nc',
]

data_file = None
for path in data_paths:
    if os.path.exists(path):
        data_file = path
        print(f"✓ Found data: {path}\n")
        break

if not data_file:
    print("⚠️  No ERA5 data found - will use synthetic data")
    print("For real training, upload ERA5 data as a Kaggle dataset\n")


# ===== STEP 8: Create Model =====
print("🧠 Creating model...")
config = GPUKaggleConfig()
model = RwandaAuroraLite(config)

device = config.get_device()
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✓ Model created")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Device: {device}\n")


# ===== STEP 9: Create Data Loaders =====
print("📊 Creating data loaders...")
train_loader, val_loader = create_dataloaders(config)
print(f"✓ Data loaders ready")
print(f"  Training samples: {len(train_loader.dataset)}")
print(f"  Validation samples: {len(val_loader.dataset)}\n")


# ===== STEP 10: Train Model =====
print("🚀 Starting training...")
print("=" * 60)

try:
    from notebooks.rwanda_aurora_training import main
    
    # Run training
    main()
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    if use_wandb:
        wandb.finish()


# ===== STEP 11: Download Results =====
print("\n📥 Downloading results...")
print("\nYour trained model is at: working/best_rwanda_aurora.pt")
print("Training history plot: working/training_history.png")
print("\nTo download:")
print("1. Click on 'Output' in the sidebar")
print("2. Download 'best_rwanda_aurora.pt'")
print("3. Download 'training_history.png'")

# Create download links (in Jupyter)
try:
    from IPython.display import FileLink, display
    print("\n📎 Download links:")
    display(FileLink('working/best_rwanda_aurora.pt'))
    display(FileLink('working/training_history.png'))
except:
    pass

print("\n🎉 All done! Your model is ready for deployment.")
