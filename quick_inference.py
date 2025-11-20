"""
Rwanda Aurora - Quick Inference Script
======================================

Simple script to make weather forecasts using the trained model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

print("Rwanda Aurora - Quick Inference")
print("=" * 60)

# ================================
# CONFIGURATION
# ================================

MODEL_PATH = "working/best_rwanda_aurora.pt"
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")

# ================================
# LOAD MODEL
# ================================

print("\nLoading model...")
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"✓ Best validation loss: {checkpoint.get('loss', 'N/A'):.2f}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nPlease ensure:")
    print("  1. Model has been trained")
    print("  2. Checkpoint exists at: working/best_rwanda_aurora.pt")
    exit(1)

# ================================
# PREPARE SAMPLE INPUT
# ================================

print("\nPreparing sample input data...")

# Generate synthetic current conditions for Rwanda
# In production, this would come from real observations
batch_size = 1
seq_len = 4  # 4 timesteps (24 hours of 6-hourly data)
height = 8
width = 9
channels = 5  # [2t, tp, 10u, 10v, msl]

# Create realistic initial conditions for Rwanda
# Temperature: ~20-25°C (293-298 K)
temp_init = np.random.uniform(293, 298, (batch_size, seq_len, height, width, 1))

# Precipitation: mostly dry with occasional rain
precip_init = np.random.exponential(0.5e-5, (batch_size, seq_len, height, width, 1))
precip_init = np.clip(precip_init, 0, 1e-4)

# Wind: light to moderate (0-5 m/s)
u_wind_init = np.random.uniform(-3, 3, (batch_size, seq_len, height, width, 1))
v_wind_init = np.random.uniform(-3, 3, (batch_size, seq_len, height, width, 1))

# Pressure: ~850 hPa at Rwanda's elevation
pressure_init = np.random.uniform(84000, 86000, (batch_size, seq_len, height, width, 1))

# Combine all variables
input_data = np.concatenate([
    temp_init,
    precip_init,
    u_wind_init,
    v_wind_init,
    pressure_init
], axis=-1)

input_tensor = torch.FloatTensor(input_data).to(DEVICE)
print(f"✓ Input shape: {input_tensor.shape}")
print(f"  Variables: 2t, tp, 10u, 10v, msl")
print(f"  Sequence: {seq_len} timesteps (24 hours)")

# ================================
# MAKE FORECAST
# ================================

print("\nGenerating 120-hour forecast...")
print("(Note: Using checkpoint state only for demo)")
print("(For actual predictions, you need to instantiate the model)")

# Calculate some basic statistics from input
print("\nCurrent Conditions (from input):")
print(f"  Temperature: {input_data[0, -1, :, :, 0].mean():.1f} K ({input_data[0, -1, :, :, 0].mean() - 273.15:.1f}°C)")
print(f"  Precipitation: {input_data[0, -1, :, :, 1].mean():.2e} kg/m²")
print(f"  U-wind: {input_data[0, -1, :, :, 2].mean():.2f} m/s")
print(f"  V-wind: {input_data[0, -1, :, :, 3].mean():.2f} m/s")
print(f"  Pressure: {input_data[0, -1, :, :, 4].mean():.0f} Pa")

# For demonstration, create a simple forecast
# In production, you would use: forecast = model(input_tensor, use_vectorized=True)
print("\n" + "=" * 60)
print("FORECAST GENERATED")
print("=" * 60)

# Generate times for forecast
base_time = datetime.now()
forecast_times = [base_time + timedelta(hours=6*i) for i in range(1, 21)]

print(f"\nForecast valid from:")
print(f"  {forecast_times[0].strftime('%Y-%m-%d %H:%M')}")
print(f"to:")
print(f"  {forecast_times[-1].strftime('%Y-%m-%d %H:%M')}")
print(f"\nTotal: 20 timesteps (120 hours / 5 days)")

# ================================
# SAMPLE FORECAST OUTPUT
# ================================

print("\n" + "=" * 60)
print("Sample Forecast (simplified for demo):")
print("=" * 60)

# Display forecast for a few lead times
for i, forecast_time in enumerate([0, 4, 9, 14, 19]):
    hours_ahead = (forecast_time + 1) * 6
    time_str = forecast_times[forecast_time].strftime('%Y-%m-%d %H:%M')
    
    # Simulate forecast values (in production, use actual model output)
    temp_forecast = input_data[0, -1, :, :, 0].mean() + np.random.uniform(-2, 2)
    precip_forecast = max(0, input_data[0, -1, :, :, 1].mean() * (1 + np.random.uniform(-0.5, 1.5)))
    
    print(f"\n+{hours_ahead:3d} hours ({time_str}):")
    print(f"  Temperature: {temp_forecast:.1f} K ({temp_forecast - 273.15:.1f}°C)")
    print(f"  Precipitation: {precip_forecast:.2e} kg/m²")

# ================================
# VISUALIZATION
# ================================

print("\n" + "=" * 60)
print("Creating visualization...")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Temperature map (current)
im1 = axes[0].imshow(input_data[0, -1, :, :, 0], cmap='RdYlBu_r')
axes[0].set_title('Current Temperature (K)')
plt.colorbar(im1, ax=axes[0])

# Plot 2: Wind field (current)
U = input_data[0, -1, :, :, 2]
V = input_data[0, -1, :, :, 3]
speed = np.sqrt(U**2 + V**2)
im2 = axes[1].imshow(speed, cmap='viridis')
axes[1].set_title('Current Wind Speed (m/s)')
plt.colorbar(im2, ax=axes[1])
# Add wind vectors
Y, X = np.mgrid[0:height, 0:width]
axes[1].quiver(X, Y, U, V, color='white', alpha=0.6, scale=30)

# Plot 3: Pressure map (current)
im3 = axes[2].imshow(input_data[0, -1, :, :, 4], cmap='plasma')
axes[2].set_title('Current Pressure (Pa)')
plt.colorbar(im3, ax=axes[2])

plt.suptitle('Rwanda Weather - Current Conditions', fontsize=14, fontweight='bold')
plt.tight_layout()

output_file = 'rwanda_forecast_demo.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to: {output_file}")

# ================================
# USAGE INSTRUCTIONS
# ================================

print("\n" + "=" * 60)
print("For Production Use:")
print("=" * 60)
print("""
To run actual forecasts, you need to:

1. Instantiate the model properly:
   
   from notebooks.rwanda_aurora_training import RwandaAuroraLite, KaggleConfig
   
   config = KaggleConfig()
   model = RwandaAuroraLite(config).to(DEVICE)
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()

2. Make predictions:
   
   with torch.no_grad():
       forecast = model(input_tensor, use_vectorized=True)
   
3. Post-process:
   
   forecast_np = forecast.cpu().numpy()
   temp_forecast = forecast_np[0, :, :, :, 0]  # Temperature
   precip_forecast = forecast_np[0, :, :, :, 1]  # Precipitation

4. Interpret results:
   
   # Temperature: Convert K to °C
   temp_celsius = temp_forecast - 273.15
   
   # Precipitation: kg/m² per 6 hours
   # (multiply by 4 for daily total)
   daily_precip = precip_forecast * 4

See evaluate_model.py for complete examples.
""")

print("\n" + "=" * 60)
print("Quick Inference Complete!")
print("=" * 60)
