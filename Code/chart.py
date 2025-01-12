import matplotlib.pyplot as plt
import numpy as np

# Updated Data
test_numbers = np.arange(1, 11)
real_quantities = [20] * 10
predicted_quantities = [17, 20, 17, 17, 15, 20, 18, 19, 21, 18]
processing_times = [11.18, 10.02, 10.21, 10.55, 10.62, 11.71, 10.34, 9.8, 10.28, 10.26]

# Create figure with subplots - using a built-in style
plt.style.use('ggplot')  # Changed from 'seaborn' to 'ggplot'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Real vs Predicted Quantities
ax1.plot(test_numbers, real_quantities, 'b-', label='Real Quantity', linewidth=2)
ax1.plot(test_numbers, predicted_quantities, 'r--o', label='Predicted Quantity', linewidth=2)
ax1.fill_between(test_numbers, real_quantities, predicted_quantities, alpha=0.2, color='gray')
ax1.set_xlabel('Test Number')
ax1.set_ylabel('Quantity')
ax1.set_title('Real vs Predicted Quantities')
ax1.grid(True)
ax1.legend()

# Add error values
for i, (real, pred) in enumerate(zip(real_quantities, predicted_quantities)):
    error = abs(real - pred)
    ax1.text(i+1, max(real, pred) + 0.5, f'Error: {error}', ha='center')

# Plot 2: Processing Times
ax2.bar(test_numbers, processing_times, color='skyblue')
ax2.axhline(y=np.mean(processing_times), color='r', linestyle='--', label=f'Mean Time: {np.mean(processing_times):.2f}s')
ax2.set_xlabel('Test Number')
ax2.set_ylabel('Processing Time (seconds)')
ax2.set_title('Processing Times for Each Test')
ax2.grid(True, axis='y')
ax2.legend()

# Add time values on top of bars
for i, time in enumerate(processing_times):
    ax2.text(i+1, time + 0.2, f'{time}s', ha='center')

# Add overall accuracy
plt.figtext(0.02, 0.02, f'Overall Accuracy: 90%', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# Adjust layout and display
plt.tight_layout()
plt.show()

# Calculate and print statistics
print(f"Average Processing Time: {np.mean(processing_times):.2f}s")
print(f"Standard Deviation of Processing Time: {np.std(processing_times):.2f}s")
print(f"Min Processing Time: {min(processing_times):.2f}s")
print(f"Max Processing Time: {max(processing_times):.2f}s")
print(f"Average Absolute Error in Counting: {np.mean(np.abs(np.array(real_quantities) - np.array(predicted_quantities))):.2f}")
