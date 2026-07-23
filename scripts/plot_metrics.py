
import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_metrics(history_path):
    if not os.path.exists(history_path):
        print(f"Error: {history_path} does not exist.")
        return

    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)

    rounds = history.get('rounds', [])
    avg_losses = history.get('avg_losses', [])
    val_metrics = history.get('val_metrics', [])

    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(rounds, avg_losses, label='Training Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Training Loss per Round')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    print("Saved training_loss.png")
    plt.close()

    # Prepare data for Validation Metrics
    val_rounds = [m['round'] for m in val_metrics]
    val_dice = [m['dice'] for m in val_metrics]
    val_hd95 = [m['hd95'] for m in val_metrics]
    
    # Handle infinite HD95 values for plotting
    val_hd95_clean = [h if h != float('inf') else np.nan for h in val_hd95]

    # Plot Validation Dice
    plt.figure(figsize=(10, 5))
    plt.plot(val_rounds, val_dice, label='Validation Dice', color='green', marker='o')
    plt.xlabel('Round')
    plt.ylabel('Dice Coefficient')
    plt.title('Validation Dice per Round')
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_dice.png')
    print("Saved validation_dice.png")
    plt.close()

    # Plot Validation HD95
    plt.figure(figsize=(10, 5))
    plt.plot(val_rounds, val_hd95_clean, label='Validation HD95', color='red', marker='x')
    plt.xlabel('Round')
    plt.ylabel('HD95')
    plt.title('Validation HD95 per Round')
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_hd95.png')
    print("Saved validation_hd95.png")
    plt.close()

if __name__ == "__main__":
    history_path = os.path.join("data", "checkpoints", "training_history.json")
    plot_metrics(history_path)
