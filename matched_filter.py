import numpy as np
import matplotlib.pyplot as plt

def matched_filter(signal, template):
    """
    Apply a causal matched filter to the signal using the provided template.

    Args:
    - signal (np.ndarray): The input signal to filter.
    - template (np.ndarray): The template pulse shape.

    Returns:
    - filtered_signal (np.ndarray): The output of the matched filter.
    """
    # Reverse the template to create the matched filter
    matched_filter = template[::-1]
    
    # Perform the convolution, only keeping the valid part (causal filter)
    filtered_signal = np.convolve(signal, matched_filter, mode='valid')
    
    return filtered_signal

# Example usage
if __name__ == "__main__":
    # Define a template pulse (e.g., a Gaussian pulse)
    pulse_length = 100
    t = np.linspace(-1, 1, pulse_length)
    template_pulse = np.exp(-t**2 / (2 * 0.1**2))
    
    # Normalize the template pulse
    template_pulse /= np.linalg.norm(template_pulse)
    
    # Create a test signal with a single photon pulse and some noise
    signal_length = 1000
    signal = np.random.normal(0, 0.1, signal_length)
    pulse_position = 500
    signal[pulse_position:pulse_position + pulse_length] += template_pulse
    
    # Apply the matched filter
    filtered_signal = matched_filter(signal, template_pulse)
    
    # Plot the original signal and the filtered signal
    plt.figure(figsize=(14, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title("Original Signal with Single Photon Pulse")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    
    plt.subplot(2, 1, 2)
    plt.plot(filtered_signal)
    plt.title("Filtered Signal using Matched Filter")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.show()