import serial
import serial.tools.list_ports
import numpy as np
import scipy.signal as signal
import time
import matplotlib.pyplot as plt
import collections
from matplotlib.animation import FuncAnimation

class SerialAnalyzer:

    def __init__(self, port = None, baudrate=2000000, window_size = 1024, num_elements = 12):

        if port is None:
            ports = serial.tools.list_ports.comports()
            if len(ports) == 0:
                raise Exception("No serial ports found. Please connect a device and try again.")
            print("Available serial ports:")
            for i, port in enumerate(ports):
                print(f"{i}: {port.device}")
            port_index = int(input("Enter the index of the serial port to use: "))
            port = ports[port_index].device
        else:
            try:
                self.ser = serial.Serial(port, baudrate)
                self.ser.setDTR(False) # Toggle it off
                time.sleep(0.1)
                self.ser.setDTR(True)  # Toggle it on
                self.ser.setRTS(True)  # Some systems require Request To Send

                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
                print("Buffers cleared. Starting data acquisition...")

                self.window_size = window_size
                self.data_buffer = [collections.deque([0.0] * window_size, maxlen=window_size) for _ in range(num_elements)]
                self.num_elements = num_elements

                self.PACKET_SIZE = 24  # 12 mics * 2 bytes
                self.HEADER = b'\x34\x12\xcd\xab' # 0xABCD1234 in Little Endian

            except serial.SerialException as e:
                print(f"Error opening serial port {port}: {e}")
                raise e
            except KeyboardInterrupt:
                print("Serial port selection cancelled by user.")
                raise KeyboardInterrupt
            
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.dtr = False
            self.ser.close()
            print("\nSerial port closed safely.")
            
    def read_mic_packet(self):
        # search for the header in two parts to avoid missing it if it spans across reads
        while True:
            if self.ser.read(1) == b'\x34': 
                if self.ser.read(3) == b'\x12\xcd\xab': 
                    break 
        
        # 2. Read the actual data (24 bytes)
        raw_payload = self.ser.read(self.PACKET_SIZE)
        
        # 3. Convert to integers instantly
        return np.frombuffer(raw_payload, dtype='<i2')
            
    def update_buffer(self):
        """Reads ALL currently available lines to prevent lag."""

        while self.ser.in_waiting > 28:
            try:
                mic_data = self.read_mic_packet()  # This will block until a full packet is received

                if mic_data is not None and len(mic_data) == self.num_elements:
                    for i, val in enumerate(mic_data):
                        norm_val = (float(val) - 2047.5) / 2047.5
                        self.data_buffer[i].append(norm_val)
                else:
                    print(f"Warning: Expected {self.num_elements} values but got {len(mic_data)}. Line: {mic_data}")
                    continue

            except (ValueError, UnicodeDecodeError) as e:
                print(f"Error parsing line: {e}. Line content: {mic_data}")
                continue # Skip malformed lines

    def animate_callback(self, frame, lines):
        """This runs every 'interval' ms"""
        self.update_buffer() # Get newest data
        for i in range(self.num_elements):
            lines[i].set_ydata(self.data_buffer[i])

        return lines

if __name__ == "__main__":
    target_port = '/dev/cu.usbmodem196242501'

    try:
        count = 0
        with SerialAnalyzer(port=target_port, window_size=4096) as analyzer:
            
            fig, ax = plt.subplots(4, 3, figsize=(12, 8))
            lines = []
            for i in range(4):
                for j in range(3):
                    ax[i, j].set_ylim(-1.1, 1.1)
                    ax[i, j].set_title(f"Teensy 4.1 Real-time Stream {i*3+j+1}")
                    line, = ax[i, j].plot(np.arange(analyzer.window_size), list(analyzer.data_buffer[i*3+j]))
                    lines.append(line)

            ani = FuncAnimation(
                fig, 
                analyzer.animate_callback, 
                interval=30, # Faster update for real-time
                fargs=(lines,),
                blit=True, 
                cache_frame_data=False
            )

            plt.show()

    except KeyboardInterrupt:
        print("\nStopping acquisition...")
    except Exception as e:
        print(f"\nApplication error: {e}")