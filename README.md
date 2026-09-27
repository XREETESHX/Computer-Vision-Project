# Parking Space Detection

A parking space occupancy detector using OpenCV. It detects whether parking spaces are free or occupied by analyzing video frames.

## What It Does

The system processes video frames to detect cars in predefined parking spaces. It uses image processing techniques like thresholding and pixel counting - no machine learning required.

## Setup

First, install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Mark Parking Spaces

Run the polygon picker to define your parking spaces:

```bash
python PolygonSpacePicker.py
```

- Left-click 4 corners for each parking space
- Right-click to delete a space
- Press 'q' to quit and save

The parking space coordinates are saved to the `polygons` file.

### Step 2: Run the Detector

```bash
python main.py
```

The detector will:

- Load the video (`carPark.mp4` by default)
- Process each frame
- Show green boxes for free spaces, red for occupied
- Save results to `detection_results.csv`

## Controls

While the detector is running:

- **ESC** or **Q** - Quit
- **P** - Pause/Resume
- **T** - Toggle processed view (shows the thresholded image)
- **S** - Save screenshot

## Command Line Options

```bash
# Use a different video file
python main.py --video myVideo.mp4

# Show the processed frames (debug mode)
python main.py --show-processed

# Don't loop the video
python main.py --no-loop

# Don't save CSV results
python main.py --no-save
```

## Files

- `main.py` - Main detector script
- `PolygonSpacePicker.py` - Tool to mark parking spaces
- `polygons` - Saved parking space coordinates (created after using PolygonSpacePicker)
- `carPark.mp4` - Video file to process
- `carParkImg.png` - Reference image for marking spaces

## How It Works

1. **Preprocessing**: Converts video frames to grayscale, applies blur, then adaptive thresholding to create a binary image
2. **Detection**: For each parking space polygon, counts white pixels (which represent cars)
3. **Filtering**: Uses a 5-frame history to smooth out flickering
4. **Display**: Shows green boxes for free spaces, red for occupied ones

## Output

Results are saved to `detection_results.csv` with columns:

- timestamp
- free (number of free spaces)
- occupied (number of occupied spaces)
- total (total spaces)
- occupancy_rate (percentage)
