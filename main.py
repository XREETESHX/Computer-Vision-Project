import cv2
import pickle
import numpy as np
from collections import deque
from datetime import datetime
import csv


class ImprovedParkingDetector:
    """Parking detector with temporal filtering to reduce flickering"""
    
    def __init__(self, video_path='carPark.mp4', spaces_file='polygons'):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            with open(spaces_file, 'rb') as f:
                data = pickle.load(f)
            
            self.polygons = []
            for item in data:
                if isinstance(item, (list, tuple)):
                    self.polygons.append(np.array(item, dtype=np.int32))
                else:
                    self.polygons.append(item)
            
            print(f"Loaded {len(self.polygons)} parking spaces")
        except FileNotFoundError:
            print(f"Parking spaces file not found: {spaces_file}")
            print("Run PolygonSpacePicker.py first to mark parking spaces")
            raise
        
        self.history_size = 5
        self.space_history = [deque(maxlen=self.history_size) for _ in self.polygons]
        
        self.gaussian_kernel = (3, 3)
        self.adaptive_block = 25
        self.adaptive_c = 16
        self.median_kernel = 5
        
        self.threshold_multiplier = 0.18
        self.min_threshold = 300
        
        self.results = []
        
        print("Detector ready")
    
    def preprocess_frame(self, frame):
        """Convert frame to binary image for detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.gaussian_kernel, 1)
        
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block,
            self.adaptive_c
        )
        
        median = cv2.medianBlur(thresh, self.median_kernel)
        
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(median, kernel, iterations=1)
        
        return dilated
    
    def check_space(self, processed_frame, polygon):
        """Check if parking space is occupied by counting white pixels"""
        h, w = processed_frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
        cropped = cv2.bitwise_and(processed_frame, processed_frame, mask=mask)
        count = cv2.countNonZero(cropped)
        area = cv2.contourArea(pts)
        threshold = max(self.min_threshold, int(area * self.threshold_multiplier))
        is_occupied = count >= threshold
        confidence = min(1.0, count / threshold) if threshold > 0 else 0.0
        
        return is_occupied, confidence, count
    
    def apply_temporal_filter(self, space_index, is_occupied):
        """Use frame history to smooth out detection flickering"""
        self.space_history[space_index].append(is_occupied)
        if len(self.space_history[space_index]) < 3:
            return is_occupied
        occupied_count = sum(self.space_history[space_index])
        return occupied_count > len(self.space_history[space_index]) / 2
    
    def visualize_frame(self, frame, space_states):
        """Draw parking spaces and stats on the video frame"""
        output = frame.copy()
        free_count = 0
        
        for i, (polygon, is_occupied, confidence, count) in enumerate(space_states):
            pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
            if is_occupied:
                color = (0, 0, 255)
                thickness = 2
            else:
                color = (0, 255, 0)
                thickness = 3
                free_count += 1
            cv2.polylines(output, [pts], isClosed=True, color=color, thickness=thickness)
            cx = int(np.mean(polygon[:, 0]))
            cy = int(np.mean(polygon[:, 1]))
            cv2.putText(output, str(i + 1), (cx - 15, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, f"{confidence:.2f}", (cx - 20, cy + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        total = len(space_states)
        occupancy_rate = (total - free_count) / total * 100 if total > 0 else 0
        cv2.rectangle(output, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(output, (10, 10), (400, 120), (255, 255, 255), 2)
        cv2.putText(output, f"Free: {free_count}/{total}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(output, f"Occupied: {total - free_count}/{total}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(output, f"Occupancy: {occupancy_rate:.1f}%", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return output, free_count, total
    
    def save_results(self, filename='detection_results.csv'):
        """Save results to CSV file"""
        if not self.results:
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'free', 'occupied', 'total', 'occupancy_rate'])
            writer.writerows(self.results)
        
        print(f"Results saved to {filename}")
    
    def run(self, show_processed=False, save_results=True, loop=True):
        """Main loop - process video frames"""
        print("\n" + "="*60)
        print("  Parking Space Detector Running")
        print("="*60)
        print("\nControls:")
        print("  ESC or Q - Quit")
        print("  P - Pause/Resume")
        print("  T - Toggle processed view")
        print("  S - Save screenshot")
        print()
        
        frame_count = 0
        paused = False
        show_thresh = show_processed
        
        try:
            while True:
                if not paused:
                    if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        if loop:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        else:
                            break
                    
                    success, frame = self.cap.read()
                    if not success:
                        break
                    processed = self.preprocess_frame(frame)
                    space_states = []
                    for i, polygon in enumerate(self.polygons):
                        is_occupied, confidence, count = self.check_space(processed, polygon)
                        is_occupied = self.apply_temporal_filter(i, is_occupied)
                        space_states.append((polygon, is_occupied, confidence, count))
                    annotated, free_count, total = self.visualize_frame(frame, space_states)
                    if save_results and frame_count % 30 == 0:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        occupied = total - free_count
                        occupancy_rate = occupied / total * 100 if total > 0 else 0
                        self.results.append([timestamp, free_count, occupied, total, occupancy_rate])
                    frame_count += 1
                    cv2.imshow("Parking Detection", annotated)
                    if show_thresh:
                        cv2.imshow("Processed", processed)
                key = cv2.waitKey(10) & 0xFF
                if key == 27 or key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('t'):
                    show_thresh = not show_thresh
                    if not show_thresh:
                        cv2.destroyWindow("Processed")
                elif key == ord('s'):
                    filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(filename, annotated)
                    print(f"Saved {filename}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            
            if save_results:
                self.save_results()
            
            print(f"\nProcessed {frame_count} frames")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parking Space Detector")
    parser.add_argument("--video", default="carPark.mp4", help="Video file path")
    parser.add_argument("--spaces", default="polygons", help="Parking spaces file")
    parser.add_argument("--show-processed", action="store_true", help="Show processed frames")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--no-loop", action="store_true", help="Don't loop video")
    
    args = parser.parse_args()
    
    try:
        detector = ImprovedParkingDetector(args.video, args.spaces)
        detector.run(
            show_processed=args.show_processed,
            save_results=not args.no_save,
            loop=not args.no_loop
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
