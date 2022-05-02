import cv2
import sys

class CarBlob:

    def __init__(self):
        self.boundRect = None
        self.trackerInitialized = False
        self.initial_car_bbox = None
        self.current_car_bbox = None

    def initializeTracker(self, frame, roi_rec):
        self.initial_car_bbox = roi_rec
        self.current_car_bbox = self.initial_car_bbox;
        trackerTypes = ["BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT" ]
        trackerType = trackerTypes[2];

        if tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            self.tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()

        self.tracker.init(frame, initial_taillight_bbox);
        self.trackerInitialized = true;

    def updateTracker(self, frame):
        ok, current_taillight_bbox = self.tracker.update(frame);
        if (not ok):
            # Tracking failure. Tracker is no longer valid.
            # cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            self.tracker.clear();
            self.trackerInitialized = false;
            return (0, 0, 0, 0);
        
        return current_taillight_bbox


'''
class TaillightBlob
{
	Point2d center; // center of contour.
	double area;
	Rect boundRect;
	int lightOn;
	unsigned char taillight_flag;
'''

