import cv2
import sys

class CarTracker:

    def __init__(self):
        self.tracker = None
        self.trackerInitialized = False
        self.bbox_credit = 0.0 # initialized to confidence score of box. if below certain threshold, we treat this tracker instance as obsolete.

    def initializeTracker(self, frame, carbbox, trackerType='MEDIANFLOW', score=1.0):
        if trackerType == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if trackerType == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if trackerType == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if trackerType == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if trackerType == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if trackerType == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        if trackerType == 'MOSSE':
            self.tracker = cv2.TrackerMOSSE_create()
        if trackerType == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()

        self.tracker.init(frame, carbbox)
        self.trackerInitialized = True
        self.bbox_credit = score

    def updateTracker(self, frame):
        ok, output = self.tracker.update(frame);
        if (not ok):
            # Tracking failure. 
            # cv2.putText(frame, "Bbox lost. Tracker is no longer valid.", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            self.tracker.clear();
            self.trackerInitialized = False
            self.bbox_credit = 0.0
            output = []
        else:
            self.bbox_credit -= 0.025
            if self.bbox_credit <= 0.0:
                # Tracking obsolete.
                # print ("Bbox outdated. Tracker is obsolete.")
                self.tracker.clear();
                self.trackerInitialized = False
                self.bbox_credit = 0.0
                ok = False
                output = []
        return ok, output

