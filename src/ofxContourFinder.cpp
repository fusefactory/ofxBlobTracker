/*
 *  ofxContourFinder.cpp
 *
 * Finds white blobs in binary images and identifies
 * centroid, bounding box, area, length and polygonal contour
 * The result is placed in a vector of Blob objects.
 *
 * Created on 2/2/09.
 * Adapted from openframeworks ofxCvContourFinder
 *
 */

#include "ofxContourFinder.h"

//--------------------------------------------------------------------------------
ofxContourFinder::ofxContourFinder(){
	myMoments = (CvMoments*)malloc( sizeof(CvMoments) );
	reset();
}

//--------------------------------------------------------------------------------
ofxContourFinder::~ofxContourFinder(){
	free( myMoments );
}

//--------------------------------------------------------------------------------
void ofxContourFinder::reset() {
    blobs.clear();
    nBlobs = 0;
}

//--------------------------------------------------------------------------------
int ofxContourFinder::findContours(	ofxCvGrayscaleImage&  input,
									int minArea,
									int maxArea,
									int nConsidered,
									double hullPress,	
									bool bFindHoles,
									bool bUseApproximation) {
	// get width/height disregarding ROI
    IplImage* ipltemp = input.getCvImage();
    width = ipltemp->width;
    height = ipltemp->height;
	reset();

	// opencv will clober the image it detects contours on, so we want to
    // copy it into a copy before we detect contours.  That copy is allocated
    // if necessary (necessary = (a) not allocated or (b) wrong size)
	// so be careful if you pass in different sized images to "findContours"
	// there is a performance penalty, but we think there is not a memory leak
    // to worry about better to create mutiple contour finders for different
    // sizes, ie, if you are finding contours in a 640x480 image but also a
    // 320x240 image better to make two ofxContourFinder objects then to use
    // one, because you will get penalized less.

	if( inputCopy.width == 0 ) {
		inputCopy.allocate( input.width, input.height );
		inputCopy = input;
	} else {
		if( inputCopy.width == input.width && inputCopy.height == input.height ) 
			inputCopy = input;
		else {
			// we are allocated, but to the wrong size --
			// been checked for memory leaks, but a warning:
			// be careful if you call this function with alot of different
			// sized "input" images!, it does allocation every time
			// a new size is passed in....
			inputCopy.clear();
			inputCopy.allocate( input.width, input.height );
			inputCopy = input;
		}
	}

	CvSeq* contour_list = NULL;
	contour_storage = cvCreateMemStorage( 1000 );
	storage	= cvCreateMemStorage( 1000 );

	cvFindContours( inputCopy.getCvImage(), contour_storage, &contour_list,
                   sizeof(CvContour), CV_RETR_EXTERNAL, bUseApproximation ? CV_CHAIN_APPROX_SIMPLE : CV_CHAIN_APPROX_NONE );
	
	CvSeq* contour_ptr = contour_list;

	nCvSeqsFound = 0;

	// put the contours from the linked list, into an array for sorting
	while( (contour_ptr != NULL) )  {
		CvBox2D box=cvMinAreaRect2(contour_ptr);
		
        float area = fabs( cvContourArea(contour_ptr, CV_WHOLE_SEQ) );
        if( (area > minArea) && (area < maxArea) ) {
            ofxBlob blob = ofxBlob();
            float area = cvContourArea( contour_ptr, CV_WHOLE_SEQ);
            cvMoments( contour_ptr, myMoments );
            
            // this is if using non-angle bounding box
            CvRect rect	= cvBoundingRect( contour_ptr, 0 );
            blob.boundingRect.x      = rect.x/width;
            blob.boundingRect.y      = rect.y/height;
            blob.boundingRect.width  = rect.width/width;
            blob.boundingRect.height = rect.height/height;
            
            //Angle Bounding rectangle
            blob.angleBoundingRect.x	  = box.center.x/width;
            blob.angleBoundingRect.y	  = box.center.y/height;
            blob.angleBoundingRect.width  = box.size.height/width;
            blob.angleBoundingRect.height = box.size.width/height;
            blob.angle = box.angle;
            
            // assign other parameters
            blob.area                = fabs(area);
            blob.hole                = area < 0 ? true : false;
            blob.length 			 = cvArcLength(contour_ptr);
            
            // The cast to int causes errors in tracking since centroids are calculated in
            // floats and they migh land between integer pixel values (which is what we really want)
            // This not only makes tracking more accurate but also more fluid
            blob.centroid.x			 = (myMoments->m10 / myMoments->m00) / width;
            blob.centroid.y 		 = (myMoments->m01 / myMoments->m00) / height;
            blob.lastCentroid.x 	 = 0;
            blob.lastCentroid.y 	 = 0;
            
            if (blob.nFingers != 0){
                
                blob.nFingers = 0;
                blob.fingers.clear();
            }
            
            // get the points for the blob:
            CvPoint           pt;
            CvSeqReader       reader;
            cvStartReadSeq( contour_ptr, &reader, 0 );
            
            for( int j=0; j < min(TOUCH_MAX_CONTOUR_LENGTH, contour_ptr->total); j++ ) {
                CV_READ_SEQ_ELEM( pt, reader );
                blob.pts.push_back( ofPoint((float)pt.x / width, (float)pt.y / height) );
            }
            blob.nPts = blob.pts.size();
            
            // Check if it´s a Hand and if it have fingers
            //

            
            blobs.push_back(blob);
        }
        contour_ptr = contour_ptr->h_next;
    }
    
	nBlobs = blobs.size();
	
	// Free the storage memory.
	// Warning: do this inside this function otherwise a strange memory leak
	if( contour_storage != NULL )
		cvReleaseMemStorage(&contour_storage);
	
	if( storage != NULL )
		cvReleaseMemStorage(&storage);
    
    free(contour_ptr);

	return nBlobs;
}
