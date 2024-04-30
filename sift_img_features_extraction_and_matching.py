""" HW9 - Image feature extraction and matching for object detection applications 
    SIFT: Scale-Invariant Feature transform
    This script performs object detection in user-specified images by extracting 
    and matching features using the SIFT algorithm, directly from the Linux/Windows 
    command line. It visualizes keypoints and matches between an object image and a scene image.

    Authors: Jorge Rodrigo Gomez Mayo & Juan Carlos Chávez Villareal
    Contact: jorger.gomez@udem.edu & juan.chavezv@udem.edu
    Organization: Universidad de Monterrey
    First created on Friday 26 April 2024

    Usage Example:
        py .\sift_img_features_extraction_and_matching.py -obj .\imgs\game_box.jpg -img .\imgs\scene_1.jpg
"""
# Import std libraries
import argparse
import cv2
import numpy as np


def parse_user_data() -> tuple[str, str]:
    """
    Parse the command-line arguments provided by the user.

    Returns:
        tuple[str, str]: A tuple containing the path to the object image and the input image.
    """
    parser = argparse.ArgumentParser(prog='HW9 - Team 3 - SIFT',
                                    description='Extract image features with SIFT for object detection', 
                                    epilog='JRGM & JCCV - 2024')
    parser.add_argument('-obj',
                        '--object_image',
                        type=str,
                        required=True,
                        help="Path to the image containing the object")
    parser.add_argument('-img',
                        '--scene_image',
                        type=str,
                        required=True,
                        help="Path to the image where we want to detect the object")
    
    args = parser.parse_args()
    return args

def extract_features(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract SIFT features from the image.

    Args:
        img (np.ndarray): Image data in which to find keypoints.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the keypoints and descriptors for the image.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(descriptors_1: np.ndarray, descriptors_2: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Match SIFT features using the FLANN based matcher.

    Args:
        descriptors_1 (np.ndarray): Descriptors from the first image.
        descriptors_2 (np.ndarray): Descriptors from the second image.

    Returns:
        tuple[np.ndarray, list]: A tuple containing the matches and the mask for good matches.
    """
    # Define FLANN-based matcher parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Use knnMatch to find matches
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply Lowe's ratio test
    matches_mask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.57 * n.distance:
            matches_mask[i] = [1, 0]
    return matches, matches_mask

def draw_keypoints(img: np.ndarray) -> np.ndarray:
    """
    Draw keypoints on the image.

    Args:
        img (np.ndarray): Image on which to draw keypoints.

    Returns:
        np.ndarray: Image with keypoints drawn.
    """
    gray = cv2.cvtColor(img["image"], cv2.COLOR_BGR2GRAY)
    img_with_kp = cv2.drawKeypoints(gray,img["features"]["kp"], None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_with_kp

def draw_matches(img_1: np.ndarray, img_2: np.ndarray, matches: np.ndarray, mask: list) -> np.ndarray:
    """
    Draw matches between two images.

    Args:
        img_1 (np.ndarray): First image.
        img_2 (np.ndarray): Second image.
        matches (np.ndarray): Matched features.
        mask (list): Mask for good matches.

    Returns:
        np.ndarray: Image with matches drawn.
    """
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=mask, flags=cv2.DrawMatchesFlags_DEFAULT)
    img = cv2.drawMatchesKnn(img_1["image"], img_1["features"]["kp"], img_2["image"], img_2["features"]["kp"], matches, None, **draw_params)
    return img

def find_matches_location(matches: list, kp2: list, mask: list) -> np.ndarray:
    """
    Find the minimum rotated rectangle around the matched keypoints in the second image.

    Args:
        matches (list): Matched keypoints indices.
        kp2 (list): Keypoints from the second image.
        mask (list): Mask for good matches.
    Returns:
        cv2.RotatedRect: A rotated rectangle that encloses the matched keypoints.
    """
    # Filter matches with the mask
    good_points = [kp2[matches[i][0].trainIdx].pt for i in range(len(matches)) if mask[i][0]]

    # Ensure there are enough points to calculate a rectangle
    if len(good_points) < 20:
        return None

    # Convert points to a numpy array
    points = np.array(good_points, dtype=np.float32)

    # Calculate the rotated rectangle
    rect = cv2.minAreaRect(points)

    return rect

def draw_rotated_rectangle(img: np.ndarray, rect: np.ndarray) -> np.ndarray:
    """
    Draw a rotated rectangle on the image.

    Args:
        img (np.ndarray): Image on which to draw.
        rect (cv2.RotatedRect): The rotated rectangle.

    Returns:
        np.ndarray: Image with the rectangle drawn.
    """
    # Get box points and draw
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    color = (255, 255, 0) # Cyan in BGR
    thickness = 5
    return cv2.drawContours(img, [box], 0, color, thickness)

def resize_image(img: np.ndarray) -> np.ndarray:
    """
    Resize the image to a specified scale for better visualization.

    Args:
        img (np.ndarray): Image to resize.

    Returns:
        np.ndarray: Resized image.
    """
    width = int(img.shape[1] * 0.35)
    height = int(img.shape[0] * 0.35)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img

def visualise_image(img: np.ndarray, title: str) -> None:
    """
    Display the image in a window with a title.

    Args:
        img (np.ndarray): Image to display.
        title (str): Title of the window.

    Returns:
        None
    """
    resized = resize_image(img)
    cv2.imshow(title, resized)
    cv2.waitKey(0)

def close_windows():
    """
    Close & destroy OpenCV windows

    The Function closes and destroy all cv2 windows that are open.
    """
    cv2.destroyAllWindows()

def run_pipeline():
    """
    Execute the entire SIFT feature extraction and matching pipeline.

    Returns:
        None
    """
    # Create dictionaries to contain image data
    print(f"\nStarting program...")
    obj  =  {"image": "", 
            "features": {"kp": "", "descriptors": ""}}
    img_1 = {"image": "", 
            "features": {"kp": "", "descriptors": ""}}

    # Parse user's input data
    user_input = parse_user_data()
    
    # Load images
    print(f"\nLoading images...")
    obj["image"] = cv2.imread(user_input.object_image)
    img_1["image"] = cv2.imread(user_input.scene_image)

    # Extract img features
    print(f"\nExtracting features...")
    obj["features"]["kp"], obj["features"]["descriptors"] = extract_features(obj["image"])
    img_1["features"]["kp"], img_1["features"]["descriptors"] = extract_features(img_1["image"])

    # Draw Keypoints
    print(f"\nDrawing keypoints...")
    obj_img_with_kp = draw_keypoints(obj)
    img_1_with_kp = draw_keypoints(img_1)
    
    # Display images with keypoints
    visualise_image(obj_img_with_kp, "Object's Keypoints")
    visualise_image(img_1_with_kp, "Scene's Keypoints")

    # Match features
    print(f"\nMatching features...")
    matches, matches_mask = match_features(obj["features"]["descriptors"], img_1["features"]["descriptors"])

    # Draw matches
    print(f"\nDrawing matches...")
    img_with_matches = draw_matches(obj, img_1, matches, matches_mask)

    # Display image with matches
    visualise_image(img_with_matches, "Matches Found")

    # Find the rotated rectangle location based on matches
    print(f"\nFinding object location in scene image...")
    rotated_rect = find_matches_location(matches, img_1["features"]["kp"], matches_mask)

    if rotated_rect:
        # Draw rotated rectangle
        print(f"\nDrawing detection area...")
        img_with_rotated_rect = draw_rotated_rectangle(img_1["image"].copy(), rotated_rect)
        visualise_image(img_with_rotated_rect, "Object detected within this region")
    else:
        print(f"\nNot enough matches found to stablish a detection region!")

    # Close Windows
    print(f"\nShutting down...\n")
    close_windows()

if __name__ == "__main__":
    run_pipeline()

"""
    References:
    [1]“OpenCV: Feature Matching.” Accessed: Apr. 25, 2024. [Online].
    Available: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

    [2]“SIFT Algorithm: How to Use SIFT for Image Matching in Python.” Accessed: Apr. 25, 2024. [Online].
    Available: https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/

    [3]“OpenCV: Introduction to SIFT (Scale-Invariant Feature Transform).” Accessed: Apr. 25, 2024. [Online]. 
    Available: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
"""