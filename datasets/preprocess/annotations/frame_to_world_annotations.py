# This block of code takes in frame based annotations and converts them to world scene graph annotations.
# We follow the following procedure:
# First estimate all the unique objects in the video based on some tracking id.
# If an object does not appear in a frame, we add the bounding box corresponding to the last seen frame or the next seen frame.
# This ensures that each object has a bounding box in every frame of the video.
# Finally, we save the world scene graph annotations in a pkl file.
