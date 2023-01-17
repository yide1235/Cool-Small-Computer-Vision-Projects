# 2D to 3D PoseEstimation
Replace the 2D PoseEstimation model with a more sophisticated model.
- Add processor_img_2d_open.py to convert input to 2D joints using the better OpenPose model
- Linearly interpolate 2d keypoints in OpenPose joints
- Convert interpolated 2D joints (18joints) to h3.6m 2D joints (17joints)
- Plot h3.6m 2d skeleton in every frame of input video
- Define classification algorithm to classify output

### Results
#### Turn Right Gesture
![right](./img/right_turn.gif)
#### Stop Gesture
![stop](./img/stop.gif)
#### Turn Left Gesture
![left](./img/left_turn.gif)

