import cv2
import numpy as np
import tensorflow as tf

# Load the MoveNet TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define keypoints for bicep curls and squats
bicep_curl_keypoints = [5, 7, 9, 11]  # Keypoints for hands, shoulders, and hips
squat_keypoints = [11, 12, 24]  # Keypoints for hips and knees

# Define exercise states
EXERCISE_IDLE = 0
EXERCISE_BICEP_CURL = 1
EXERCISE_SQUAT = 2

# Initialize variables for exercise tracking
current_exercise = EXERCISE_IDLE
repetition_count = 0
correct_repetitions = 0
correct_form = False
exercise_frames = 0

# Initialize a video capture object (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for the model
    input_image = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = (input_image / 255.0).astype(np.float32)

    # Run pose estimation
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])

    # Example: Check for bicep curls and squats
    def is_bicep_curl(keypoints):
        return all(np.all(keypoints[keypoint] > 0) for keypoint in bicep_curl_keypoints)

    def is_squat(keypoints):
        return all(np.all(keypoints[keypoint] > 0) for keypoint in squat_keypoints)

    # Update exercise state
    if current_exercise == EXERCISE_IDLE:
        if is_bicep_curl(keypoints[0]):
            current_exercise = EXERCISE_BICEP_CURL
            exercise_frames = 0
        elif is_squat(keypoints[0]):
            current_exercise = EXERCISE_SQUAT
            exercise_frames = 0

    if current_exercise == EXERCISE_BICEP_CURL:
        if not is_bicep_curl(keypoints[0]):
            current_exercise = EXERCISE_IDLE
        else:
            exercise_frames += 1

    if current_exercise == EXERCISE_SQUAT:
        if not is_squat(keypoints[0]):
            current_exercise = EXERCISE_IDLE
        else:
            exercise_frames += 1

    # Evaluate exercise correctness and count repetitions
    if current_exercise == EXERCISE_BICEP_CURL and exercise_frames > 30:
        if is_bicep_curl(keypoints[0]):
            correct_form = True
        elif correct_form:
            repetition_count += 1
            correct_form = False

    if current_exercise == EXERCISE_SQUAT and exercise_frames > 30:
        if is_squat(keypoints[0]):
            correct_form = True
        elif correct_form:
            repetition_count += 1
            correct_form = False

    # Display feedback on the frame
    if current_exercise == EXERCISE_BICEP_CURL:
        feedback_text = f"Bicep Curl Repetitions: {repetition_count}"
    elif current_exercise == EXERCISE_SQUAT:
        feedback_text = f"Squat Repetitions: {repetition_count}"
    else:
        feedback_text = "No Exercise Detected"

    cv2.putText(frame, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Workout Correctness", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
