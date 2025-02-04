import cv2
import mediapipe as mp
import socket

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Socket Client
def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect(('localhost', 65432))
        return True, client_socket
    except:
        return False, None

def pose_estimation_loop(client_socket):
    is_jumping = False

    while True:
        ret, frame = cap.read()
        if not ret or not client_socket:  # Check if client_socket exists before using it
            break

        frame = cv2.flip(frame, 1)  # Flip webcam horizontally
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result and result.pose_landmarks:
            mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get Hip y-position (for jump detection) - using LEFT_HIP
            hip = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            hip_y = int(hip.y * h)  # Convert to pixel position

            # Define a fixed jump threshold at 1/3 of the frame height
            jump_threshold = int(h / 2)

            # Draw the threshold line
            cv2.line(frame, (0, jump_threshold), (w, jump_threshold), (0, 255, 0), 2)

            # If hip goes above the threshold and was below it previously, trigger jump
            if (hip_y < jump_threshold) and (not is_jumping):
                client_socket.send("jump".encode())  # Send jump command to server
                is_jumping = True
                print("Jump triggered!")

            # Reset jump state when hip returns below the threshold
            elif hip_y >= jump_threshold:
                is_jumping = False

        cv2.putText(frame, "Jump to Control Dino!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam Pose Tracking", frame)

        # Exit Condition for Webcam and Socket
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Connect to server before starting pose estimation
client_socket = None
success, client_socket = connect_to_server()

if success:
    print("Connected to server. Starting pose estimation...")
    pose_estimation_loop(client_socket)
else:
    print("Failed to connect to server.")
