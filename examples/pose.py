import cv2
import mediapipe as mp


def display_video(video_path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # Read and display video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        
        # If frame is read correctly ret is True
        if not ret:
            print("End of video file")
            break
            
        # Display the frame
        cv2.imshow('Video Player', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     video_path = "./VID_20250423_195109.mp4"  # Replace with your video file path
#     display_video(video_path)


def display_video_with_pose(video_path):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # Read and display video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("End of video file")
            break
            
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect poses
        results = pose.process(image)
        
        # Convert the image back to BGR for displaying
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        # Display the frame
        cv2.imshow('Pose Detection', image)
        
        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Release resources
    pose.close()
    cap.release()
    cv2.destroyAllWindows()

 


def display_save_video_with_pose(video_path, output_path):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Read and process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("End of video file")
            break
            
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect poses
        results = pose.process(image)
        
        # Convert the image back to BGR for displaying and saving
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        # Write the frame to output video
        out.write(image)
        
        # Display the frame
        cv2.imshow('Pose Detection', image)
        
        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Release resources
    pose.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def display_save_video_with_pose_3d(video_path, output_path):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_segmentation=True,
        model_complexity=2  # Use the most accurate model
    )

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    
    # Create arrays to store 3D coordinates
    landmarks_3d = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("End of video file")
            break
            
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect poses
        results = pose.process(image)
        
        # Extract 3D coordinates if landmarks are detected
        if results.pose_world_landmarks:
            frame_landmarks = []
            for landmark in results.pose_world_landmarks.landmark:
                # Extract x, y, z coordinates (in meters)
                frame_landmarks.append([landmark.x, landmark.y, landmark.z])
            landmarks_3d.append(frame_landmarks)
            
            # Display 3D coordinates for specific joints
            # Example: Display coordinates for nose (landmark 0)
            nose = results.pose_world_landmarks.landmark[0]
            cv2.putText(image, f"Nose 3D: ({nose.x:.2f}, {nose.y:.2f}, {nose.z:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert back to BGR and draw landmarks
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        # Write frame and display
        out.write(image)
        cv2.imshow('3D Pose Detection', image)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Save 3D coordinates to file
    if landmarks_3d:
        np.save('pose_3d_coordinates.npy', np.array(landmarks_3d))
    
    # Release resources
    pose.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #video_path = "./climb.mp4"
    video_path = "./janja1.mp4"
    output_path = "./janja1_with_pose.mp4"
    #display_video_with_pose(video_path)  
    #display_save_video_with_pose(video_path, output_path)    
    display_save_video_with_pose_3d(video_path, output_path)    
