import boto3  # AWS SDK for Rekognition and S3
import time  # Delay between polling Rekognition job status
import cv2   # OpenCV for video I/O and drawing
import os    # File and path operations
import sys

# Initialize AWS clients for label detection and file storage
rekognition_client = boto3.client('rekognition')
s3_client = boto3.client('s3')

# Configuration: fill in the paths and bucket name for your environment
SOURCE_DIRECTORY = ""  # TODO: set path to your local input videos directory
OUTPUT_DIRECTORY = ""  # TODO: set path to your annotated videos output directory
S3_BUCKET = ""         # TODO: set your S3 bucket name for Rekognition
CONFIDENCE_THRESHOLD = 50.0  # Mid-range threshold to include useful labels

SUPPORTED_FORMATS = ('.mp4', '.avi', '.mov')  # Video file extensions to handle


def list_media_files():
    # Attempt to read all video files from local source directory
    local_files = [f for f in os.listdir(SOURCE_DIRECTORY)
                   if f.lower().endswith(SUPPORTED_FORMATS)]

    if local_files:
        print(f"Found {len(local_files)} files locally.")
        return local_files

    # If none found locally, list from S3 under 'videos/' prefix
    print("No local videos found. Checking S3 bucket...")
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix="videos/")
    s3_keys = [obj['Key'] for obj in response.get('Contents', [])
               if obj['Key'].lower().endswith(SUPPORTED_FORMATS)]
    print(f"Found {len(s3_keys)} files in S3.")
    return s3_keys


def upload_to_s3(local_path, s3_key):
    # Send selected video file to S3 for Rekognition processing
    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"Uploaded to S3: {s3_key}")


def analyze_with_rekognition(s3_key):
    # Start asynchronous Rekognition label detection job
    response = rekognition_client.start_label_detection(
        Video={'S3Object': {'Bucket': S3_BUCKET, 'Name': s3_key}}
    )
    job_id = response['JobId']
    print(f"Label detection started (JobId: {job_id})")

    # Poll until job completes (SUCCEEDED or ERROR)
    while True:
        result = rekognition_client.get_label_detection(JobId=job_id)
        status = result['JobStatus']
        if status != 'IN_PROGRESS':
            break
        time.sleep(5)

    if status != 'SUCCEEDED':
        print("Rekognition job did not succeed.")
        return None

    return result  # Contains labels with timestamps and bounding boxes


def filter_labels(results):
    # Extract label instances meeting confidence threshold
    filtered = []
    for entry in results.get('Labels', []):
        label_name = entry['Label']['Name']
        timestamp = entry.get('Timestamp', 0)
        for instance in entry['Label'].get('Instances', []):
            conf = instance.get('Confidence', 0)
            if conf >= CONFIDENCE_THRESHOLD:
                filtered.append({
                    'Name': label_name,
                    'Confidence': conf,
                    'Timestamp': timestamp,
                    'BoundingBox': instance.get('BoundingBox', {})
                })
    return filtered


def annotate_video(video_name, annotations):
    # Prepare video reader and writer for overlaying annotations
    input_path = os.path.join(SOURCE_DIRECTORY, video_name)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Construct output filename with annotation suffix, ensure uniqueness
    base = os.path.splitext(video_name)[0]
    out_name = f"{base}_annotated.mp4"
    out_path = os.path.join(OUTPUT_DIRECTORY, out_name)
    counter = 1
    while os.path.exists(out_path):
        out_name = f"{base}_annotated_{counter}.mp4"
        out_path = os.path.join(OUTPUT_DIRECTORY, out_name)
        counter += 1

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Organize annotations keyed by frame index for quick lookup
    frames = {}
    for ann in annotations:
        idx = int((ann['Timestamp'] / 1000.0) * fps)
        frames.setdefault(idx, []).append(ann)

    active = []  # Track annotations to draw over a 2-second window
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add new annotations at this frame
        if idx in frames:
            active.extend(frames[idx])

        # Overlay each active annotation as bounding box + label
        for ann in active:
            x = int(ann['BoundingBox']['Left'] * width)
            y = int(ann['BoundingBox']['Top'] * height)
            w = int(ann['BoundingBox']['Width'] * width)
            h = int(ann['BoundingBox']['Height'] * height)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{ann['Name']} ({ann['Confidence']:.1f}%)"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Remove annotations older than 2 seconds for clarity
        active = [a for a in active if idx - int((a['Timestamp'] / 1000.0) * fps) <= fps * 2]

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"Annotated video saved: {out_path}")
    return out_path


def main():
    # Discover videos, select one, then run full analysis+annotation pipeline
    files = list_media_files()
    if not files:
        print("No videos available to process.")
        return

    for i, name in enumerate(files, 1):
        print(f"{i}. {name}")
    choice = int(input("Select a video by number: ")) - 1
    if choice < 0 or choice >= len(files):
        print("Invalid selection.")
        return

    video = files[choice]
    local = os.path.join(SOURCE_DIRECTORY, video)

    # Ensure video is available locally for upload
    if not os.path.exists(local) and video.lower().startswith('videos/'):
        s3_client.download_file(S3_BUCKET, video, local)
        print(f"Downloaded {video} from S3.")

    # Upload and process via Rekognition
    key = f"videos/{os.path.basename(video)}"
    upload_to_s3(local, key)
    result = analyze_with_rekognition(key)
    if not result:
        return

    # Keep only meaningful labels, then overlay them into output video
    annotations = filter_labels(result)
    if not annotations:
        print("No annotations above confidence threshold.")
        return
    annotate_video(os.path.basename(video), annotations)


if __name__ == '__main__':
    main()
