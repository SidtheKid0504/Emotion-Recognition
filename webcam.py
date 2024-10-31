import boto3
import cv2

rekognition = boto3.client(
  'rekognition',
  aws_access_key_id="ACCESS_KEY",
  aws_secret_access_key="SECRET_KEY",
  region_name="us-east-1"
)  

session = boto3.Session()

width = 1280
height = 720
scale_factor = 0.1
text_color = (0, 0, 255)

def find_emotion(emotions):
    num = {'Confidence': 0.0, 'Type': None}
    for item in emotions:
        if item['Confidence'] > num['Confidence']:
            num = item
    return num

def display(frame, responses):
    faces = responses['FaceDetails']

    emotion = ""
    for face in faces:
        emotions = find_emotion(face['Emotions'])
        emotion = emotions['Type']

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    cv2.putText(frame, emotion, (40, 240),
                font, 2, text_color, 1, cv2.LINE_AA)

    window_name = 'Emotional Recognition'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        success, frame = cap.read()
        if success:
            scaled_frame = cv2.resize(
                frame, (int(width * scale_factor), int(height * scale_factor)))

            rval, buffer = cv2.imencode('.jpg', scaled_frame)
            _bytes = bytearray(buffer)
            response = rekognition.detect_faces(Image={'Bytes': _bytes},
                                                    Attributes=['ALL'])
            print(response)
            display(frame, response)
        if cv2.waitKey(20) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
