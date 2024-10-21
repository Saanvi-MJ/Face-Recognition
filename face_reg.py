import cv2
import face_recognition

def capture_selfie():
    # Initialize webcam
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Selfie")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Selfie", frame)


        if cv2.waitKey(1) & 0xFF == 13:
            selfie_path = "selfie.jpg"
            cv2.imwrite(selfie_path, frame)
            print(f"Selfie saved as {selfie_path}")
            break

    cam.release()
    cv2.destroyAllWindows()
    return selfie_path


def compare_images(imageA_path, selfie_path):
   
    imageA = face_recognition.load_image_file(imageA_path)
    selfie = face_recognition.load_image_file(selfie_path)

    
    imageA_encoding = face_recognition.face_encodings(imageA)
    selfie_encoding = face_recognition.face_encodings(selfie)

   
    if len(imageA_encoding) == 0 or len(selfie_encoding) == 0:
        print("No face detected in one of the images.")
        return False

    result = face_recognition.compare_faces([imageA_encoding[0]], selfie_encoding[0])

    return result[0]  

def main():
    
    imageA_path = "Dummy.jpg"  
 
    selfie_path = capture_selfie()

    is_same_person = compare_images(imageA_path, selfie_path)

    if is_same_person:
        print("Yes, both images are of the same person.")
    else:
        print("No, the images are of different people.")

if __name__ == "__main__":
    main()
