import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import time 

# --- CÁC THAM SỐ CÀI ĐẶT ---
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
MODEL_PATH = "best_model_augmented.h5" 
# -----------------------------

# 1. TẢI MODEL DỰ ĐOÁN
print(f"Đang tải model từ: {MODEL_PATH}...")
try:
    best_model = tf.keras.models.load_model(MODEL_PATH)
    print("Tải model thành công!")
except Exception as e:
    print(f"LỖI: Không thể tải model. Hãy chắc chắn file '{MODEL_PATH}' tồn tại.")
    print(f"Lỗi chi tiết: {e}")
    exit()

# 2. KHỞI TẠO MEDIAPIPE
print("Đang khởi tạo MediaPipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands( 
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 3. KHỞI TẠO CANVAS VÀ WEBCAM
canvas = np.zeros((WEBCAM_HEIGHT, WEBCAM_WIDTH, 3), dtype="uint8")
print("Đang khởi động webcam...")
cap = cv2.VideoCapture(0)
cap.set(3, WEBCAM_WIDTH)
cap.set(4, WEBCAM_HEIGHT)

if not cap.isOpened():
    print("Lỗi: Không thể mở webcam.")
    exit()

print("Khởi động thành công! Nhấn 'ESC' để thoát.")
print("CỬ CHỈ: [1 ngón trỏ]=VẼ, [Trỏ+Giữa]=XÓA, [1 ngón út]=DỰ ĐOÁN")

# 4. BIẾN TRẠNG THÁI
prev_point = None
prediction_text = ""
prediction_cooldown = 0 

# 5. HÀM TIỀN XỬ LÝ ẢNH (LOGIC TỪ STEP 8)
def preprocess_for_model(img_canvas):
    img_pil = Image.fromarray(cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY))
    
    # 1. LÀM MỎNG NÉT VẼ (Erosion)
    img_eroded = img_pil.filter(ImageFilter.MinFilter(3))

    # 2. Tìm bounding box
    bbox = img_eroded.getbbox()
    if bbox is None:
        return None 
        
    # 3. Cắt (crop)
    img_cropped = img_eroded.crop(bbox)
    
    # 4. Tạo canvas đen hình vuông + thêm padding
    width, height = img_cropped.size
    new_size = max(width, height) + 40 
    new_canvas_pil = Image.new('L', (new_size, new_size), 0) 
    
    # 5. Dán chữ số vào giữa
    paste_x = (new_size - width) // 2
    paste_y = (new_size - height) // 2
    new_canvas_pil.paste(img_cropped, (paste_x, paste_y))
    
    # 6. Resize về 28x28
    img_resized = new_canvas_pil.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 7. Chuyển về NumPy
    img_array = np.array(img_resized)
    
    # 8. Chuẩn hóa
    img_normalized = img_array.astype('float32') / 255.0
    
    # 9. Reshape cho model (1, 28, 28, 1)
    img_reshaped = img_normalized.reshape(1, 28, 28, 1)
    
    return img_reshaped
# --- HẾT HÀM PREPROCESS ---


# 6. VÒNG LẶP XỬ LÝ VIDEO
while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    gesture_text = ""
    
    # 7. LOGIC CỬ CHỈ
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        landmarks = hand_landmarks.landmark
        
        index_tip_y = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        index_pip_y = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        
        middle_tip_y = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        middle_pip_y = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        
        pinky_tip_y = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
        pinky_pip_y = landmarks[mp_hands.HandLandmark.PINKY_PIP].y
        
        is_index_up = index_tip_y < index_pip_y
        is_middle_up = middle_tip_y < middle_pip_y
        is_pinky_up = pinky_tip_y < pinky_pip_y
        
        cx = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1])
        cy = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0])
        
        
        # --- XỬ LÝ CỬ CHỈ ---
        
        if is_index_up and is_middle_up and not is_pinky_up:
            gesture_text = "XOA (Clear)"
            canvas = np.zeros((WEBCAM_HEIGHT, WEBCAM_WIDTH, 3), dtype="uint8")
            prediction_text = "" 
            prev_point = None

        elif is_index_up and not is_middle_up and not is_pinky_up:
            gesture_text = "DANG VE..."
            cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            
            if prev_point is None:
                prev_point = (cx, cy)
            
            cv2.line(canvas, prev_point, (cx, cy), (255, 255, 255), thickness=10)
            prev_point = (cx, cy)

        elif is_pinky_up and not is_index_up and not is_middle_up and prediction_cooldown == 0:
            gesture_text = "DANG DU DOAN..."
            
            processed_img = preprocess_for_model(canvas)
            
            if processed_img is not None:
                probabilities = best_model.predict(processed_img)[0]
                predicted_digit = np.argmax(probabilities)
                confidence = probabilities[predicted_digit] * 100
                
                prediction_text = f"SO: {predicted_digit} ({confidence:.1f}%)"
                prediction_cooldown = 90 
            
            canvas = np.zeros((WEBCAM_HEIGHT, WEBCAM_WIDTH, 3), dtype="uint8")
            prev_point = None

        else:
            if not is_pinky_up: 
                gesture_text = "DI CHUYEN (Nhac but)"
            prev_point = None

    else:
        gesture_text = "KHONG CO BAN TAY"
        prev_point = None

    # 8. GỘP ẢNH VÀ HIỂN THỊ
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_mask = cv2.threshold(canvas_gray, 50, 255, cv2.THRESH_BINARY)
    img_mask_inv = cv2.bitwise_not(img_mask)
    img_bg = cv2.bitwise_and(image, image, mask=img_mask_inv)
    img_fg = cv2.bitwise_and(canvas, canvas, mask=img_mask)
    final_image = cv2.add(img_bg, img_fg)
    
    cv2.putText(final_image, gesture_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    if prediction_text:
        cv2.putText(final_image, prediction_text, (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    if prediction_cooldown > 0:
        prediction_cooldown -= 1
        if prediction_cooldown == 0:
            prediction_text = ""

    cv2.imshow('AR MNIST Predictor', final_image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 9. DỌN DẸP
print("Đang đóng webcam...")
cap.release()
cv2.destroyAllWindows()