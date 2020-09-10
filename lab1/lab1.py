import cv2

cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 25, (640, 480), isColor=True)

while cap1.isOpened():
    ret, frame = cap1.read()
    if ret:
        out.write(frame)
        cv2.imshow('first window', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
out.release()

cap2 = cv2.VideoCapture('output.mp4', cv2.CAP_ANY)

while cap2.isOpened():

    ret, frame = cap2.read()
    if not ret:
        break
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray_scale, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(gray, (100, 100), (550, 400), (250, 5, 5))
    cv2.line(gray, (5, 5), (600, 200), (0, 250, 0))

    cv2.imshow('second window', gray)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


# Release everything if job is finished
cap1.release()
cap2.release()

cv2.destroyAllWindows()
