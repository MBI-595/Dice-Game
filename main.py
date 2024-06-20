from ultralytics import YOLO
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt
import cv2
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


model = YOLO("runs\\detect\\train2\\weights\\best.pt")
imageDataSet = load_images_from_folder('Test\Images')

threshold = 0.7

names = ["1", "2", "3", "4", "5", "6"]

user_score = 0
j = 0
dice1 = 0
dice2 = 0
for i in imageDataSet:
    j += 1
    results = model(i)[0]

    flag = True

    img_score = 0

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(i, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(i, names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            if flag:
                dice1 = int(names[int(class_id)].upper())
            else:
                dice2 = int(names[int(class_id)].upper())

            img_score = img_score + int(names[int(class_id)].upper())

            flag = False

    print("Dice 1:", dice1)
    print("Dice 2:", dice2)

    if dice1 == dice2:
        plt.imshow(i)
        display_score = "Game Over!!! Final Score: " + str(user_score)
        plt.title(display_score)
        plt.show()
        print("Game Over")
        print("Final Score is: ", user_score)
        break

    user_score = user_score + img_score
    print("Score for image", j, "is:", user_score)

    plt.imshow(i)
    display_score = "Score: " + str(user_score)
    plt.title(display_score)
    plt.show()
