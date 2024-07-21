from ultralytics import YOLO


if __name__ == "__main__":

    weight = "./weights.pt"
    model = YOLO(weight)

    # 'path to your image'
    img_path = './asset/demo_1_textbook.png'  
    det_res = model.predict(img_path , imgsz=1280, conf=0.25, iou=0.45)[0]
    # print bbox info
    print(det_res.boxes)           
    # visualize the bbox   
    det_res.save(filename=img_path.replace('.png', '_res.png'))     